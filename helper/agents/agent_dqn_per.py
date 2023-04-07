from typing import Tuple

import chex
import haiku as hk
import numpy as np
import optax
import jax
import jax.numpy as jnp

from helper import compute_features_from_observation
import random


############ Replay Buffer cell #######################################

@chex.dataclass
class Transition:
    state_t: chex.Array
    action_t: chex.Array
    reward_t: chex.Array
    done_t: chex.Array
    state_tp1: chex.Array


# SumTree
# a binary tree data structure where the parent’s value is the sum of its children
# Source of the implementation: https://github.com/rlcode/per/blob/master/SumTree.py
class SumTree:
    write = 0

    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.n_entries = 0

    # update to the root node
    def _propagate(self, idx, change):
        parent = (idx - 1) // 2

        self.tree[parent] += change

        if parent != 0:
            self._propagate(parent, change)

    # find sample on leaf node
    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self):
        return self.tree[0]

    # store priority and sample
    def add(self, p, data):
        idx = self.write + self.capacity - 1

        self.data[self.write] = data
        self.update(idx, p)

        self.write += 1
        if self.write >= self.capacity:
            self.write = 0

        if self.n_entries < self.capacity:
            self.n_entries += 1

    # update priority
    def update(self, idx, p):
        change = p - self.tree[idx]

        self.tree[idx] = p
        self._propagate(idx, change)

    # get priority and sample
    def get(self, s):
        idx = self._retrieve(0, s)
        dataIdx = idx - self.capacity + 1

        return (idx, self.tree[idx], self.data[dataIdx])


class PrioritizedReplayBuffer:
    """Fixed-size buffer to store transition tuples."""

    def __init__(self, buffer_capacity: int):
        """Initialize a ReplayBuffer object.
        Args:
            buffer_capacity (int): maximal number of tuples to store at once
        """
        self._memory = SumTree(buffer_capacity)
        self._maxlen = buffer_capacity
        self._alpha = 0.6
        self._beta = 0.4
        self._beta_increment_per_sampling = 0.001
        self._epsilon = 0.01
        self._max_priority = 1

    def _get_priority(self, error):
        return (np.abs(error) + self._epsilon) ** self._alpha

    @property
    def size(self) -> int:
        # Return the current number of elements in the buffer.
        return self._memory.n_entries

    def add(
            self,
            state_t: chex.Array,
            action_t: chex.Array,
            reward_t: chex.Array,
            done_t: chex.Array,
            state_tp1: chex.Array,
    ) -> None:
        """Add a new transition to memory."""
        priority = self._max_priority
        self._memory.add(
            priority,
            Transition(
                state_t=state_t,
                action_t=action_t,
                reward_t=reward_t,
                done_t=done_t,
                state_tp1=state_tp1,
            )
        )

    def sample(self) -> Transition:
        """Randomly sample a transition from memory."""
        assert self._memory, "replay buffer is unfilled"
        # Your code here !
        index = np.random.randint(self.size)
        return self._memory[index]

    def sample_batch(self, batch_size) -> Tuple[Transition, list, list]:
        """Randomly sample a transition from memory."""
        assert self._memory, "replay buffer is unfilled"
        batch = []
        idxs = []
        segment = self._memory.total() / batch_size
        priorities = []

        # Increment beta
        self._beta = np.min([1., self._beta + self._beta_increment_per_sampling])

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)

            s = random.uniform(a, b)
            (idx, p, transition) = self._memory.get(s)
            priorities.append(p)
            batch.append(transition)
            idxs.append(idx)

        sampling_probabilities = priorities / self._memory.total()
        imp_sampling_weight = np.power(self._memory.n_entries * sampling_probabilities, -self._beta)
        imp_sampling_weight /= imp_sampling_weight.max()

        kwargs = dict()
        for attr in ["state_t", "state_tp1"]:
            kwargs[attr] = np.array([
                compute_features_from_observation(getattr(sample, attr)) for sample in batch
            ])
        for attr in ["action_t", "reward_t", "done_t"]:
            kwargs[attr] = np.array([
                getattr(sample, attr) for sample in batch
            ])
        batch_transitions = Transition(**kwargs)
        return batch_transitions, idxs, imp_sampling_weight

    def update(self, idx, error):
        priority = self._get_priority(error)
        self._memory.update(idx, priority)
        self._max_priority = max(self._max_priority, priority)

############ Network cell ##############################################
def network(x: chex.Array, n_actions: int, hidden_dim: int) -> chex.Array:
    out = hk.nets.MLP([hidden_dim, n_actions], activation=jax.nn.relu)(x)
    return out


############ Agent cell ###################################################
@chex.dataclass
class LearnerState:
    online_params: hk.Params
    target_params: hk.Params
    opt_state: optax.OptState


class DqnPerAgent:
    def __init__(
            self,
            env,
            gamma: float,
            eps: float,
            learning_rate: float,
            buffer_capacity: int,
            min_buffer_capacity: int,
            batch_size: int,
            target_ema: float,
            network_hdim: int,
            foreseen_bars: int = 2,
            seed: int = 0,
    ) -> None:
        """Initializes the DQN agent.

        Args:
          env: input catch environment.
          gamma: discount factor
          eps: probability to perform a random exploration when picking a new action.
          learning_rate: learning rate of the online network
          buffer_capacity: capacity of the replay buffer
          min_buffer_capacity: min buffer size before picking batches from the
            replay buffer to update the online network
          batch_size: batch size when updating the online network
          target_ema: weight when updating the target network.
          seed: seed of the random generator.
        """
        self._env = env
        self._learning_rate = learning_rate
        self._eps = eps
        self._gamma = gamma
        self._batch_size = batch_size
        self._target_ema = target_ema
        self._Na = env.N_ACTIONS
        self._network_hdim = network_hdim
        self._foreseen_bars = foreseen_bars

        # track the visit of each state
        # The keys are the hashed states, the values are the number of times we visited it.
        self._visits = {}

        # Define the neural network for this agent
        self._init, self._apply = hk.without_apply_rng(hk.transform(self._hk_qfunction))
        # Jit the forward pass of the neural network for better performances
        self.apply = jax.jit(self._apply)

        # Also jit the update functiom
        self._update_fn = jax.jit(self._update_fn)

        # Initialize the network's parameters
        self._rng = jax.random.PRNGKey(seed)
        self._rng, init_rng = jax.random.split(self._rng)
        self._learner_state = self._init_state(init_rng)

        # Initialize the replay buffer
        self._min_buffer_capacity = min_buffer_capacity
        self._buffer = PrioritizedReplayBuffer(buffer_capacity)

        # Build a variable to store the last state observed by the agent
        self._state = None

    def _optimizer(self) -> optax.GradientTransformation:
        return optax.adam(learning_rate=self._learning_rate)

    def _hk_qfunction(self, state: chex.Array) -> chex.Array:
        return network(state, self._Na, self._network_hdim)

    def first_observe(self, state: chex.Array) -> None:
        self._state = state

    def _init_state(self, rng: chex.PRNGKey) -> LearnerState:
        """Initialize the online parameters, the target parameters and the
        optimizer's state."""
        dummy_step = compute_features_from_observation(self._env.reset(), self._foreseen_bars)[None]

        online_params = self._init(rng, dummy_step)
        target_params = online_params
        opt_state = self._optimizer().init(online_params)

        return LearnerState(
            online_params=online_params,
            target_params=target_params,
            opt_state=opt_state,
        )

    def sample_action(
            self,
            state: chex.Array,
            evaluation: bool,
    ) -> chex.Array:
        """Picks the next action using an epsilon greedy policy.

        Args:
          state: observed state.
          eval: if True the agent is acting in evaluation mode (which means it only
            acts according to the best policy it knows.)
        """
        # Fill in this function to act using an epsilon-greedy policy.
        if not evaluation and np.random.uniform() < self._eps:
            return np.random.randint(self._Na)
        state_features = compute_features_from_observation(state, foreseen_bars=self._foreseen_bars)
        return np.argmax(self._apply(self._learner_state.online_params, state_features[None]))

    def td_error(
            self,
            online_params: hk.Params,
            target_params: hk.Params,
            state_t: chex.Array,
            action_t: chex.Array,
            reward_t: chex.Array,
            done_t: chex.Array,
            state_tp1: chex.Array,
    ) -> Tuple[chex.Array]:
        # Step one: compute the target Q-value for state t+1
        q_tp1 = self._apply(target_params, state_tp1)

        # We do not want to consider the Q-value of states that are done !
        # For theses states, q(t+1) = 0
        q_tp1 = (1.0 - done_t[..., None]) * q_tp1

        # Now deduce the value of the target cumulative reward
        y_t = reward_t + self._gamma * jnp.max(q_tp1, axis=1)  # Shape B

        # Compute the online Q-value for state t
        q_t = self._apply(online_params, state_t)  # Shape B , Na

        # Ok, but we only want the Q value for the actions that have actually
        # been played
        q_at = jax.vmap(lambda idx, q: q[idx])(action_t, q_t)

        # Compute the square error
        td_error = (q_at - y_t)
        return td_error

    def loss_fn(
            self,
            online_params: hk.Params,
            target_params: hk.Params,
            state_t: chex.Array,
            action_t: chex.Array,
            reward_t: chex.Array,
            done_t: chex.Array,
            state_tp1: chex.Array,
    ) -> chex.Array:
        """Computes the Q-learning loss

        Args:
          online_params: parameters of the online network
          target_params: parameters of the target network
          state_t: batch of observations at time t
          action_t: batch of actions performed at time t
          reward_t: batch of rewards obtained at time t
          done_t: batch of end of episode status at time t
          state_tp1: batch of states at time t+1
        Returns:
          The Q-learning loss.
        """

        # Get TD error
        td_error = self.td_error(
            online_params,
            target_params,
            state_t,
            action_t,
            reward_t,
            done_t,
            state_tp1
        )
        # Derive the loss
        loss = jnp.mean(td_error ** 2)
        return loss

    def _update_fn(
            self,
            state: LearnerState,
            batch: Transition,
    ) -> Tuple[chex.Array, LearnerState]:
        """Get the next learner state given the current batch of transitions.

        Args:
          state: learner state before update.
          batch: batch of experiences (st, at, rt, done_t, stp1)
        Returns:
          loss, learner state after update
        """
        # Compute gradients
        loss, gradients = jax.value_and_grad(self.loss_fn)(
            state.online_params,
            state.target_params,
            batch.state_t,
            batch.action_t,
            batch.reward_t,
            batch.done_t,
            batch.state_tp1,
        )

        # Apply gradients
        updates, new_opt_state = self._optimizer().update(gradients, state.opt_state)
        new_online_params = optax.apply_updates(state.online_params, updates)

        # Update target network params as:
        # target_params <- ema * target_params + (1 - ema) * online_params
        new_target_params = jax.tree_map(
            lambda x, y: x + (1 - self._target_ema) * (y - x),
            state.target_params,
            new_online_params,
        )

        # Compute TD error for Prioritized Experience Replay
        td_error = self.td_error(
            state.online_params,
            state.target_params,
            batch.state_t,
            batch.action_t,
            batch.reward_t,
            batch.done_t,
            batch.state_tp1
        )

        return loss, td_error, LearnerState(
            online_params=new_online_params,
            target_params=new_target_params,
            opt_state=new_opt_state,
        )

    def observe(
            self,
            action_t: chex.Array,
            reward_t: chex.Array,
            done_t: chex.Array,
            state_tp1: chex.Array,
    ) -> chex.Array:
        """Updates the agent from the given observations.

        Args:
          action_t: action performed at time t.
          reward_t: reward obtained after having performed action_t.
          done_t: whether or not the episode is over after performing action_t.
          state_tp1: state at which the environment is at time t+1.
        Returns:
          DQN loss obtained when updating the online network.
        """
        # First, we need to add the new transition to the memory buffer
        # Exploration bonus for Big Maze
        self._buffer.add(self._state, action_t, reward_t, done_t, state_tp1)
        self._state = state_tp1

        # We update the agent if and only if we have enought state stored in
        # memory.
        if self._buffer.size >= self._min_buffer_capacity:
            batch, idxs, imp_sampling_weight = self._buffer.sample_batch(self._batch_size)
            loss, td_errors, self._learner_state = self._update_fn(self._learner_state, batch)

            # Update priorities
            for idx, td_error in zip(idxs, td_errors):
                self._buffer.update(idx, td_error)

            return loss
        return 0.0

    def compute_q_value_from_observation(self, observation) -> chex.Array:
        """Compute the q value vector from a single observation

        Args:
            observation: observation (unpreprocessed) of the environment. May be obtained through `env.reset()`

        Returns:
            chex.Array: q vector of length the number of actions
        """
        features = compute_features_from_observation(observation, self._foreseen_bars)
        # q value: update if using other kind of agent
        q = self.apply(self._learner_state.online_params, features)
        return q
