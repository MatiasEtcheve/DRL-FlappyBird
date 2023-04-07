from collections import deque
from typing import Tuple, Optional
import chex
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import optax
from helper import compute_features_from_observation


def cumulative_returns(rewards: chex.Array,
                       dones: chex.Array,
                       gamma: float,
                       ) -> chex.Array:
    """Computes the cumulative returns for the given serie of rewards.

    Args:
    rewards: an array of shape (T,) where reward[t] is the reward at time t.
    dones: an array of shape (T,) where dones[t] is 1 is the episode is over at
      time t and 0 otherwise.
    gamma: discount factor.
    Returns:
    An array C of shape (T,) where C[t] is the cumulative return at time t.
    """
    # Your code here !
    returns = [0.]
    belongs_to_an_unfinished_episode = [1.]
    for r, d in zip(rewards[::-1], dones[::-1]):
        returns.append(r + gamma * (1 - d) * returns[-1])
        belongs_to_an_unfinished_episode.append((1 - d) * belongs_to_an_unfinished_episode[-1])
    does_not_belong_to_an_unfinished_episode = 1 - jnp.stack(belongs_to_an_unfinished_episode)[::-1][:-1]
    return jnp.stack(returns)[::-1][:-1] * does_not_belong_to_an_unfinished_episode


def policy_network(x: chex.Array, hdim: int = 128)-> chex.Array:
    return hk.nets.MLP(output_sizes=[hdim, 2])(x)


@chex.dataclass
class LearnerState:
    params: hk.Params
    opt_state: optax.OptState


class REINFORCEAgent:
    def __init__(self,
                 env,
                 gamma: float,
                 learning_rate: float,
                 steps_between_updates: int,
                 seed: int = 0,
                 foreseen_bars: int = 2,
                 hdim: int = 128
                 ) -> None:
        """Constructor.

        Args:
          env: input Catch environment.
          gamma: discount factor.
          learning_rate: learning rate when training the neural network.
          steps_between_updates: number of step between each update of the network.
          seed: seed of the random generator.
        """
        # Basic parameters
        self._env = env
        self._Na = env.N_ACTIONS
        self._learning_rate = learning_rate
        self._gamma = gamma

        self._foreseen_bars = foreseen_bars
        self._hdim = hdim

        # The agent is not updated at every step.
        self._steps_between_updates = steps_between_updates
        self._step = 0

        # Initialize the random generator
        self._rng = jax.random.PRNGKey(seed)
        self._rng, init_rng = jax.random.split(self._rng)

        # Initialize the network function
        self._init, self._apply = hk.without_apply_rng(hk.transform(self._hk_policy_function))
        # Jit both the forward pass and the update function for more efficiency
        self.apply = jax.jit(self._apply)
        self._update_fn = jax.jit(self._update_fn)

        # Initialize the parameters of the neural network as well as the optimizer
        self._learner_state = self._init_state(init_rng)

        # Small buffers to store trajectories
        self._states = deque([], steps_between_updates + 1)
        self._actions = deque([], steps_between_updates + 1)
        self._rewards = deque([], steps_between_updates + 1)
        self._dones = deque([], steps_between_updates + 1)

    def _optimizer(self) -> optax.GradientTransformation:
        return optax.adam(learning_rate=self._learning_rate)

    def _hk_policy_function(self, state: chex.Array) -> chex.Array:
        return policy_network(state, hdim=self._hdim)

    def observe(self,
                state_t: chex.Array,
                action_t: chex.Array,
                reward_t: chex.Array,
                done_t: chex.Array,
                ) -> Optional[chex.Array]:
        """Observes the current transition and updates the network if necessary.

        Args:
          state_t: state observed at time t.
          action_t: action performed at time t.
          reward_t: reward obtained after performing  action_t.
          done_t: wether or not the episode is over after the step.
        Returns:
          The training loss if the model was updated, None elsewhere.
        """

        features_t = compute_features_from_observation(state_t, foreseen_bars=self._foreseen_bars)
        self._states.append(features_t)
        self._actions.append(action_t)
        self._rewards.append(reward_t)
        self._dones.append(done_t)
        self._step += 1

        do_update = self._step % self._steps_between_updates
        do_update = do_update and self._step >= self._steps_between_updates + 1

        if do_update:
            states = np.stack(self._states, axis=0)
            actions = np.stack(self._actions, axis=0)[:-1]
            rewards = np.stack(self._rewards, axis=0)[:-1]
            dones = np.stack(self._dones, axis=0)[:-1]

            loss, self._learner_state = self._update_fn(self._learner_state,
                                                        states,
                                                        actions,
                                                        rewards,
                                                        dones)
            return loss
        return None

    def _init_state(self, rng: chex.PRNGKey) -> LearnerState:
        """Initializes the parameter of the neural network and the optimizer."""
        state =  compute_features_from_observation(self._env.reset(), foreseen_bars=self._foreseen_bars)
        bstate = state[None]
        params = self._init(rng, bstate)
        opt_state = self._optimizer().init(params)
        return LearnerState(params=params, opt_state=opt_state)

    def act(self,
            state: chex.Array,
            eval: bool,
            ) -> chex.Array:
        """Pick the next action according to the learnt policy."""
        features = compute_features_from_observation(state, foreseen_bars=self._foreseen_bars)
        bstate = features[None]
        logits = self.apply(self._learner_state.params, bstate)[0]
        if eval:
            return jnp.argmax(logits)
        self._rng, rng = jax.random.split(self._rng)
        return jax.random.categorical(rng, logits)

    def loss_fn(self,
                params: hk.Params,
                states: chex.Array,
                actions: chex.Array,
                rewards: chex.Array,
                dones: chex.Array,
                ) -> chex.Array:
        """Compute the loss function for REINFORCE.

        Args:
        params: network parameters.
        states: a tensor of shape (T+1, N_rows, N_cols) representing the states
          observed from time 0 to T. N_rows, and N_cols are respectively the
          number of rows and columns in the catch environment.
        actions: a tensor of shape (T,) giving the actions performed from time
          0 to T-1.
        rewards: a tensor of shape (T,) giving the rewards obtained from time
          0 to T-1.
        dones: a tensor of shape (T,) giving the 'end of episode status' from
        time 0 to T-1.
        Returns:
        training_loss
        """
        # Your code here !
        log_probs = jax.nn.log_softmax(self._apply(params, states), axis=-1) # (T + 1, A)
        log_probs_actions = jax.vmap(lambda l, a: l[a])(log_probs[:-1], actions) # (T,)

        cs = cumulative_returns(rewards, dones, self._gamma) # (T,)

        return -jnp.mean(log_probs_actions * cs)

    def _update_fn(self,
                   learner_state: LearnerState,
                   states: chex.Array,
                   actions: chex.Array,
                   rewards: chex.Array,
                   dones: chex.Array,
                   ) -> Tuple[chex.Array, LearnerState]:
        """Updates the network.

        Args:
          learner_state: network and optimizer parameters.
          states: a tensor of shape (T+1, N_rows, N_cols) representing the states
            observed from time 0 to T. N_rows, and N_cols are respectively the
            number of rows and columns in the catch environment.
          actions: a tensor of shape (T,) giving the actions performed from time
            0 to T-1.
          rewards: a tensor of shape (T,) giving the rewards obtained from time
            0 to T-1.
          dones: a tensor of shape (T,) giving the 'end of episode status' from time
            0 to T-1.
        Returns:
          training_loss, next network ann optimizer's parameters
        """
        loss, grad = jax.value_and_grad(self.loss_fn)(learner_state.params, states, actions, rewards, dones)
        updates, new_opt_state = self._optimizer().update(grad, learner_state.opt_state)
        new_params = optax.apply_updates(learner_state.params, updates)

        return loss, LearnerState(params=new_params, opt_state=new_opt_state)
