import chex
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import optax
from collections import deque
from typing import *
from helper import compute_features_from_observation


# Function to make A2C work
def one_step_temporal_difference(values: chex.Array,
                                 rewards: chex.Array,
                                 dones: chex.Array,
                                 gamma: float,
                                 ) -> chex.Array:
    """Computes the bootstrapped cumulative returns.

    Args:
    values: an array of shape (T+1,) where values[t] is the value estimated for
      the state at time t.
    rewards: an array of shape (T,) where reward[t] is the reward at time t.
    dones: an array of shape (T,) where dones[t] is 1 is the episode is over at
      time t and 0 otherwise.
    gamma: discount factor.
    Returns:
    A vector C of shape (T,) where C[t] is the bootstrapped cumulative returns
    at time t.
    """
    return rewards + gamma * jax.lax.stop_gradient(values[1:]*(1 - dones)) - values[:-1]


# A2C requires 2 neural networks

def policy_network(x: chex.Array, hdim1: int = 128, hdim2: int = 64) -> chex.Array:
    return hk.nets.MLP(output_sizes=[hdim1, hdim2, 2])(x)

def value_network(x: chex.Array, hdim: int = 128):
    return hk.nets.MLP(output_sizes=[hdim, 1])(x)[..., 0] # [B,]


@chex.dataclass
class LearnerState:
    policy_params: hk.Params
    value_params: hk.Params
    policy_opt_state: optax.OptState
    value_opt_state: optax.OptState


class A2CAgent:
    def __init__(
            self,
            env,
            gamma: float,
            learning_rate: float,
            steps_between_updates: int,
            seed: int = 0,
            foreseen_bars: int = 2,
            hdim_policy_1: int = 128,
            hdim_policy_2: int = 64,
            hdim_value: int = 128,
    ) -> None:

        self._env = env
        self._learning_rate = learning_rate
        self._gamma = gamma
        self._Na = env.N_ACTIONS

        self._foreseen_bars = foreseen_bars

        # Do not update at each step
        self._steps_between_updates = steps_between_updates
        self._step = 0
        self._hdim1 = hdim_policy_1
        self._hdim2 = hdim_policy_2
        self._hdim = hdim_value

        # Random generator
        self._rng = jax.random.PRNGKey(seed)
        self._rng, init_rng = jax.random.split(self._rng)

        # Build the policy and value networks
        self._policy_init, self._policy_apply = hk.without_apply_rng(hk.transform(self._hk_policy_function))
        self._value_init, self._value_apply = hk.without_apply_rng(hk.transform(self._hk_value_function))

        # Jit the forward and update functions for more efficiency
        self.policy_apply = jax.jit(self._policy_apply)
        self.value_apply = jax.jit(self._value_apply)
        self._update_fn = jax.jit(self._update_fn)

        # Initialize the networks and optimizer's parameters
        self._learner_state = self._init_state(init_rng)

        # Small buffers to store trajectories
        self._states = deque([], steps_between_updates + 1)
        self._actions = deque([], steps_between_updates + 1)
        self._rewards = deque([], steps_between_updates + 1)
        self._dones = deque([], steps_between_updates + 1)


    def _hk_policy_function(self, state: chex.Array) -> chex.Array:
        return policy_network(state, hdim1=self._hdim1, hdim2=self._hdim2)

    def _hk_value_function(self, state: chex.Array) -> chex.Array:
        return value_network(state, hdim=self._hdim)

    def _optimizer(self) -> optax.GradientTransformation:
        return optax.adam(learning_rate=self._learning_rate)

    def first_observe(self, state: chex.Array) -> None:
        pass
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
        # /!\ READ THIS PART CAREFULLY IN CONJUNCTION WITH THE TRAINING LOOP /!\
        # /!\ IT IS IMPORTANT TO KEEP ALL THINGS IN THE SAME ORDER IN THE    /!\
        # /!\ BUFFERS, AND TO NOT MESS UP THE INDICES (NOTABLY THE FIRST).   /!\


        states_features_t = compute_features_from_observation(state_t, foreseen_bars=self._foreseen_bars)

        self._states.append(states_features_t)
        self._actions.append(action_t)
        self._rewards.append(reward_t)
        self._dones.append(done_t)
        self._step += 1

        if self._step %  self._steps_between_updates and self._step >= self._steps_between_updates + 1:
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
        """Initializea the parameters of the policy network, the value network and
        the optimizer.
        """
        # Your code here
        state = compute_features_from_observation(self._env.reset(), foreseen_bars=self._foreseen_bars)
        bstate = state[None]
        policy_params = self._policy_init(rng, bstate)
        value_params = self._value_init(rng, bstate)
        policy_opt_state = self._optimizer().init(policy_params)
        value_opt_state = self._optimizer().init(value_params)
        return LearnerState(policy_params=policy_params, policy_opt_state=policy_opt_state, value_params=value_params, value_opt_state=value_opt_state)


    def sample_action(self,
                      state: chex.Array,
                      eval: bool,
                      ) -> chex.Array:
        """Pick the next action according to the learnt poslicy."""
        # Your code here
        state_features = compute_features_from_observation(state, foreseen_bars=self._foreseen_bars)
        bstate = state_features[None]
        logits = self.policy_apply(self._learner_state.policy_params, bstate)[0]
        if eval:
            return jnp.argmax(logits)
        self._rng, rng = jax.random.split(self._rng)
        return jax.random.categorical(rng, logits)


    def losses_fn(self,
                  policy_params: hk.Params,
                  value_params: hk.Params,
                  states: chex.Array,
                  actions: chex.Array,
                  rewards: chex.Array,
                  dones: chex.Array,
                  ) -> Tuple[chex.Array, chex.Array]:
        """Compute the policy and value losses.

        Args:
          policy_params: parameters of the policy network.
          value_params: parameters of the value network.
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
          policy_loss, value_loss
        """
        # Your code here !
        log_probs = jax.nn.log_softmax(self._policy_apply(policy_params, states), axis=-1) # (T + 1, A)
        log_probs_actions = jax.vmap(lambda l, a: l[a])(log_probs[:-1], actions) # (T,)

        values = self._value_apply(value_params, states)

        advantages = one_step_temporal_difference(values, rewards, dones, self._gamma)
        value_loss = jnp.mean(advantages ** 2)
        policy_loss = -jnp.mean(log_probs_actions * advantages)

        return policy_loss, value_loss

    def _update_fn(self,
                   learner_state: LearnerState,
                   states: chex.Array,
                   actions: chex.Array,
                   rewards: chex.Array,
                   dones: chex.Array,
                   ) -> Tuple[chex.Array, LearnerState]:
        """Update the policy and the value networks.

        Args:
          learner_state: networks and optimizer parameters.
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
        # Your code here !
        def _policy_loss(*args, **kwargs):
            return self.losses_fn(*args, **kwargs)[0]

        def _value_loss(*args, **kwargs):
            return self.losses_fn(*args, **kwargs)[1]

        args = [learner_state.policy_params, learner_state.value_params, states, actions, rewards, dones]

        policy_loss, policy_grad = jax.value_and_grad(_policy_loss)(*args)
        value_loss, value_grad = jax.value_and_grad(_value_loss, argnums=1)(*args)
        policy_updates, new_policy_opt_state = self._optimizer().update(policy_grad, learner_state.policy_opt_state)
        value_updates, new_value_opt_state = self._optimizer().update(value_grad, learner_state.value_opt_state)

        new_policy_params = optax.apply_updates(learner_state.policy_params, policy_updates)
        new_value_params = optax.apply_updates(learner_state.value_params, value_updates)

        return policy_loss + value_loss, LearnerState(policy_params=new_policy_params, policy_opt_state=new_policy_opt_state, value_params=new_value_params, value_opt_state=new_value_opt_state)

