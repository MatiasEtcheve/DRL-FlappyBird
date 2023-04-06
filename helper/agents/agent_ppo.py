from collections import deque
from typing import *

import chex
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import optax
from deep_rl.environments.flappy_bird import Bar


def split_bars_by_activity(bird, bars) -> Tuple[List[Bar], List[Bar]]:
    """
    Returns [active bars], [inactive bars]
    The active bars are sorted by X_LEFT
    """
    if len(bars) == 0:
        return [], []
    x_bird, y_bird, v_bird = bird
    bars = np.array(bars, dtype="object")
    idx_active_bars = np.array(
        [
            True if x_right_bar > x_bird else False
            for (x_left_bar, x_right_bar, _, _) in bars
        ]
    )

    return sorted(bars[idx_active_bars], key=lambda bar: bar[0]), bars[~idx_active_bars]


def split_bar_by_position(bars) -> Tuple[List[Bar], List[Bar]]:
    """
    Returns list of [bottom bars], [top bars]
    """
    if len(bars) == 0:
        return [], []
    bars = np.array(bars, dtype="object")
    idx_bottom_bars = np.array([position for (_, _, _, position) in bars])
    return bars[idx_bottom_bars], bars[~idx_bottom_bars]


def compute_bird_bar_distance(bird, bars: List[Bar]):
    """
    see Amric
    """
    x_bird, y_bird, v_bird = bird
    active_bars, _ = split_bars_by_activity(bird, bars)
    bottom_active_bars, top_active_bars = split_bar_by_position(active_bars)
    closest_bar_top = (
        (1, 1 + 0.05, 0, False) if len(top_active_bars) == 0 else top_active_bars[0]
    )
    closest_bar_bottom = (
        (1, 1 + 0.05, 0, True)
        if len(bottom_active_bars) == 0
        else bottom_active_bars[0]
    )
    dx_top = closest_bar_top[0] - x_bird
    dx_bottom = closest_bar_bottom[0] - x_bird

    dy_top = 1 - y_bird - closest_bar_top[2]
    dy_bottom = y_bird - closest_bar_bottom[2]

    return (dx_top, dy_top, dx_bottom, dy_bottom)

def compute_features_from_observation(observation):
    bird, bars = observation
    x_bird, y_bird, v_bird = bird
    distances = compute_bird_bar_distance(bird, bars)
    features = np.array([y_bird, v_bird] + list(distances), dtype=np.float32)
    return features

def seek_distance_to_next_bar(bird, bars):
    x_bird, y_bird, v_bird = bird
    active_bars, _ = split_bars_by_activity(bird, bars)
    bottom_active_bars, top_active_bars = split_bar_by_position(active_bars)

    idx_infront_top_bars = np.array(
        [1 - y_bird - bar[2] <= 0 for bar in top_active_bars]
    )
    in_front_top_bar = (
        top_active_bars[idx_infront_top_bars][0]
        if idx_infront_top_bars.sum() > 0
        else (1, 1 + 0.05, 0, False)
    )
    dx_top = in_front_top_bar[0] - x_bird

    idx_infront_bottom_bars = np.array(
        [y_bird - bar[2] <= 0 for bar in bottom_active_bars]
    )
    in_front_bottom_bar = (
        bottom_active_bars[idx_infront_bottom_bars][0]
        if idx_infront_bottom_bars.sum() > 0
        else (1, 1 + 0.05, 0, True)
    )
    dx_bottom = in_front_bottom_bar[0] - x_bird
    return min(dx_top, dx_bottom)

def split_bars_by_activity(bird, bars) -> Tuple[List[Bar], List[Bar]]:
    """
    Returns [active bars], [inactive bars]
    The active bars are sorted by X_LEFT
    """
    if len(bars) == 0:
        return [], []
    x_bird, y_bird, v_bird = bird
    bars = np.array(bars, dtype="object")
    idx_active_bars = np.array(
        [
            True if x_right_bar > x_bird else False
            for (x_left_bar, x_right_bar, _, _) in bars
        ]
    )

    return sorted(bars[idx_active_bars], key=lambda bar: bar[0]), bars[~idx_active_bars]


def split_bar_by_position(bars) -> Tuple[List[Bar], List[Bar]]:
    """
    Returns list of [bottom bars], [top bars]
    """
    if len(bars) == 0:
        return [], []
    bars = np.array(bars, dtype="object")
    idx_bottom_bars = np.array([position for (_, _, _, position) in bars])
    return bars[idx_bottom_bars], bars[~idx_bottom_bars]


def compute_bird_bar_distance(bird, bars: List[Bar]):
    """
    see Amric
    """
    x_bird, y_bird, v_bird = bird
    active_bars, _ = split_bars_by_activity(bird, bars)
    bottom_active_bars, top_active_bars = split_bar_by_position(active_bars)
    closest_bar_top = (
        (1, 1 + 0.05, 0, False) if len(top_active_bars) == 0 else top_active_bars[0]
    )
    closest_bar_bottom = (
        (1, 1 + 0.05, 0, True)
        if len(bottom_active_bars) == 0
        else bottom_active_bars[0]
    )
    dx_top = closest_bar_top[0] - x_bird
    dx_bottom = closest_bar_bottom[0] - x_bird

    dy_top = 1 - y_bird - closest_bar_top[2]
    dy_bottom = y_bird - closest_bar_bottom[2]

    return (dx_top, dy_top, dx_bottom, dy_bottom)

def compute_features_from_observation(observation):
    bird, bars = observation
    x_bird, y_bird, v_bird = bird
    distances = compute_bird_bar_distance(bird, bars)
    features = np.array([y_bird, v_bird] + list(distances), dtype=np.float32)
    return features

def seek_distance_to_next_bar(bird, bars):
    x_bird, y_bird, v_bird = bird
    active_bars, _ = split_bars_by_activity(bird, bars)
    bottom_active_bars, top_active_bars = split_bar_by_position(active_bars)

    idx_infront_top_bars = np.array(
        [1 - y_bird - bar[2] <= 0 for bar in top_active_bars]
    )
    in_front_top_bar = (
        top_active_bars[idx_infront_top_bars][0]
        if idx_infront_top_bars.sum() > 0
        else (1, 1 + 0.05, 0, False)
    )
    dx_top = in_front_top_bar[0] - x_bird

    idx_infront_bottom_bars = np.array(
        [y_bird - bar[2] <= 0 for bar in bottom_active_bars]
    )
    in_front_bottom_bar = (
        bottom_active_bars[idx_infront_bottom_bars][0]
        if idx_infront_bottom_bars.sum() > 0
        else (1, 1 + 0.05, 0, True)
    )
    dx_bottom = in_front_bottom_bar[0] - x_bird
    return min(dx_top, dx_bottom)


#########################################################################################################

def value_network(x: chex.Array):
    return hk.nets.MLP(output_sizes=[32, 1])(x)[..., 0]  # [B,]


def policy_network(x: chex.Array) -> chex.Array:
    return hk.nets.MLP(output_sizes=[32, 2])(x)  # [B, number_of_actions]


def dynamics_network(state_t: chex.Array, action_t: chex.Array):
    """ This network estimates the transition dynamics of the environement: P(s_tp1|s_t,a_t).
    Similarly to what was done in https://arxiv.org/abs/1708.02596 we do NOT directly predict the next
    state s_tp1. Instead we predict the difference s_tp1-s_t as it is easier to learn.
    """
    state_dim = state_t.shape[1]
    next_state = hk.Linear(state_dim)(state_t)
    # print('action_t.shape: ', action_t.shape, 'state_t.shape: ', state_t.shape )
    # reward = hk.nets.MLP(output_sizes=[16, 1])(jnp.concatenate([action_t, state_t, next_state], axis=1))
    return next_state


ARGS = {'gae_lambda': 0.95,
        'update_epochs': 4,
        'clip_coef': 0.2,
        'ent_coef': 0.01,
        'vf_coef': 0.5,
        'update_epochs': 2,
        'minibatch_size': 16,
        'batch_size': 16,
        'max_grad_norm': 0.5
        }


@chex.dataclass
class LearnerState:
    policy_params: hk.Params
    value_params: hk.Params
    dynamics_params: hk.Params
    policy_opt_state: optax.OptState
    value_opt_state: optax.OptState
    dynamics_opt_state: optax.OptState


class PPOAgent:

    def __init__(
            self,
            env,
            gamma: float,
            learning_rate: float,
            steps_between_updates: int,
            seed: int = 0,
            PPO_supplementary_ARGS: dict = ARGS,
    ) -> None:

        self._env = env
        self._learning_rate = learning_rate
        self._gamma = gamma
        self._Na = env.N_ACTIONS
        self.clip_coef = ARGS['clip_coef']
        self.ARGS = PPO_supplementary_ARGS
        # Do not update at each step
        self._steps_between_updates = steps_between_updates
        self._step = 0

        # Random generator
        self._rng = jax.random.PRNGKey(seed)
        self._rng, init_rng = jax.random.split(self._rng)

        # Build the policy and value networks
        self._policy_init, self._policy_apply = hk.without_apply_rng(hk.transform(self._hk_policy_function))
        self._value_init, self._value_apply = hk.without_apply_rng(hk.transform(self._hk_value_function))
        self._dynamics_init, self._dynamics_apply = hk.without_apply_rng(hk.transform(self._hk_dynamics_function))

        # Jit the forward and update functions for more efficiency
        self._policy_apply = jax.jit(self._policy_apply)
        self._value_apply = jax.jit(self._value_apply)
        self._update_fn = jax.jit(self._update_fn)
        self._dynamics_apply = jax.jit(self._dynamics_apply)

        # Initialize the networks and optimizer's parameters
        self._learner_state = self._init_state(init_rng)

        # Small buffers to store trajectories
        self.coef_buffer = 3
        self._states = deque([], self.coef_buffer * steps_between_updates + 1)
        self._actions = deque([], self.coef_buffer * steps_between_updates + 1)
        self._rewards = deque([], self.coef_buffer * steps_between_updates + 1)
        self._dones = deque([], self.coef_buffer * steps_between_updates + 1)
        self._logprobs = deque([], self.coef_buffer * steps_between_updates + 1)
        self._values = deque([], self.coef_buffer * steps_between_updates + 1)
        self.logs = {}

    def _hk_policy_function(self, state: chex.Array) -> chex.Array:
        return policy_network(state)

    def _hk_value_function(self, state: chex.Array) -> chex.Array:
        return value_network(state)

    def _hk_dynamics_function(self, state: chex.Array, action: chex.Array) -> chex.Array:
        return dynamics_network(state, action)

    def _optimizer(self) -> optax.GradientTransformation:
        return optax.chain(optax.clip_by_global_norm(self.ARGS['max_grad_norm']),
                           optax.adam(learning_rate=self._learning_rate))

    def _init_state(self, rng: chex.PRNGKey) -> LearnerState:
        """Initializea the parameters of the policy network, the value network and
        the optimizer.
        """
        # Your code here
        state = compute_features_from_observation(self._env.reset())
        bstate = state[None]
        baction = np.zeros(1)[None]
        policy_params = self._policy_init(rng, bstate)
        value_params = self._value_init(rng, bstate)
        # dynamics_params = self._dynamics_init(rng, bstate, baction)
        policy_opt_state = self._optimizer().init(policy_params)
        value_opt_state = self._optimizer().init(value_params)
        # dynamics_opt_state = self._optimizer().init(dynamics_params)
        return LearnerState(policy_params=policy_params,
                            policy_opt_state=policy_opt_state,
                            value_params=value_params,
                            value_opt_state=value_opt_state,
                            dynamics_params=None,
                            dynamics_opt_state=None)

    def observe(self,
                state_t: chex.Array,
                action_t: chex.Array,
                reward_t: chex.Array,
                done_t: chex.Array
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
        self._states.append(state_t)
        self._actions.append(action_t)
        self._rewards.append(reward_t)
        self._dones.append(done_t)
        self._step += 1

        if self._step % self._steps_between_updates and self._step >= self.coef_buffer * self._steps_between_updates + 1:
            self._rng, rng = jax.random.split(self._rng)

            # Compute the value and the log_prob of actions
            states_batch = np.stack(self._states, axis=0)[:-1]
            next_states_batch = np.stack(self._states, axis=0)[1:]
            actions_batch = np.stack(self._actions, axis=0)[:-1]
            rewards_batch = np.stack(self._rewards, axis=0)[:-1]
            values_batch = self._value_apply(self._learner_state.value_params, states_batch)
            dones_batch = np.stack(self._dones, axis=0)

            logits = self._policy_apply(self._learner_state.policy_params, states_batch)
            self._rng, rng = jax.random.split(self._rng)
            u = jax.random.uniform(rng, shape=logits.shape)
            actions = jnp.argmax(logits - jnp.log(-jnp.log(u)), axis=1)[:-1]
            logprobs_batch = jax.nn.log_softmax(logits)[jnp.arange(actions.shape[0]), actions]

            advantages_gae = np.array(self.compute_gae(rewards=rewards_batch, values=values_batch, dones=dones_batch))
            values_batch, dones_batch = values_batch[:-1], dones_batch[:-1]

            n_updates = self.ARGS['update_epochs']

            for _ in range(n_updates):
                # indexes of the samples chosen in the mini-batch
                mini_batch_idx = np.random.choice(self.coef_buffer * self._steps_between_updates,
                                                  self.ARGS['minibatch_size'], replace=False)

                advantages = advantages_gae[mini_batch_idx]
                states = states_batch[mini_batch_idx]
                next_states = next_states_batch[mini_batch_idx]
                actions = actions_batch[mini_batch_idx]
                rewards = rewards_batch[mini_batch_idx]
                dones = dones_batch[mini_batch_idx]
                logprobs = logprobs_batch[mini_batch_idx]
                values = values_batch[mini_batch_idx]
                returns = advantages + values

                policy_loss, value_loss, self._learner_state = self._update_fn(self._learner_state,
                                                                               states=states,
                                                                               actions=actions,
                                                                               rewards=rewards,
                                                                               logprobs=logprobs,
                                                                               advantages=advantages,
                                                                               returns=returns,
                                                                               next_states=next_states
                                                                               )
                self.logs['policy_grad_loss'] = policy_loss
                self.logs['value_loss'] = value_loss
                # self.logs['dynamics_model_loss'] = dyn_loss
            return policy_loss + value_loss
        return None

    def act(self,
            state: chex.Array,
            eval: bool,
            ) -> chex.Array:
        """Pick the next action according to the learnt poslicy."""
        # Your code here
        bstate = state[None]
        logits = self._policy_apply(self._learner_state.policy_params, bstate)

        if eval:
            action = jnp.argmax(logits[0])
            """
            cum_rew1, cum_rew2 = self.plan_T_steps(bstate, action, T=10)
            self.logs['Estimated_Cum_reward_PPO']= cum_rew1
            self.logs['Estimated_Cum_reward_concu']= cum_rew2
            if cum_rew1 < cum_rew2 + 1.:
              action = 1-action
            """
            return action
        self._rng, rng = jax.random.split(self._rng)
        u = jax.random.uniform(rng, shape=logits.shape)
        action = jnp.argmax(logits - jnp.log(-jnp.log(u)), axis=1)[0]
        if self._step % 500 == 0:
            message = [key + ': ' + str(value) for (key, value) in self.logs.items()]
            print(message)
        return action

    def sample_action(self,
                      state,
                      eval: bool,
                      ) -> chex.Array:
        """Pick the next action according to the learnt poslicy."""
        # Your code here
        state = compute_features_from_observation(state)
        bstate = state[None]
        logits = self._policy_apply(self._learner_state.policy_params, bstate)[0]
        if eval:
            return jnp.argmax(logits)
        self._rng, rng = jax.random.split(self._rng)
        return jax.random.categorical(rng, logits)

    def plan_T_steps(self, state, action, T=10):
        """
        Generate two plans with an horiron of T.

        The first plan corresponds to the trajectory in which the agent takes the action
        'action' at the state 'state'. The second plan corresponds to the trajectory in which
        the agent take the OPPOSITE action (we name this trajectory 'concurrent' trajectory).
        After this first state, actions are sampled using the policy network.
        """
        s_t, conc_s_t = state.copy(), state.copy()
        action_t, concurrent_a_t = action, 1 - action
        cum_reward, conc_cum_reward = 0., 0.

        for t in range(T):
            cat_states = jnp.concatenate([s_t, conc_s_t], axis=0)
            if t > 0:
                cat_actions = jnp.argmax(self._policy_apply(self._learner_state.policy_params, cat_states), axis=0)
                cat_actions = np.array(cat_actions).reshape((2, -1))
            else:
                cat_actions = np.array([action_t, concurrent_a_t]).reshape((2, -1))

            # Get the predicted reward and the predicted difference: next_state-current_state
            diff_states = self._dynamics_apply(self._learner_state.dynamics_params, cat_states, cat_actions)

            # next states
            s_tp1, concurrent_s_tp1 = diff_states[0] + s_t, diff_states[1] + conc_s_t
            # print('s_tp1', s_tp1.shape)
            cum_reward += 1. * (s_tp1[0, 2] < -0.2 or s_tp1[0, 4] < -0.2)
            conc_cum_reward += 1. * (concurrent_s_tp1[0, 2] < -0.2 or concurrent_s_tp1[0, 4] < -0.2)
            s_t, conc_s_t = s_tp1.copy(), concurrent_s_tp1.copy()

        return [cum_reward, conc_cum_reward]

    def get_action_and_value2(self,
                              policy_params: hk.Params,
                              value_params: hk.Params,
                              states: np.ndarray,
                              actions: np.ndarray,
                              ):
        """calculate value, logprob of supplied `action`, and entropy"""
        logits = self._policy_apply(policy_params, states)
        ##logits = actor.apply(params.actor_params, hidden)
        logprob = jax.nn.log_softmax(logits)[jnp.arange(actions.shape[0]), actions]
        # normalize the logits https://gregorygundersen.com/blog/2020/02/09/log-sum-exp/
        logits = logits - jax.scipy.special.logsumexp(logits, axis=-1, keepdims=True)
        logits = logits.clip(min=jnp.finfo(logits.dtype).min)
        p_log_p = logits * jax.nn.softmax(logits)
        entropy = -p_log_p.sum(-1)
        value = self._value_apply(value_params, states).squeeze()

        return logprob, entropy, value

    def ppo_loss(self,
                 policy_params: hk.Params,
                 value_params: hk.Params,
                 dynamics_params: hk.Params,
                 states: np.ndarray,
                 actions: np.ndarray,
                 rewards,
                 logp,
                 mb_advantages,
                 mb_returns,
                 next_states):

        newlogprob, entropy, newvalue = self.get_action_and_value2(policy_params,
                                                                   value_params,
                                                                   states,
                                                                   actions)
        # print('lop.shape', logp.shape, 'value.shape', )
        logratio = newlogprob - logp
        ratio = jnp.exp(logratio)
        approx_kl = ((ratio - 1) - logratio).mean()
        # print('newvalue.shape:', newvalue.shape, 'mb_returns: ', mb_returns.shape)
        # if args.norm_adv:
        # mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-10)

        # Policy loss
        pg_loss1 = -mb_advantages * ratio
        pg_loss2 = -mb_advantages * jnp.clip(ratio, 1 - self.clip_coef, 1 + self.clip_coef)
        pg_loss = jnp.maximum(pg_loss1, pg_loss2).mean()

        # Value loss
        v_loss = 0.5 * ((newvalue - mb_returns) ** 2).mean()

        entropy_loss = entropy.mean()
        # loss = pg_loss - self.ARGS['ent_coef'] * entropy_loss + v_loss * self.ARGS['vf_coef']

        # the 'dynamics network' predicts the reward and the difference s_{t+1}-s_t
        # pred_diff_states = self._dynamics_apply(dynamics_params, states, actions[:,None])
        # loss_diff_states = jnp.mean((next_states-states-pred_diff_states)**2)
        # loss_rewards = jnp.mean((rewards-pred_rewards)**2)
        # loss_dynamics_model = loss_diff_states #+ loss_rewards
        # loss_dynamics_model = loss_dynamics_model * (loss_dynamics_model < 0.5)
        return pg_loss - self.ARGS['ent_coef'] * entropy_loss + v_loss  # , loss_dynamics_model

    def compute_gae(self,
                    rewards: np.array,
                    values: np.array,
                    dones: np.array):
        advantages = np.zeros(self.coef_buffer * self._steps_between_updates, dtype=np.float32)
        lastgaelam = 0
        for t in reversed(range(self.coef_buffer * self._steps_between_updates)):
            delta = rewards[t] + self._gamma * values[t + 1] * (1 - dones[t + 1]) - values[t]
            lastgaelam = delta + self._gamma * self.ARGS['gae_lambda'] * (1 - dones[t + 1]) * lastgaelam
            advantages[t] = lastgaelam

        return advantages

    def _update_fn(self,
                   learner_state: LearnerState,
                   states: chex.Array,
                   actions: chex.Array,
                   rewards: chex.Array,
                   logprobs,
                   advantages,
                   returns,
                   next_states: chex.Array,
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
            return self.ppo_loss(*args, **kwargs)[0]

        def _value_loss(*args, **kwargs):
            return self.ppo_loss(*args, **kwargs)[1]

        def _dynamics_model_loss(*args, **kwargs):
            return self.ppo_loss(*args, **kwargs)[2]

        args = [learner_state.policy_params, learner_state.value_params,
                learner_state.dynamics_params, states, actions, rewards, logprobs,
                advantages, returns, next_states]

        policy_loss, policy_grad = jax.value_and_grad(_policy_loss)(*args)
        value_loss, value_grad = jax.value_and_grad(_value_loss, argnums=1)(*args)
        # dynamics_loss, dynamics_grad = jax.value_and_grad(_dynamics_model_loss, argnums=2)(*args)
        policy_updates, new_policy_opt_state = self._optimizer().update(policy_grad, learner_state.policy_opt_state)
        value_updates, new_value_opt_state = self._optimizer().update(value_grad, learner_state.value_opt_state)
        # dynamics_updates, new_dynamics_opt_state = self._optimizer().update(dynamics_grad, learner_state.dynamics_opt_state)

        new_policy_params = optax.apply_updates(learner_state.policy_params, policy_updates)
        new_value_params = optax.apply_updates(learner_state.value_params, value_updates)
        # new_dynamics_params = optax.apply_updates(learner_state.dynamics_params, dynamics_updates)

        return policy_loss, value_loss, LearnerState(policy_params=new_policy_params,
                                                     policy_opt_state=new_policy_opt_state,
                                                     value_params=new_value_params, value_opt_state=new_value_opt_state,
                                                     dynamics_params=None, dynamics_opt_state=None)