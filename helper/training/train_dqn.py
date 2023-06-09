import copy
import time
from typing import Tuple, List, Optional

import numpy as np
import optuna.exceptions
from matplotlib import pyplot as plt
from optuna import Trial
from tqdm import trange

from helper import save_best_model, plot_value_and_policy


def run_dqn_episode(
    agent,
    env,
    evaluation: bool,
    max_steps: int = 1000,
    renderer = None,
    time_between_frame: float = 0.1,
) -> Tuple[float, int]:
    """Runs an agent on the environment `env`.

    Args:
        agent (DeepAgent): agent to run
        env (FlappyBird): running environment
        evaluation (bool): eval mode. If `True`, the policy is greedy.
        max_steps (int, optional): Number of steps of the agent. Defaults to 1000.
        renderer (optional): Real-time rendering of the policy run by the agent. Defaults to None.
        time_between_frame (float, optional): time bewteen 2 rendered frames

    Returns:
        Tuple[float, int]:  final reward and  number of steps of the agent
    """
    state = copy.deepcopy(env.reset())
    agent.first_observe(state)
    n_steps = 0
    total_reward = 0

    # Run an episode.
    while True:
        # Generate an action from the agent's policy and step the environment.
        action = agent.sample_action(state, evaluation)
        next_state, reward, done = env.step(action)
        if not evaluation:
            agent.observe(action, reward, done, copy.deepcopy(next_state))
        n_steps += 1
        total_reward += reward

        state = copy.deepcopy(next_state)
        if done or n_steps > max_steps:
            break

        # Render episode if a renderer was passed as an argument
        if renderer is not None:
            renderer.clear()
            renderer.draw_list(env.render())
            renderer.draw_title(f"TOTAL REWARD : {total_reward}")
            renderer.render()
            time.sleep(time_between_frame)

    return total_reward, n_steps


def train_agent(
        agent,
        env,
        num_episodes: int = 2000,
        num_eval_episodes: int = 10,
        eval_every_N: Optional[int] = 100,
        max_steps: int = 1000,
        max_eval_steps: int = 1000,
        verbose: int = 0,
        trial: Optional[Trial] = None
) -> Tuple[List[int], Tuple[List[float]], Tuple[List[float]]]:
    """Trains an agent for a specified number of iterations

    Args:
        agent (DeepAgent): agent to run
        env (FlappyBird): running environment
        num_episodes (int, optional): Number of episodes to run. Defaults to 2000.
        num_eval_episodes (int, optional): Number of validation episodes. Defaults to 10.
        eval_every_N (int, optional): Number of training episodes between evaluations. Defaults to 100.
        max_steps (int, optional): maximum number of steps per episode. Defaults to 1000.
        max_eval_steps (int, optional): maximum number of steps per episode during evaluation. Defaults to 1000.
        verbose (int, optional): Verbose level. Defaults to 0.
            * 0: display only the validation metrics
            * 1: display the training metrics
            * 2: plot the value function and the visited states
            * 3: 1 & 2
        trial (Trial, optional): Optuna trial object used in hyperparameter search

    Returns:
        Tuple[List[int], Tuple[List[float]], Tuple[List[float]]]: val episodes, val rewards and val number of steps of the agent
    """
    episodes = []
    eval_reward = 0
    eval_rewards_means = []
    eval_n_steps_means = []
    eval_rewards_stds = []
    eval_n_steps_stds = []
    train_rewards = []
    train_n_steps = []
    starting_time = time.time()
    if eval_every_N is None:
        verbose = 0

    # Diplay header line
    print(
        "{:^18}|{:^18}|{:^40}|{:^40}".format(
            "Episode number:",
            "Elapsed time (min)",
            f"Average reward on {num_eval_episodes} episodes",
            f"Average Steps on {num_eval_episodes} episodes",
        )
    )
    print(
        "------------------------------------------------------------------------------------------------------------------------"
    )
    # Display settings
    display_pbar = verbose in [1, 3]
    display_plots = verbose in [2, 3]
    if display_pbar:
        pbar = trange(1, num_episodes + 1, desc="Training", leave=True)
    else:
        pbar = range(1, num_episodes + 1)

    # Run training episodes
    for episode in pbar:
        # Training step
        reward, n_step = run_dqn_episode(
            agent, env, evaluation=False, max_steps=max_steps
        )
        train_rewards.append(reward)
        train_n_steps.append(n_step)

        # Display training metrics, nothing interesting
        if display_pbar:
            pbar.set_postfix(
                {
                    f"Average last {eval_every_N} rewards": np.mean(
                        train_rewards[-100:]
                    ),
                    f"Average last {eval_every_N} n_steps": np.mean(
                        train_n_steps[-100:]
                    ),
                }
            )

        # Evaluation
        if eval_every_N is not None and episode % eval_every_N == 0:
            rewards, n_steps = np.concatenate(
                [
                    [run_dqn_episode(agent, env, evaluation=True, max_steps=1000)]
                    for _ in range(num_eval_episodes)
                ],
                axis=0,
            ).T
            mean_rewards = np.mean(rewards)
            mean_n_steps = np.mean(n_steps)
            std_rewards = np.std(rewards)
            std_n_steps = np.std(n_steps)
            print(
                "{:^18}|{:^20.2f}|{:^30} ({:^6.0f}) |{:^30} ({:^6.0f}) ".format(
                    episode,
                    (time.time() - starting_time) / 60,
                    mean_rewards,
                    std_rewards,
                    mean_n_steps,
                    std_n_steps,
                )
            )
            eval_rewards_means.append(mean_rewards)
            eval_n_steps_means.append(mean_n_steps)
            eval_rewards_stds.append(std_rewards)
            eval_n_steps_stds.append(std_n_steps)
            episodes.append(episode)

            # display value and action maps
            if display_plots:
                fig, axs = plot_value_and_policy(agent)
                plt.show()

            # Report result to Optuna (in case of hyperparameter search)
            if trial is not None:
                trial.report(reward, episode)
                # Stop the trial if reward is not progressing enough
                if trial.should_prune():
                    optuna.exceptions.TrialPruned()
            else:
                # save best model
                save_best_model(agent, eval_rewards_means[-1], eval_reward, filename=f"{eval_reward:06.1f}_model.pkl")
                save_best_model(agent, eval_rewards_means[-1], eval_reward, filename=f"best_model.pkl")
                eval_reward = np.max(eval_rewards_means)

    return episodes, (eval_rewards_means, eval_rewards_stds), (eval_n_steps_means, eval_n_steps_stds)


def evaluate_agent(agent, env, n_episodes: int = 100):
    reward = 0
    for _ in trange(n_episodes):
        reward += run_dqn_episode(agent, env, evaluation=True, max_steps=1000, renderer=None)[0]
    reward /= n_episodes
    return reward