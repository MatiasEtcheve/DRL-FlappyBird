import copy
import time
from datetime import time
from typing import Tuple, List

import numpy as np
from matplotlib import pyplot as plt
from tqdm import trange

from helper import compute_features_from_observation, save_best_model, plot_value_and_policy


def run_dqn_episode(
    agent,
    env,
    evaluation: bool,
    max_steps: int = 1000,
    renderer: bool = None,
    time_between_frame: float = 0.1,
) -> Tuple[float, int]:
    """Runs an agent on the environment `env`.

    Args:
        agent (DeepAgent): agent to run
        env (FlappyBird): running environment
        evaluation (bool): eval mode. If `True`, the policy is greedy.
        max_steps (int, optional): Number of steps of the agent. Defaults to 1000.
        render (bool, optional): Real-time rendering of the policy run by the agent. Defaults to False.
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
        action = agent.act(compute_features_from_observation(state), evaluation)
        next_state, reward, done = env.step(action)
        if not evaluation:
            agent.observe(action, reward, done, copy.deepcopy(next_state))
        n_steps += 1
        total_reward += reward

        state = copy.deepcopy(next_state)
        if done or n_steps > max_steps:
            break

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
        eval_every_N: int = 100,
        max_steps: int = 1000,
        max_eval_steps: int = 1000,
        verbose: int = 0,
) -> Tuple[List[int], List[float], List[int]]:
    """Trains an agent for a specified number of iterations

    Args:
        agent (DeepAgent): agent to run
        env (Maze): running environment
        num_episodes (int, optional): Number of episodes to run. Defaults to 2000.
        num_eval_episodes (int, optional): Number of validation episodes. Defaults to 10.
        eval_every_N (int, optional): Number of training episodes between a validation step. Defaults to 100.
        max_steps (int, optional): maximum number of steps per episode. Defaults to 1000.
        verbose (int, optional): Verbose level. Defaults to 0.
            * 0: display only the validation metrics
            * 1: display the training metrics
            * 2: plot the value function and the visited states
            * 3: 1 & 2

    Returns:
        Tuple[List[int], List[float], List[int]]: val episodes, val rewards and val number of steps of the agent
    """
    episodes = []
    eval_reward = 0
    eval_rewards = []
    eval_n_steps = []
    train_rewards = []
    train_n_steps = []
    starting_time = time.time()
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
    display_pbar = verbose in [1, 3]
    display_plots = verbose in [2, 3]
    if display_pbar:
        pbar = trange(1, num_episodes + 1, desc="Training", leave=True)
    else:
        pbar = range(1, num_episodes + 1)

    for episode in pbar:
        # training step
        reward, n_step = run_dqn_episode(
            agent, env, evaluation=False, max_steps=max_steps
        )
        train_rewards.append(reward)
        train_n_steps.append(n_step)

        # training metrics, nothing interesting
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

        # evaluation
        if episode % eval_every_N == 0:
            reward, n_step = np.mean(
                np.concatenate(
                    [
                        [
                            run_dqn_episode(
                                agent, env, evaluation=True, max_steps=max_eval_stepsx
                            )
                        ]
                        for _ in range(num_eval_episodes)
                    ],
                    axis=0,
                ),
                axis=0,
            )
            print(
                "{:^18}|{:^18.2f}|{:^40}|{:^40}".format(
                    episode, (time.time() - starting_time) / 60, reward, n_step
                )
            )
            eval_rewards.append(reward)
            eval_n_steps.append(n_step)
            episodes.append(episode)

            # save best model
            save_best_model(agent, eval_rewards[-1], eval_reward, filename="best_model.pkl")
            eval_reward = np.max(eval_rewards)

            # display value and action maps
            if display_plots:
                fig, axs = plot_value_and_policy(agent)
                plt.show()

    return episodes, eval_rewards, eval_n_steps
