import copy
from typing import Tuple, List

import numpy as np
from deep_rl.environments.flappy_bird import FlappyBird
from tqdm import trange

from helper import compute_features_from_observation


def run_ppo_episode(ppo_agent, last_state: np.array, env, eval: bool, max_steps: int) -> float:
    # Reset any counts and start the environment.
    # state = copy.deepcopy(last_state)
    # if eval:
    state = copy.deepcopy(env.reset())
    n_steps = 0
    total_reward = 0
    # Run an episode.
    while True:
        # Generate an action from the agent's policy and step the environment.
        state_features = compute_features_from_observation(state)
        action = ppo_agent.act(state_features, eval)
        # print('action', action)
        next_state, reward, done = env.step(action)
        if not eval:
            ppo_agent.observe(state_features, action, reward, done)
        state = copy.deepcopy(next_state)
        n_steps += 1
        total_reward += reward
        if not eval:
            last_state = copy.deepcopy(state)
        if done or n_steps > max_steps:
            last_state = copy.deepcopy(env.reset())
            break

    return total_reward, n_steps, last_state


def train_agent_ppo(
        agent,
        env: FlappyBird,
        num_episodes: int = 2000,
        num_eval_episodes: int = 10,
        eval_every_N: int = 100,
        max_steps: int = 1000,
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
    eval_rewards = []
    eval_n_steps = []
    train_rewards = []
    train_n_steps = []

    print(
        "{:^18}|{:^40}|{:^40}".format(
            "Episode number:",
            f"Average reward on {num_eval_episodes} episodes",
            f"Average Steps on {num_eval_episodes} episodes",
        )
    )
    print(
        "------------------------------------------------------------------------------------------------------------------------"
    )
    display_pbar = verbose in [1, 3]
    if display_pbar:
        pbar = trange(1, num_episodes + 1, desc="Training", leave=True)
    else:
        pbar = range(1, num_episodes + 1)

    last_state = copy.deepcopy(env.reset())
    reward, n_step = np.mean(
        np.concatenate(
            [
                [run_ppo_episode(agent, last_state=last_state, env=env, eval=True, max_steps=max_steps)[0:2]]
                for _ in range(num_eval_episodes)
            ],
            axis=0,
        ),
        axis=0,
    )
    for episode in pbar:
        reward, n_step, last_state = run_ppo_episode(agent, last_state, env, eval=False, max_steps=max_steps)
        train_rewards.append(reward)
        train_n_steps.append(n_step)

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

        if episode % eval_every_N == 0:
            reward, n_step = np.mean(
                np.concatenate(
                    [
                        [run_ppo_episode(agent, last_state=last_state, env=env, eval=True, max_steps=max_steps)[0:2]]
                        for _ in range(num_eval_episodes)
                    ],
                    axis=0,
                ),
                axis=0,
            )
            print("{:^18}|{:^40}|{:^40}".format(episode, reward, n_step))
            eval_rewards.append(reward)
            eval_n_steps.append(n_step)
            episodes.append(episode)

    return episodes, eval_rewards, eval_n_steps