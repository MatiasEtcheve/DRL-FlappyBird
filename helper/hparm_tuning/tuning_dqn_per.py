from deep_rl.project_values import PROJECT_FLAPPY_BIRD_ENV
from optuna import Trial

from helper.agents.agent_dqn_per import DqnPerAgent
from helper.hparm_tuning.hyperparm_tuning import launch_study
from helper.training.train_dqn import train_agent, evaluate_agent


def objective_dqn_per(trial: Trial):

    # Agent variables
    gamma = trial.suggest_float('gama', 0.9, 0.99, step=0.01)
    eps = trial.suggest_float("eps", 0.1, 0.8, step=0.1)
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
    target_ema = trial.suggest_float("target_ema", 0.7, 0.95, step=0.05)
    network_hdim = trial.suggest_int("network_hdim", 8, 64, step=8)


    env = PROJECT_FLAPPY_BIRD_ENV
    agent = DqnPerAgent(
        env=env,
        gamma=gamma,
        eps=eps,
        learning_rate=learning_rate,
        buffer_capacity=5000,
        min_buffer_capacity=64,
        batch_size=64,
        target_ema=target_ema,
        network_hdim=network_hdim
    )
    _ = train_agent(
        agent,
        env=env,
        num_episodes=TRIAL_NUM_EPISODES,
        num_eval_episodes=10,
        eval_every_N=100,
        max_steps=1000,
        max_eval_steps=1000,
        verbose=1,
        trial=trial
    )
    print("Final evaluation... ", end='')
    reward = evaluate_agent(agent, env, n_episodes=100)
    print(reward)
    return reward


if __name__ == '__main__':
    N_TRIALS = 30
    TRIAL_NUM_EPISODES = 1000
    TIMEOUT_MINS = 15
    study_name = "DQN_PER"

    launch_study(
        objective_function=objective_dqn_per,
        n_trials=N_TRIALS,
        timeout_minutes=TIMEOUT_MINS,
        study_name=study_name
    )


