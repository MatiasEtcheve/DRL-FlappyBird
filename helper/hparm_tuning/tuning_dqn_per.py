from deep_rl.project_values import PROJECT_FLAPPY_BIRD_ENV
from optuna import Trial

from helper.agents.agent_dqn_per import DqnPerAgent
from helper.hparm_tuning.hyperparm_tuning import launch_study
from helper.training.train_dqn import train_agent, evaluate_agent


def objective_dqn_per(trial: Trial):

    # Agent variables
    gamma = 0.95 # trial.suggest_float('gama', 0.92, 0.97)
    eps = trial.suggest_float("eps", 0., 0.25)
    learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
    target_ema = trial.suggest_float("target_ema", 0.55, 0.8)
    network_hdim = trial.suggest_int("network_hdim", 36, 60, step=8)
    alpha_priority = trial.suggest_float("alpha_priority", 0.4, 0.8, step=0.1)
    beta_priority = trial.suggest_float("beta_priority", 0.4, 0.6, step=0.1)

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
        network_hdim=network_hdim,
        alpha_priority=alpha_priority,
        beta_priority=beta_priority
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
    N_TRIALS = 15
    TRIAL_NUM_EPISODES = 1000
    study_name = "DQN_PER_extended"

    launch_study(
        objective_function=objective_dqn_per,
        n_trials=N_TRIALS,
        timeout_minutes=None,
        study_name=study_name
    )


