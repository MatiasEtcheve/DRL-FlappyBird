from deep_rl.project_values import PROJECT_FLAPPY_BIRD_ENV
from optuna import Trial

from helper.agents.agent_ppo import PPOAgent
from helper.hparm_tuning.hyperparm_tuning import launch_study
from helper.training.tain_ppo import train_agent_ppo
from helper.training.train_dqn import evaluate_agent


def objective_ppo(trial: Trial):

    # Agent variables
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
    steps_between_updates = trial.suggest_int("steps_between_updates", 4, 36, step=4)
    minibatch_size = trial.suggest_int("minibatch_size", 4, 32, step=4)

    ppo_supplementary_args = {
        'gae_lambda': 0.95,
        'update_epochs': 4,
        'clip_coef': 0.2,
        'ent_coef': 0.01,
        'vf_coef': 0.5,
        'update_epochs': 2,
        'minibatch_size': minibatch_size,
        'batch_size': 16,
        'max_grad_norm': 0.5
    }
    env = PROJECT_FLAPPY_BIRD_ENV
    agent = PPOAgent(
        env=env,
        gamma=0.95,
        learning_rate=learning_rate,
        steps_between_updates=steps_between_updates,
        seed=0,
        PPO_supplementary_ARGS=ppo_supplementary_args
    )
    _ = train_agent_ppo(
        agent,
        env=env,
        num_episodes=TRIAL_NUM_EPISODES,
        num_eval_episodes=10,
        eval_every_N=100,
        max_steps=1000,
        verbose=1
    )
    print("Final evaluation... ", end='')
    reward = evaluate_agent(agent, env, n_episodes=100)
    print(reward)
    return reward


if __name__ == '__main__':
    N_TRIALS = 30
    TRIAL_NUM_EPISODES = 1000
    TIMEOUT_MINS = 15
    study_name = "PPO"

    launch_study(
        objective_function=objective_ppo,
        n_trials=N_TRIALS,
        timeout_minutes=TIMEOUT_MINS,
        study_name=study_name
    )