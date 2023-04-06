from deep_rl.project_values import PROJECT_FLAPPY_BIRD_ENV
from optuna import Trial

from helper.agents.agent_a2c import A2CAgent
from helper.hparm_tuning.hyperparm_tuning import launch_study
from helper.training.train_a2c import train_agent_a2c
from helper.training.train_dqn import evaluate_agent


def objective_a2c(trial: Trial):

    # Agent variables
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
    steps_between_updates = trial.suggest_int("steps_between_updates", 4, 36, step=4)
    foreseen_bars = trial.suggest_int("foreseen_bars", 1, 4)
    hdim_policy_1 = trial.suggest_int("hdim_policy_1", 32, 128, step=32)
    hdim_policy_2 = trial.suggest_int("hdim_policy_2", 8, 64, step=8)
    hdim_value = trial.suggest_int("hdim_value", 32, 128, step=32)


    env = PROJECT_FLAPPY_BIRD_ENV
    agent = A2CAgent(
        env=env,
        gamma=0.95,
        learning_rate=learning_rate,
        steps_between_updates=steps_between_updates,
        seed=0,
        foreseen_bars=foreseen_bars,
        hdim_policy_1=hdim_policy_1,
        hdim_policy_2=hdim_policy_2,
        hdim_value=hdim_value
    )
    _ = train_agent_a2c(
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
    study_name = "A2C"

    launch_study(
        objective_function=objective_a2c,
        n_trials=N_TRIALS,
        timeout_minutes=TIMEOUT_MINS,
        study_name=study_name
    )
