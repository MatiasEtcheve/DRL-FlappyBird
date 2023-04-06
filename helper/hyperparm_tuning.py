import logging
import sys

import optuna
from deep_rl.project_values import PROJECT_FLAPPY_BIRD_ENV
from optuna import Trial
from optuna.trial import TrialState
from optuna.visualization import plot_parallel_coordinate
from tqdm import trange

from helper.agent_dqn_per import DeepAgent
from helper.training import train_agent, run_dqn_episode

TRIAL_NUM_EPISODES = 300

def objective(trial: Trial):

    # Agent variables
    gamma = trial.suggest_float('gama', 0.9, 0.99, step=0.01)
    eps = trial.suggest_float("eps", 0.1, 0.8, step=0.1)
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
    target_ema = trial.suggest_float("target_ema", 0.7, 0.95, step=0.05)


    env = PROJECT_FLAPPY_BIRD_ENV
    agent = DeepAgent(
        env=env,
        gamma=gamma,
        eps=eps,
        learning_rate=learning_rate,
        buffer_capacity=5000,
        min_buffer_capacity=64,
        batch_size=64,
        target_ema=target_ema
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
    reward = 0
    print("Final evaluation")
    for _ in trange(1000):
        reward += run_dqn_episode(agent, env, evaluation=True, max_steps=1000, renderer=None)[0]
    reward /= 1000
    return reward

def launch_study(n_trials: int, timeout_minutes: int, name: str):
    optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))

    # Make unique name for the study and database filename
    storage_name = f"sqlite:///optuna_{name}.db"

    study = optuna.create_study(
        study_name=name,
        storage=storage_name,
        direction="maximize",
        load_if_exists=True
    )
    study.optimize(objective, n_trials=n_trials, timeout=60 * timeout_minutes)

    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
    plot_parallel_coordinate(study)


if __name__ == '__main__':
    N_TRIALS = 30
    TIMEOUT_MINS = 15
    study_name = "DQN_PER"

    launch_study(N_TRIALS, TIMEOUT_MINS, study_name)



