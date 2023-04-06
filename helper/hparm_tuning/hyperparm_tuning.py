import logging
import sys
from typing import Callable, Optional

import optuna
from optuna.trial import TrialState


def launch_study(objective_function: Callable, n_trials: int, timeout_minutes: Optional[int], study_name: str):
    optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))

    # Make unique name for the study and database filename
    storage_name = f"sqlite:///optuna_{study_name}.db"

    study = optuna.create_study(
        study_name=study_name,
        storage=storage_name,
        direction="maximize",
        load_if_exists=True
    )
    timeout_secs = 60 * timeout_minutes if timeout_minutes is not None else timeout_minutes
    study.optimize(objective_function, n_trials=n_trials, timeout=timeout_secs)

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




