import pickle
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.colors import to_rgba
from matplotlib.figure import Figure
from mpl_toolkits.axes_grid1 import make_axes_locatable

from helper.defined_in_notebook import BirdObs, BarObs, FlappyObs


def split_bars_by_activity(
    bird: BirdObs, bars: List[BarObs]
) -> Tuple[List[BirdObs], List[BirdObs]]:
    """
    Returns [active bars], [inactive bars]
    The active bars are sorted by X_LEFT
    """
    if len(bars) == 0:
        return [], []
    x_bird, y_bird, v_bird = bird
    bars = np.array(bars, dtype="object")
    idx_active_bars = np.array(
        [True if x_right_bar > x_bird else False for (_, x_right_bar, _, _) in bars]
    )

    return sorted(bars[idx_active_bars], key=lambda bar: bar[0]), bars[~idx_active_bars]


def split_bar_by_position(bars: List[BarObs]) -> Tuple[List[BirdObs], List[BirdObs]]:
    """
    Returns list of [bottom bars], [top bars]
    """
    if len(bars) == 0:
        return [], []
    bars = np.array(bars, dtype="object")
    idx_bottom_bars = np.array([position for (_, _, _, position) in bars])
    return bars[idx_bottom_bars], bars[~idx_bottom_bars]


def compute_bird_bar_distance(bird: BirdObs, bars: List[BirdObs], foreseen_bars=2):
    """
    Computes the distance from the bird to the next `foreseen_bars` top bars and next `foreseen_bars` bottom bars
    """
    x_bird, y_bird, v_bird = bird
    active_bars, _ = split_bars_by_activity(bird, bars)
    bottom_active_bars, top_active_bars = split_bar_by_position(active_bars)

    closest_bars_top = (
        list(top_active_bars) + [(1, 1 + 0.05, 0, False)] * foreseen_bars
    )[:foreseen_bars]
    closest_bars_bottom = (
        list(bottom_active_bars) + [(1, 1 + 0.05, 0, True)] * foreseen_bars
    )[:foreseen_bars]

    # list of dx, dy distances from the top closest bars
    distances_top = [(bar[0] - x_bird, 1 - y_bird - bar[2]) for bar in closest_bars_top]
    # list of dx, dy distances from the bottom closest bars
    distances_bottom = [
        (bar[0] - x_bird, y_bird - bar[2]) for bar in closest_bars_bottom
    ]

    # we finally flatten the distances
    return [item for sublist in distances_top for item in sublist] + [
        item for sublist in distances_bottom for item in sublist
    ]


def compute_features_from_observation(observation: FlappyObs, foreseen_bars: int = 2):
    """
    Computes all the features from a specific observation as follow
        [Y_BIRD, V_BIRD, distances to the `foreseen_bars` top bars, distances to the `foreseen_bars` bottom bars]
    """
    bird, bars = observation
    _, y_bird, v_bird = bird
    distances = compute_bird_bar_distance(bird, bars, foreseen_bars=foreseen_bars)
    features = np.array([y_bird, v_bird] + list(distances))
    return features


def seek_distance_to_next_bar(bird: BirdObs, bars: List[BarObs]):
    """
    Computes the distance to the closest bar in front of the bird. If nothing is in front of the bird, it should return 0.
    """
    x_bird, y_bird, _ = bird
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


def get_value_and_policy(
    agent, observation: FlappyObs, n_x: int = 100, n_y: int = 50
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes the value and policy heatmap for a specific observation and a specific agent.

    WARNING: the agent must have the `compute_q_value_from_observation(self, observation: FlappyObs)` method
        which returns the q vector of length the number of action for a specific observation

    n_x and n_y are the size of heatmap. As the heatmap is discrete, we need to compute the q value for every point on the grid.

    """
    (x_bird, y_bird, v_bird), bars = observation
    value_renderer = np.zeros((n_y, n_x))
    action_renderer = np.zeros((n_y, n_x))
    for idx_x, x_bird in enumerate(np.linspace(0, 1, n_x)):
        for idx_y, y_bird in enumerate(np.linspace(0, 1, n_y)):
            observation = ((x_bird, y_bird, v_bird), bars)
            # q value: update if using other kind of agent
            q = agent.compute_q_value_from_observation(observation)
            value_renderer[idx_y, idx_x] = np.mean(q)
            # action
            proba_action = np.exp(q) / sum(np.exp(q))
            action_renderer[idx_y, idx_x] = proba_action[1]
    return value_renderer, action_renderer


def plot_value_and_policy(agent, observation: FlappyObs) -> Tuple[Figure, Axes]:
    """Plot the value and action heatmaps for a particular observation of a environment

    WARNING: the agent must have the `compute_q_value_from_observation(self, observation: FlappyObs)` method
        which returns the q vector of length the number of action for a specific observation
    """
    (x_bird, y_bird, v_bird), bars = observation
    n_x = 100
    n_y = n_x // 2
    value, action = get_value_and_policy(agent, observation, n_x, n_y)
    fig, axs = plt.subplots(1, 2, figsize=(24, 8))
    divider = make_axes_locatable(axs[0])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    im = axs[0].imshow(value, cmap="viridis", origin="lower")
    fig.colorbar(im, cax=cax, orientation="vertical")
    axs[0].axis("off")
    axs[0].set_title("Value function")

    divider = make_axes_locatable(axs[1])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    im = axs[1].imshow(action, origin="lower")
    fig.colorbar(im, cax=cax, orientation="vertical")
    axs[1].axis("off")
    axs[1].set_title("Policy")

    for i in range(2):
        for bar in bars:
            x_left_bar, x_right_bar, height, position = bar
            position = 1 - int(position)
            axs[i].add_patch(
                plt.Rectangle(
                    (x_left_bar * n_x, position * (n_y + 1) - 1),
                    (x_right_bar - x_left_bar) * n_x,
                    (1 - 2 * position) * height * n_y,
                    facecolor=to_rgba("dimgray", 1),
                    edgecolor="black",
                )
            )
    return fig, axs


def plot_observation(observation: FlappyObs) -> Tuple[Figure, Axes]:
    _, bars = observation
    fig, axs = plt.subplots(1, 1, figsize=(8, 4))
    for bar in bars:
        x_left_bar, x_right_bar, height, position = bar
        position = int(position)
        bottom = 1 - position
        axs.add_patch(
            plt.Rectangle(
                (x_left_bar, bottom),
                (x_right_bar - x_left_bar),
                (1 - 2 * bottom) * height,
                facecolor=to_rgba("dimgray", 1),
                edgecolor="black",
            )
        )
    return fig, axs


def save_best_model(agent, new_reward: float, old_reward: float, filename: str) -> None:
    if new_reward > old_reward:
        print(f"Saving best model at {filename}")
        with open(filename, "wb") as file:
            pickle.dump(agent._learner_state, file)
