import matplotlib.animation as animation
import matplotlib.pyplot as plt
from tqdm import trange

from IPython.display import Video

def plot_observation(observation: FlappyObs, ax=None):
    width_bird = 0.01
    (x_bird, y_bird, _), bars = observation
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    ax.add_patch(
        plt.Rectangle(
            (x_bird, y_bird),
            width_bird,
            2 * width_bird,
            facecolor="yellow",  # to_rgba("dimgray", 1),
            edgecolor="red",
        )
    )
    for bar in bars:
        x_left_bar, x_right_bar, height, position = bar
        position = int(position)
        bottom = 1 - position
        ax.add_patch(
            plt.Rectangle(
                (x_left_bar, bottom),
                (x_right_bar - x_left_bar),
                (1 - 2 * bottom) * height,
                facecolor="green",  # to_rgba("dimgray", 1),
                edgecolor="green",
            )
        )



def save_gif_episode(
    agent,
    env: FlappyBird,
    evaluation: bool = True,
    max_steps: int = 1000,
    filename="test.gif",
    fps: int = 10,
):
    # FIRST
    # RUN AGENT UNTIL DEATH
    # SAVE CUMUL REWARD + OBSERVATIONS
    state = copy.deepcopy(env.reset())
    agent.first_observe(state)
    n_steps = 0
    total_reward = 0

    observations = [state]
    rewards = [0]
    # Run an episode.
    while True:
        # Generate an action from the agent's policy and step the environment.
        action = agent.sample_action(compute_features_from_observation(state), evaluation)
        next_state, reward, done = env.step(action)
        if not evaluation:
            agent.observe(action, reward, done, copy.deepcopy(next_state))
        n_steps += 1
        total_reward += reward

        state = copy.deepcopy(next_state)
        observations.append(state)
        rewards.append(rewards[-1] + reward)
        if done or n_steps > max_steps:
            break

    # SECOND
    # HERE COMES MPL NIGHTMARE
    # PLOT OBSERVATIONS AND REWARD TEXT
    fig, axs = plt.subplots(1, 1, figsize=(16, 8), squeeze=False)
    pbar = trange(len(observations), desc="Creating video from observations")

    def plot_reward(reward, ax):
        width = 0.1
        height = 0.1
        fontsize = 30
        ax.add_patch(
            plt.Rectangle(
                (0, 1 - height),
                width,
                height,
                facecolor="white",  # to_rgba("dimgray", 1),
                edgecolor="black",
            )
        )
        ax.text(
            width / 2,
            1 - height / 2,
            reward,
            horizontalalignment="center",
            verticalalignment="center",
            transform=ax.transAxes,
            fontsize=fontsize,
        )

    def animate(num):
        pbar.update(1)
        axs[0, 0].clear()
        plot_observation(observation=observations[num], ax=axs[0, 0])
        plot_reward(rewards[num], axs[0, 0])
        axs[0, 0].axis("off")
        plt.tight_layout()

    ani = animation.FuncAnimation(
        fig,
        animate,
        interval=1 / fps * 1000,
        blit=False,
        frames=len(observations),
    )
    Writer = animation.writers["ffmpeg"]
    writer = Writer(fps=fps, metadata=dict(artist="Me"))
    ani.save(filename, writer=writer)
    pbar.close()
    plt.close()
    return observations[:-1], rewards[:-1]


observations, _ = save_gif_episode(
    deep_agent, env, evaluation=True, max_steps=1000, filename="test.mp4", fps=10
)
Video("test.mp4", embed=True)