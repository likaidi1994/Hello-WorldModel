from __future__ import annotations

from pathlib import Path
from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch

from .models import WorldModel
from .memory import Episode
from .training import Transition


def plot_dataset_phase_space(data: Sequence[Transition], output_path: Path) -> None:
    """Show what states the random data collection covered."""

    states = np.stack([transition[0] for transition in data])
    actions = np.stack([transition[1] for transition in data]).reshape(-1)

    _prepare_output(output_path)
    fig, ax = plt.subplots(figsize=(7, 5))
    scatter = ax.scatter(
        states[:, 0],
        states[:, 1],
        c=actions,
        s=8,
        alpha=0.55,
        cmap="coolwarm",
    )
    ax.axvline(0.0, color="black", linewidth=1, linestyle="--", label="goal")
    ax.set_title("Random Data Coverage")
    ax.set_xlabel("position")
    ax.set_ylabel("velocity")
    ax.legend(loc="best")
    fig.colorbar(scatter, ax=ax, label="action")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_training_loss(losses: Sequence[float], output_path: Path) -> None:
    """Show whether the learned dynamics/reward model is fitting the data."""

    _prepare_output(output_path)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(np.arange(1, len(losses) + 1), losses, linewidth=2)
    ax.set_title("World Model Training Loss")
    ax.set_xlabel("epoch")
    ax.set_ylabel("MSE loss")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_reward_model_losses(
    losses: dict[str, Sequence[float]],
    output_path: Path,
) -> None:
    """Show total, observation, and reward losses for a reward world model."""

    _prepare_output(output_path)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    epochs = np.arange(1, len(losses["total"]) + 1)
    ax.plot(epochs, losses["total"], label="total", linewidth=2)
    ax.plot(epochs, losses["obs"], label="observation", linewidth=1.8)
    ax.plot(epochs, losses["reward"], label="reward", linewidth=1.8)
    ax.set_title("Reward World Model Training Losses")
    ax.set_xlabel("epoch")
    ax.set_ylabel("MSE loss")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_latent_model_losses(
    losses: dict[str, Sequence[float]],
    output_path: Path,
) -> None:
    """Show latent dynamics model training losses."""

    _prepare_output(output_path)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    epochs = np.arange(1, len(losses["total"]) + 1)
    ax.plot(epochs, losses["total"], label="total", linewidth=2)
    ax.plot(epochs, losses["obs"], label="decoded observation", linewidth=1.8)
    ax.plot(epochs, losses["reward"], label="reward", linewidth=1.8)
    ax.set_title("Latent Dynamics Model Training Losses")
    ax.set_xlabel("epoch")
    ax.set_ylabel("MSE loss")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_stochastic_model_losses(
    losses: dict[str, Sequence[float]],
    output_path: Path,
) -> None:
    """Show stochastic latent model observation, reward, and KL losses."""

    _prepare_output(output_path)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    epochs = np.arange(1, len(losses["total"]) + 1)
    ax.plot(epochs, losses["total"], label="total", linewidth=2)
    ax.plot(epochs, losses["obs"], label="observation", linewidth=1.8)
    ax.plot(epochs, losses["reward"], label="reward", linewidth=1.8)
    ax.plot(epochs, losses["kl"], label="KL", linewidth=1.8)
    ax.set_title("Stochastic Latent Model Training Losses")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


@torch.no_grad()
def plot_one_step_predictions(
    model: WorldModel,
    data: Sequence[Transition],
    output_path: Path,
    max_points: int = 800,
    device: str = "cpu",
) -> None:
    """Compare real next states with one-step predictions."""

    model.eval()
    sample = list(data[:max_points])

    states = torch.tensor(
        np.stack([transition[0] for transition in sample]),
        dtype=torch.float32,
        device=device,
    )
    actions = torch.tensor(
        np.stack([transition[1] for transition in sample]),
        dtype=torch.float32,
        device=device,
    )
    true_next = np.stack([transition[2] for transition in sample])
    pred_next, _ = model(states, actions)
    pred_next_np = pred_next.cpu().numpy()

    _prepare_output(output_path)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    _plot_prediction_panel(
        axes[0],
        true_next[:, 0],
        pred_next_np[:, 0],
        "next position",
    )
    _plot_prediction_panel(
        axes[1],
        true_next[:, 1],
        pred_next_np[:, 1],
        "next velocity",
    )

    fig.suptitle("One-Step Prediction: Real Environment vs Learned Model")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_planned_trajectory(
    states: np.ndarray,
    actions: np.ndarray,
    rewards: Sequence[float],
    output_path: Path,
) -> None:
    """Show how planning moves the point mass toward the goal."""

    steps = np.arange(len(states))
    action_steps = np.arange(len(actions))
    cumulative_reward = np.cumsum(rewards)

    _prepare_output(output_path)
    fig, axes = plt.subplots(3, 1, figsize=(8, 8), sharex=False)

    axes[0].plot(steps, states[:, 0], linewidth=2, label="position")
    axes[0].axhline(0.0, color="black", linewidth=1, linestyle="--", label="goal")
    axes[0].set_title("Planner-Controlled Position")
    axes[0].set_ylabel("position")
    axes[0].legend(loc="best")
    axes[0].grid(True, alpha=0.25)

    axes[1].plot(action_steps, actions, color="tab:orange", linewidth=1.5)
    axes[1].set_title("Actions Selected by Random Shooting")
    axes[1].set_ylabel("acceleration")
    axes[1].set_ylim(-1.05, 1.05)
    axes[1].grid(True, alpha=0.25)

    axes[2].plot(action_steps, cumulative_reward, color="tab:green", linewidth=2)
    axes[2].set_title("Cumulative Reward")
    axes[2].set_xlabel("step")
    axes[2].set_ylabel("return")
    axes[2].grid(True, alpha=0.25)

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_partial_observation_ambiguity(
    episodes: Sequence[Episode],
    output_path: Path,
    max_points: int = 1000,
) -> None:
    """Show that one observed position can map to many hidden velocities."""

    true_states = np.concatenate([episode["true_state"] for episode in episodes], axis=0)
    if len(true_states) > max_points:
        true_states = true_states[:max_points]

    _prepare_output(output_path)
    fig, ax = plt.subplots(figsize=(7, 5))
    scatter = ax.scatter(
        true_states[:, 0],
        true_states[:, 1],
        c=true_states[:, 1],
        s=9,
        alpha=0.6,
        cmap="coolwarm",
    )
    ax.axvline(0.0, color="black", linewidth=1, linestyle="--", label="goal")
    ax.set_title("Partial Observation Ambiguity")
    ax.set_xlabel("observed position")
    ax.set_ylabel("hidden velocity")
    ax.legend(loc="best")
    fig.colorbar(scatter, ax=ax, label="hidden velocity")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_gru_sequence_prediction(
    true_next_obs: np.ndarray,
    pred_next_obs: np.ndarray,
    output_path: Path,
) -> None:
    """Compare a real position sequence with GRU one-step predictions."""

    steps = np.arange(len(true_next_obs))

    _prepare_output(output_path)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(steps, true_next_obs[:, 0], label="real next position", linewidth=2)
    ax.plot(
        steps,
        pred_next_obs[:, 0],
        label="predicted next position",
        linewidth=2,
        linestyle="--",
    )
    ax.axhline(0.0, color="black", linewidth=1, linestyle=":", label="goal")
    ax.set_title("GRU World Model Sequence Prediction")
    ax.set_xlabel("time step")
    ax.set_ylabel("position")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_hidden_state_velocity_probe(
    hidden_states: np.ndarray,
    episode: Episode,
    output_path: Path,
) -> None:
    """Visualize whether a hidden dimension tracks the unobserved velocity."""

    velocity = episode["true_state"][:, 1]
    correlations = [
        abs(np.corrcoef(hidden_states[:, dim], velocity)[0, 1])
        for dim in range(hidden_states.shape[1])
    ]
    best_dim = int(np.nanargmax(correlations))
    hidden_signal = hidden_states[:, best_dim]
    hidden_signal = (hidden_signal - hidden_signal.mean()) / (hidden_signal.std() + 1e-8)
    velocity_signal = (velocity - velocity.mean()) / (velocity.std() + 1e-8)
    steps = np.arange(len(velocity))

    _prepare_output(output_path)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(steps, velocity_signal, label="hidden true velocity", linewidth=2)
    ax.plot(
        steps,
        hidden_signal,
        label=f"GRU hidden dim {best_dim}",
        linewidth=2,
        linestyle="--",
    )
    ax.set_title("Hidden State Probe")
    ax.set_xlabel("time step")
    ax.set_ylabel("standardized value")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_reward_landscape(output_path: Path, goal: float = 0.0) -> None:
    """Show how the toy reward depends on next position and action size."""

    positions = np.linspace(-2.0, 2.0, 120)
    actions = np.linspace(-1.0, 1.0, 100)
    grid_x, grid_a = np.meshgrid(positions, actions)
    rewards = -((grid_x - goal) ** 2 + 0.1 * grid_a**2)

    _prepare_output(output_path)
    fig, ax = plt.subplots(figsize=(7.5, 5))
    mesh = ax.contourf(grid_x, grid_a, rewards, levels=24, cmap="viridis")
    ax.axvline(goal, color="white", linewidth=1.2, linestyle="--", label="goal")
    ax.set_title("Toy Reward Landscape")
    ax.set_xlabel("next position")
    ax.set_ylabel("action")
    ax.legend(loc="best")
    fig.colorbar(mesh, ax=ax, label="reward")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_reward_predictions(
    true_reward: np.ndarray,
    pred_reward: np.ndarray,
    output_path: Path,
) -> None:
    """Compare true rewards with model-predicted rewards."""

    _prepare_output(output_path)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    steps = np.arange(len(true_reward))

    axes[0].plot(steps, true_reward[:, 0], label="true reward", linewidth=2)
    axes[0].plot(
        steps,
        pred_reward[:, 0],
        label="predicted reward",
        linewidth=2,
        linestyle="--",
    )
    axes[0].set_title("Reward Along One Episode")
    axes[0].set_xlabel("time step")
    axes[0].set_ylabel("reward")
    axes[0].legend(loc="best")
    axes[0].grid(True, alpha=0.25)

    _plot_prediction_panel(axes[1], true_reward[:, 0], pred_reward[:, 0], "reward")
    axes[1].set_title("Reward Prediction Scatter")

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_action_reward_scores(
    actions: np.ndarray,
    scores: np.ndarray,
    output_path: Path,
) -> None:
    """Show how predicted reward ranks candidate actions."""

    best_idx = int(np.argmax(scores))

    _prepare_output(output_path)
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.plot(actions, scores, linewidth=2)
    ax.scatter(
        [actions[best_idx]],
        [scores[best_idx]],
        s=60,
        label=f"best action = {actions[best_idx]:.2f}",
    )
    ax.set_title("Reward-Guided One-Step Action Scores")
    ax.set_xlabel("candidate action")
    ax.set_ylabel("predicted reward")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_first_planning_rollouts(
    imagined_rollouts: Sequence[np.ndarray],
    best_rollout: np.ndarray,
    output_path: Path,
) -> None:
    """Show top imagined futures from the first planning step."""

    _prepare_output(output_path)
    fig, ax = plt.subplots(figsize=(8, 4.5))

    for rollout in imagined_rollouts:
        steps = np.arange(len(rollout))
        ax.plot(steps, rollout[:, 0], linewidth=1, alpha=0.35)

    if len(best_rollout) > 0:
        steps = np.arange(len(best_rollout))
        ax.plot(
            steps,
            best_rollout[:, 0],
            linewidth=2.5,
            label="best imagined rollout",
        )

    ax.axhline(0.0, color="black", linewidth=1, linestyle="--", label="goal")
    ax.set_title("First Planning Step: Imagined Position Rollouts")
    ax.set_xlabel("imagined step")
    ax.set_ylabel("predicted position")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_receding_horizon_control(
    planning_log: dict[str, np.ndarray | list[np.ndarray]],
    output_path: Path,
) -> None:
    """Show the real trajectory produced by repeated world-model planning."""

    true_states = np.asarray(planning_log["true_states"])
    actions = np.asarray(planning_log["actions"]).reshape(-1)
    rewards = np.asarray(planning_log["real_rewards"]).reshape(-1)
    steps = np.arange(len(true_states))
    action_steps = np.arange(len(actions))

    _prepare_output(output_path)
    fig, axes = plt.subplots(3, 1, figsize=(8, 8), sharex=False)

    axes[0].plot(steps, true_states[:, 0], linewidth=2, label="real position")
    axes[0].axhline(0.0, color="black", linewidth=1, linestyle="--", label="goal")
    axes[0].set_title("Receding-Horizon Planning in the Real Environment")
    axes[0].set_ylabel("position")
    axes[0].legend(loc="best")
    axes[0].grid(True, alpha=0.25)

    axes[1].plot(action_steps, actions, color="tab:orange", linewidth=1.5)
    axes[1].set_title("First Action Executed After Each Replan")
    axes[1].set_ylabel("acceleration")
    axes[1].set_ylim(-1.05, 1.05)
    axes[1].grid(True, alpha=0.25)

    axes[2].plot(action_steps, np.cumsum(rewards), color="tab:green", linewidth=2)
    axes[2].set_title("Real Cumulative Reward")
    axes[2].set_xlabel("real environment step")
    axes[2].set_ylabel("return")
    axes[2].grid(True, alpha=0.25)

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_imagined_vs_real_feedback(
    planning_log: dict[str, np.ndarray | list[np.ndarray]],
    output_path: Path,
) -> None:
    """Compare model-estimated plan quality with real received rewards."""

    imagined_returns = np.asarray(planning_log["imagined_returns"]).reshape(-1)
    real_rewards = np.asarray(planning_log["real_rewards"]).reshape(-1)
    steps = np.arange(len(real_rewards))

    _prepare_output(output_path)
    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

    axes[0].plot(steps, imagined_returns, linewidth=2)
    axes[0].set_title("Best Imagined Return at Each Replan")
    axes[0].set_ylabel("imagined return")
    axes[0].grid(True, alpha=0.25)

    axes[1].plot(steps, real_rewards, color="tab:green", linewidth=2)
    axes[1].set_title("Real Reward Received After Executing First Action")
    axes[1].set_xlabel("real environment step")
    axes[1].set_ylabel("real reward")
    axes[1].grid(True, alpha=0.25)

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_rollout_prediction_comparison(
    real_obs: np.ndarray,
    obs_rollout: np.ndarray,
    latent_rollout: np.ndarray,
    output_path: Path,
) -> None:
    """Compare real, observation-space, and latent-space open-loop rollouts."""

    steps = np.arange(len(real_obs))

    _prepare_output(output_path)
    fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True)

    axes[0].plot(steps, real_obs[:, 0], label="real position", linewidth=2)
    axes[0].plot(
        steps,
        obs_rollout[:, 0],
        label="observation rollout",
        linewidth=1.8,
        linestyle="--",
    )
    axes[0].plot(
        steps,
        latent_rollout[:, 0],
        label="latent rollout",
        linewidth=1.8,
        linestyle=":",
    )
    axes[0].axhline(0.0, color="black", linewidth=1, linestyle="--", label="goal")
    axes[0].set_title("Open-Loop Rollout Position")
    axes[0].set_ylabel("position")
    axes[0].legend(loc="best")
    axes[0].grid(True, alpha=0.25)

    axes[1].plot(steps, real_obs[:, 1], label="real velocity", linewidth=2)
    axes[1].plot(
        steps,
        obs_rollout[:, 1],
        label="observation rollout",
        linewidth=1.8,
        linestyle="--",
    )
    axes[1].plot(
        steps,
        latent_rollout[:, 1],
        label="latent rollout",
        linewidth=1.8,
        linestyle=":",
    )
    axes[1].set_title("Open-Loop Rollout Velocity")
    axes[1].set_xlabel("rollout step")
    axes[1].set_ylabel("velocity")
    axes[1].legend(loc="best")
    axes[1].grid(True, alpha=0.25)

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_rollout_error_growth(
    real_obs: np.ndarray,
    obs_rollout: np.ndarray,
    latent_rollout: np.ndarray,
    output_path: Path,
) -> None:
    """Show how open-loop prediction error grows over rollout horizon."""

    steps = np.arange(len(real_obs))
    obs_error = np.linalg.norm(obs_rollout - real_obs, axis=1)
    latent_error = np.linalg.norm(latent_rollout - real_obs, axis=1)

    _prepare_output(output_path)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(steps, obs_error, label="observation rollout error", linewidth=2)
    ax.plot(steps, latent_error, label="latent rollout error", linewidth=2)
    ax.set_title("Compounding Error During Open-Loop Rollout")
    ax.set_xlabel("rollout step")
    ax.set_ylabel("state prediction error")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_latent_space_rollout(
    latents: np.ndarray,
    output_path: Path,
) -> None:
    """Visualize the imagined trajectory inside the learned latent space."""

    _prepare_output(output_path)
    fig, ax = plt.subplots(figsize=(6, 5))

    if latents.shape[1] == 1:
        ax.plot(np.arange(len(latents)), latents[:, 0], linewidth=2)
        ax.set_xlabel("rollout step")
        ax.set_ylabel("latent dim 0")
    else:
        steps = np.arange(len(latents))
        scatter = ax.scatter(
            latents[:, 0],
            latents[:, 1],
            c=steps,
            s=28,
            cmap="viridis",
        )
        ax.plot(latents[:, 0], latents[:, 1], linewidth=1, alpha=0.5)
        ax.set_xlabel("latent dim 0")
        ax.set_ylabel("latent dim 1")
        fig.colorbar(scatter, ax=ax, label="rollout step")

    ax.set_title("Latent-Space Rollout Trajectory")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_one_step_future_distribution(
    obs: np.ndarray,
    action: np.ndarray,
    true_next_obs: np.ndarray,
    sampled_next_obs: np.ndarray,
    output_path: Path,
) -> None:
    """Show many plausible one-step futures from a stochastic latent model."""

    _prepare_output(output_path)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(
        sampled_next_obs[:, 0],
        sampled_next_obs[:, 1],
        s=18,
        alpha=0.45,
        label="sampled next states",
    )
    ax.scatter(
        [obs[0]],
        [obs[1]],
        s=80,
        marker="x",
        label=f"current state, action={float(action[0]):.2f}",
    )
    ax.scatter(
        [true_next_obs[0]],
        [true_next_obs[1]],
        s=70,
        marker="*",
        label="one real next state",
    )
    ax.set_title("One-Step Stochastic Future Distribution")
    ax.set_xlabel("position")
    ax.set_ylabel("velocity")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_stochastic_rollout_fan(
    real_obs: np.ndarray,
    sampled_rollouts: np.ndarray,
    output_path: Path,
) -> None:
    """Show a fan of sampled stochastic futures against one real rollout."""

    steps = np.arange(sampled_rollouts.shape[1])
    position_samples = sampled_rollouts[:, :, 0]
    lower = np.percentile(position_samples, 10, axis=0)
    upper = np.percentile(position_samples, 90, axis=0)
    mean = position_samples.mean(axis=0)

    _prepare_output(output_path)
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for rollout in sampled_rollouts[:20]:
        ax.plot(steps, rollout[:, 0], linewidth=0.8, alpha=0.18)
    ax.fill_between(steps, lower, upper, alpha=0.25, label="10-90% sample band")
    ax.plot(steps, mean, linewidth=2.2, label="sample mean")
    ax.plot(steps, real_obs[:, 0], linewidth=2, linestyle="--", label="one real rollout")
    ax.axhline(0.0, color="black", linewidth=1, linestyle=":", label="goal")
    ax.set_title("Stochastic Latent Open-Loop Rollout Fan")
    ax.set_xlabel("rollout step")
    ax.set_ylabel("position")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_stochastic_uncertainty_growth(
    sampled_rollouts: np.ndarray,
    output_path: Path,
) -> None:
    """Show how sampled future uncertainty changes over rollout horizon."""

    steps = np.arange(sampled_rollouts.shape[1])
    position_std = sampled_rollouts[:, :, 0].std(axis=0)
    velocity_std = sampled_rollouts[:, :, 1].std(axis=0)

    _prepare_output(output_path)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(steps, position_std, label="position std", linewidth=2)
    ax.plot(steps, velocity_std, label="velocity std", linewidth=2)
    ax.set_title("Uncertainty Across Sampled Futures")
    ax.set_xlabel("rollout step")
    ax.set_ylabel("sample standard deviation")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_prior_latent_samples(
    latent_samples: np.ndarray,
    output_path: Path,
) -> None:
    """Visualize stochastic next latent samples from the learned prior."""

    _prepare_output(output_path)
    fig, ax = plt.subplots(figsize=(6, 5))

    if latent_samples.shape[1] == 1:
        ax.hist(latent_samples[:, 0], bins=30, alpha=0.75)
        ax.set_xlabel("latent dim 0")
        ax.set_ylabel("count")
    else:
        ax.scatter(latent_samples[:, 0], latent_samples[:, 1], s=16, alpha=0.5)
        ax.set_xlabel("latent dim 0")
        ax.set_ylabel("latent dim 1")

    ax.set_title("Samples from p(z_next | z, action)")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_prior_vs_posterior_latent_means(
    analysis: dict[str, np.ndarray],
    output_path: Path,
    max_lines: int = 120,
) -> None:
    """Show where prior and posterior latent means land for real transitions."""

    prior = analysis["prior_mean"]
    posterior = analysis["posterior_mean"]
    count = min(len(prior), max_lines)

    _prepare_output(output_path)
    fig, ax = plt.subplots(figsize=(7, 5.5))

    if prior.shape[1] == 1:
        steps = np.arange(len(prior))
        ax.plot(steps, prior[:, 0], label="prior mean", linewidth=1.8)
        ax.plot(steps, posterior[:, 0], label="posterior mean", linewidth=1.8)
        ax.set_xlabel("transition index")
        ax.set_ylabel("latent dim 0")
    else:
        ax.scatter(
            prior[:, 0],
            prior[:, 1],
            s=18,
            alpha=0.45,
            label="prior mean",
        )
        ax.scatter(
            posterior[:, 0],
            posterior[:, 1],
            s=18,
            alpha=0.45,
            label="posterior mean",
        )
        for i in range(count):
            ax.plot(
                [prior[i, 0], posterior[i, 0]],
                [prior[i, 1], posterior[i, 1]],
                color="gray",
                linewidth=0.6,
                alpha=0.25,
            )
        ax.set_xlabel("latent dim 0")
        ax.set_ylabel("latent dim 1")

    ax.set_title("Prior vs Posterior Latent Means")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_prior_posterior_latent_samples(
    samples: dict[str, np.ndarray],
    output_path: Path,
) -> None:
    """Show prior and posterior latent sample clouds for one transition."""

    prior = samples["prior_samples"]
    posterior = samples["posterior_samples"]
    prior_mean = samples["prior_mean"]
    posterior_mean = samples["posterior_mean"]

    _prepare_output(output_path)
    fig, ax = plt.subplots(figsize=(7, 5.5))

    if prior.shape[1] == 1:
        ax.hist(prior[:, 0], bins=30, alpha=0.55, label="prior samples")
        ax.hist(posterior[:, 0], bins=30, alpha=0.55, label="posterior samples")
        ax.axvline(prior_mean[0], color="tab:blue", linestyle="--", label="prior mean")
        ax.axvline(
            posterior_mean[0],
            color="tab:orange",
            linestyle="--",
            label="posterior mean",
        )
        ax.set_xlabel("latent dim 0")
        ax.set_ylabel("count")
    else:
        ax.scatter(
            prior[:, 0],
            prior[:, 1],
            s=18,
            alpha=0.42,
            label="prior samples",
        )
        ax.scatter(
            posterior[:, 0],
            posterior[:, 1],
            s=18,
            alpha=0.42,
            label="posterior samples",
        )
        ax.scatter(
            [prior_mean[0]],
            [prior_mean[1]],
            s=90,
            marker="x",
            linewidths=2.5,
            label="prior mean",
        )
        ax.scatter(
            [posterior_mean[0]],
            [posterior_mean[1]],
            s=95,
            marker="*",
            label="posterior mean",
        )
        ax.set_xlabel("latent dim 0")
        ax.set_ylabel("latent dim 1")

    ax.set_title("Prior and Posterior Samples for One Transition")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_prior_posterior_alignment(
    analysis: dict[str, np.ndarray],
    output_path: Path,
) -> None:
    """Summarize KL and mean-distance alignment between prior and posterior."""

    kl = analysis["kl"]
    mean_distance = analysis["mean_distance"]

    _prepare_output(output_path)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    axes[0].hist(kl, bins=35, alpha=0.75)
    axes[0].axvline(float(np.mean(kl)), color="black", linestyle="--", linewidth=1.2)
    axes[0].set_title("KL(q posterior || p prior)")
    axes[0].set_xlabel("KL value")
    axes[0].set_ylabel("transition count")
    axes[0].grid(True, alpha=0.25)

    axes[1].hist(mean_distance, bins=35, alpha=0.75, color="tab:orange")
    axes[1].axvline(
        float(np.mean(mean_distance)),
        color="black",
        linestyle="--",
        linewidth=1.2,
    )
    axes[1].set_title("Latent Mean Distance")
    axes[1].set_xlabel("||posterior mean - prior mean||")
    axes[1].set_ylabel("transition count")
    axes[1].grid(True, alpha=0.25)

    fig.suptitle("Prior / Posterior Alignment on Real Transitions")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_prior_decoded_next_state(
    analysis: dict[str, np.ndarray],
    output_path: Path,
) -> None:
    """Compare true next observations with next observations decoded from prior."""

    true_next = analysis["true_next_obs"]
    prior_decoded = analysis["prior_decoded_next_obs"]

    _prepare_output(output_path)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    _plot_prediction_panel(axes[0], true_next[:, 0], prior_decoded[:, 0], "next position")
    _plot_prediction_panel(axes[1], true_next[:, 1], prior_decoded[:, 1], "next velocity")
    fig.suptitle("Decode Next State from Prior Mean")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _plot_prediction_panel(
    ax: plt.Axes,
    target: np.ndarray,
    prediction: np.ndarray,
    label: str,
) -> None:
    ax.scatter(target, prediction, s=8, alpha=0.55)
    min_value = float(min(target.min(), prediction.min()))
    max_value = float(max(target.max(), prediction.max()))
    ax.plot([min_value, max_value], [min_value, max_value], color="black", linewidth=1)
    ax.set_xlabel(f"real {label}")
    ax.set_ylabel(f"predicted {label}")
    ax.grid(True, alpha=0.25)


def _prepare_output(output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
