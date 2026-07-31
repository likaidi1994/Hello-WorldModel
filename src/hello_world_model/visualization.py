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
