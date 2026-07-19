from __future__ import annotations

from pathlib import Path
from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch

from .models import WorldModel
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
