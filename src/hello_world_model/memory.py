from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
from torch import nn, optim

from .envs import PartialPointMass1DEnv
from .models import GRURewardWorldModel, GRUWorldModel

Episode = dict[str, np.ndarray]


def collect_partial_episodes(
    env: PartialPointMass1DEnv,
    episodes: int = 200,
) -> list[Episode]:
    """Collect full episodes where observations hide the true velocity."""

    all_episodes: list[Episode] = []

    for _ in range(episodes):
        obs_seq: list[np.ndarray] = []
        action_seq: list[np.ndarray] = []
        next_obs_seq: list[np.ndarray] = []
        reward_seq: list[np.ndarray] = []
        true_state_seq: list[np.ndarray] = []

        obs = env.reset()
        done = False

        while not done:
            action_value = float(np.random.uniform(-1.0, 1.0))
            next_obs, reward, done, _ = env.step(action_value)

            obs_seq.append(obs.copy())
            action_seq.append(np.array([action_value], dtype=np.float32))
            next_obs_seq.append(next_obs.copy())
            reward_seq.append(np.array([reward], dtype=np.float32))
            true_state_seq.append(env.get_true_state())

            obs = next_obs

        all_episodes.append(
            {
                "obs": np.stack(obs_seq),
                "action": np.stack(action_seq),
                "next_obs": np.stack(next_obs_seq),
                "reward": np.stack(reward_seq),
                "true_state": np.stack(true_state_seq),
            }
        )

    return all_episodes


def train_gru_world_model(
    model: GRUWorldModel,
    episodes: Sequence[Episode],
    epochs: int = 50,
    lr: float = 1e-3,
    device: str = "cpu",
) -> list[float]:
    """Train a GRU model to predict the next position-only observation."""

    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    mse = nn.MSELoss()
    losses: list[float] = []

    for epoch in range(epochs):
        total_loss = 0.0

        for episode in episodes:
            obs = _to_sequence_tensor(episode["obs"], device)
            action = _to_sequence_tensor(episode["action"], device)
            next_obs = _to_sequence_tensor(episode["next_obs"], device)

            pred_next_obs, _ = model(obs, action)
            loss = mse(pred_next_obs, next_obs)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(episodes)
        losses.append(avg_loss)
        print(f"Epoch {epoch + 1:03d} | loss = {avg_loss:.6f}")

    return losses


@torch.no_grad()
def predict_episode(
    model: GRUWorldModel,
    episode: Episode,
    device: str = "cpu",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return true next observations, predicted next observations, and hidden states."""

    model.eval()
    obs = _to_sequence_tensor(episode["obs"], device)
    action = _to_sequence_tensor(episode["action"], device)
    pred_next_obs, _ = model(obs, action)

    x = torch.cat([obs, action], dim=-1)
    hidden_seq, _ = model.gru(x)

    return (
        episode["next_obs"],
        pred_next_obs.squeeze(0).cpu().numpy(),
        hidden_seq.squeeze(0).cpu().numpy(),
    )


def train_gru_reward_world_model(
    model: GRURewardWorldModel,
    episodes: Sequence[Episode],
    epochs: int = 50,
    lr: float = 1e-3,
    reward_loss_weight: float = 1.0,
    device: str = "cpu",
) -> dict[str, list[float]]:
    """Train a recurrent model to predict next observation and reward."""

    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    mse = nn.MSELoss()
    losses = {"total": [], "obs": [], "reward": []}

    for epoch in range(epochs):
        total_loss = 0.0
        total_obs_loss = 0.0
        total_reward_loss = 0.0

        for episode in episodes:
            obs = _to_sequence_tensor(episode["obs"], device)
            action = _to_sequence_tensor(episode["action"], device)
            next_obs = _to_sequence_tensor(episode["next_obs"], device)
            reward = _to_sequence_tensor(episode["reward"], device)

            pred_next_obs, pred_reward, _ = model(obs, action)
            obs_loss = mse(pred_next_obs, next_obs)
            reward_loss = mse(pred_reward, reward)
            loss = obs_loss + reward_loss_weight * reward_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_obs_loss += obs_loss.item()
            total_reward_loss += reward_loss.item()

        n = len(episodes)
        losses["total"].append(total_loss / n)
        losses["obs"].append(total_obs_loss / n)
        losses["reward"].append(total_reward_loss / n)
        print(
            f"Epoch {epoch + 1:03d} | "
            f"total={losses['total'][-1]:.6f} | "
            f"obs={losses['obs'][-1]:.6f} | "
            f"reward={losses['reward'][-1]:.6f}"
        )

    return losses


@torch.no_grad()
def predict_episode_with_reward(
    model: GRURewardWorldModel,
    episode: Episode,
    device: str = "cpu",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return true/predicted observations and true/predicted rewards."""

    model.eval()
    obs = _to_sequence_tensor(episode["obs"], device)
    action = _to_sequence_tensor(episode["action"], device)
    pred_next_obs, pred_reward, _ = model(obs, action)

    return (
        episode["next_obs"],
        pred_next_obs.squeeze(0).cpu().numpy(),
        episode["reward"],
        pred_reward.squeeze(0).cpu().numpy(),
    )


@torch.no_grad()
def score_candidate_actions(
    model: GRURewardWorldModel,
    episode: Episode,
    step_index: int,
    action_values: np.ndarray,
    device: str = "cpu",
) -> np.ndarray:
    """Score candidate actions after summarizing history into hidden memory."""

    model.eval()
    step_index = int(np.clip(step_index, 0, len(episode["obs"]) - 1))

    h0 = None
    if step_index > 0:
        history_obs = _to_sequence_tensor(episode["obs"][:step_index], device)
        history_action = _to_sequence_tensor(episode["action"][:step_index], device)
        _, _, h0 = model(history_obs, history_action)
        h0 = h0.repeat(1, len(action_values), 1)

    obs_value = float(episode["obs"][step_index, 0])
    obs = torch.tensor(
        [[[obs_value]] for _ in action_values], dtype=torch.float32, device=device
    )
    actions = torch.tensor(
        action_values.reshape(-1, 1, 1), dtype=torch.float32, device=device
    )
    _, pred_reward, _ = model(obs, actions, h0=h0)
    return pred_reward.squeeze(-1).squeeze(-1).cpu().numpy()


def _to_sequence_tensor(array: np.ndarray, device: str) -> torch.Tensor:
    return torch.tensor(array, dtype=torch.float32, device=device).unsqueeze(0)
