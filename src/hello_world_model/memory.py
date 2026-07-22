from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
from torch import nn, optim

from .envs import PartialPointMass1DEnv
from .models import GRUWorldModel

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
        true_state_seq: list[np.ndarray] = []

        obs = env.reset()
        done = False

        while not done:
            action_value = float(np.random.uniform(-1.0, 1.0))
            next_obs, _, done, _ = env.step(action_value)

            obs_seq.append(obs.copy())
            action_seq.append(np.array([action_value], dtype=np.float32))
            next_obs_seq.append(next_obs.copy())
            true_state_seq.append(env.get_true_state())

            obs = next_obs

        all_episodes.append(
            {
                "obs": np.stack(obs_seq),
                "action": np.stack(action_seq),
                "next_obs": np.stack(next_obs_seq),
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


def _to_sequence_tensor(array: np.ndarray, device: str) -> torch.Tensor:
    return torch.tensor(array, dtype=torch.float32, device=device).unsqueeze(0)
