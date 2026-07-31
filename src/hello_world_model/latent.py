from __future__ import annotations

import random
from collections.abc import Sequence

import numpy as np
import torch
from torch import nn, optim

from .envs import PointMass1DEnv
from .models import LatentDynamicsWorldModel, WorldModel
from .training import Transition


def train_latent_dynamics_model(
    model: LatentDynamicsWorldModel,
    data: Sequence[Transition],
    epochs: int = 50,
    batch_size: int = 128,
    lr: float = 1e-3,
    device: str = "cpu",
) -> dict[str, list[float]]:
    """Train a minimal latent dynamics model on transition data."""

    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    mse = nn.MSELoss()
    shuffled_data = list(data)
    losses = {"total": [], "obs": [], "reward": []}

    for epoch in range(epochs):
        random.shuffle(shuffled_data)
        total_loss = 0.0
        total_obs_loss = 0.0
        total_reward_loss = 0.0

        for i in range(0, len(shuffled_data), batch_size):
            batch = shuffled_data[i : i + batch_size]
            obs = torch.tensor(
                np.stack([x[0] for x in batch]), dtype=torch.float32, device=device
            )
            action = torch.tensor(
                np.stack([x[1] for x in batch]), dtype=torch.float32, device=device
            )
            next_obs = torch.tensor(
                np.stack([x[2] for x in batch]), dtype=torch.float32, device=device
            )
            reward = torch.tensor(
                np.stack([x[3] for x in batch]), dtype=torch.float32, device=device
            )

            pred_next_obs, pred_reward, _ = model(obs, action)
            obs_loss = mse(pred_next_obs, next_obs)
            reward_loss = mse(pred_reward, reward)
            loss = obs_loss + reward_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(batch)
            total_obs_loss += obs_loss.item() * len(batch)
            total_reward_loss += reward_loss.item() * len(batch)

        n = len(shuffled_data)
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


def collect_real_rollout(
    env: PointMass1DEnv,
    action_sequence: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Run a fixed action sequence in the real environment."""

    obs = env.reset()
    observations = [obs.copy()]
    rewards: list[float] = []

    for action in action_sequence:
        obs, reward, done, _ = env.step(float(action[0]))
        observations.append(obs.copy())
        rewards.append(reward)
        if done:
            break

    return np.stack(observations), np.array(rewards, dtype=np.float32)


@torch.no_grad()
def rollout_observation_model(
    model: WorldModel,
    initial_obs: np.ndarray,
    action_sequence: np.ndarray,
    device: str = "cpu",
) -> tuple[np.ndarray, np.ndarray]:
    """Open-loop rollout that feeds predicted observations back into the model."""

    model.eval()
    obs = torch.tensor(initial_obs, dtype=torch.float32, device=device).unsqueeze(0)
    observations = [obs.squeeze(0).cpu().numpy()]
    rewards: list[float] = []

    for action_np in action_sequence:
        action = torch.tensor(action_np, dtype=torch.float32, device=device).unsqueeze(0)
        obs, reward = model(obs, action)
        observations.append(obs.squeeze(0).cpu().numpy())
        rewards.append(float(reward.item()))

    return np.stack(observations), np.array(rewards, dtype=np.float32)


@torch.no_grad()
def rollout_latent_dynamics_model(
    model: LatentDynamicsWorldModel,
    initial_obs: np.ndarray,
    action_sequence: np.ndarray,
    device: str = "cpu",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Open-loop rollout that advances the model in latent space."""

    model.eval()
    obs = torch.tensor(initial_obs, dtype=torch.float32, device=device).unsqueeze(0)
    latent = model.encoder(obs)
    latents = [latent.squeeze(0).cpu().numpy()]
    observations = [obs.squeeze(0).cpu().numpy()]
    rewards: list[float] = []

    for action_np in action_sequence:
        action = torch.tensor(action_np, dtype=torch.float32, device=device).unsqueeze(0)
        latent, reward = model.step_latent(latent, action)
        decoded_obs = model.decoder(latent)
        latents.append(latent.squeeze(0).cpu().numpy())
        observations.append(decoded_obs.squeeze(0).cpu().numpy())
        rewards.append(float(reward.item()))

    return (
        np.stack(observations),
        np.array(rewards, dtype=np.float32),
        np.stack(latents),
    )


def make_probe_action_sequence(horizon: int = 40) -> np.ndarray:
    """Create a smooth action sequence for open-loop rollout comparison."""

    steps = np.linspace(0.0, 2.5 * np.pi, horizon)
    actions = 0.8 * np.sin(steps)
    return actions.reshape(-1, 1).astype(np.float32)
