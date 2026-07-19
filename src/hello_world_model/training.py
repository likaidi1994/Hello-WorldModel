from __future__ import annotations

import random
from collections.abc import Sequence

import numpy as np
import torch
from torch import nn, optim

from .envs import PointMass1DEnv
from .models import WorldModel
from .planning import plan_action_random_shooting

Transition = tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]


def collect_random_data(env: PointMass1DEnv, episodes: int = 200) -> list[Transition]:
    data: list[Transition] = []

    for _ in range(episodes):
        state = env.reset()
        done = False

        while not done:
            action_value = np.random.uniform(-1.0, 1.0)
            next_state, reward, done, _ = env.step(action_value)

            action = np.array([action_value], dtype=np.float32)
            reward_array = np.array([reward], dtype=np.float32)
            data.append((state.copy(), action, next_state.copy(), reward_array))
            state = next_state

    return data


def train_world_model(
    model: WorldModel,
    data: Sequence[Transition],
    epochs: int = 50,
    batch_size: int = 128,
    lr: float = 1e-3,
    device: str = "cpu",
) -> list[float]:
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    mse = nn.MSELoss()
    shuffled_data = list(data)

    losses: list[float] = []

    for epoch in range(epochs):
        random.shuffle(shuffled_data)
        total_loss = 0.0

        for i in range(0, len(shuffled_data), batch_size):
            batch = shuffled_data[i : i + batch_size]
            state = torch.tensor(
                np.stack([x[0] for x in batch]), dtype=torch.float32, device=device
            )
            action = torch.tensor(
                np.stack([x[1] for x in batch]), dtype=torch.float32, device=device
            )
            next_state = torch.tensor(
                np.stack([x[2] for x in batch]), dtype=torch.float32, device=device
            )
            reward = torch.tensor(
                np.stack([x[3] for x in batch]), dtype=torch.float32, device=device
            )

            pred_next_state, pred_reward = model(state, action)
            loss = mse(pred_next_state, next_state) + mse(pred_reward, reward)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(batch)

        avg_loss = total_loss / len(shuffled_data)
        losses.append(avg_loss)
        print(f"Epoch {epoch + 1:03d} | loss = {avg_loss:.6f}")

    return losses


def evaluate_planner(
    env: PointMass1DEnv,
    model: WorldModel,
    episodes: int = 10,
    device: str = "cpu",
) -> list[float]:
    returns: list[float] = []

    for episode in range(episodes):
        state = env.reset()
        done = False
        episode_return = 0.0

        while not done:
            action = plan_action_random_shooting(
                model, state, horizon=15, num_samples=300, device=device
            )
            state, reward, done, _ = env.step(action)
            episode_return += reward

        returns.append(episode_return)
        print(f"Episode {episode + 1:02d} return = {episode_return:.2f}")

    print(f"Average return: {np.mean(returns):.2f}")
    return returns


def rollout_planner_episode(
    env: PointMass1DEnv,
    model: WorldModel,
    device: str = "cpu",
) -> tuple[np.ndarray, np.ndarray, list[float]]:
    """Run one planned episode and keep states/actions for visualization."""

    states: list[np.ndarray] = []
    actions: list[float] = []
    rewards: list[float] = []

    state = env.reset()
    done = False

    while not done:
        states.append(state.copy())
        action = plan_action_random_shooting(
            model, state, horizon=15, num_samples=300, device=device
        )
        state, reward, done, _ = env.step(action)
        actions.append(action)
        rewards.append(reward)

    states.append(state.copy())
    return np.stack(states), np.array(actions, dtype=np.float32), rewards
