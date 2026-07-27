from __future__ import annotations

import numpy as np
import torch

from .envs import PartialPointMass1DEnv
from .models import GRURewardWorldModel, WorldModel


@torch.no_grad()
def plan_action_random_shooting(
    model: WorldModel,
    current_state: np.ndarray,
    horizon: int = 15,
    num_samples: int = 500,
    device: str = "cpu",
) -> float:
    """Sample candidate action sequences and return the best first action."""

    model.eval()
    best_return = -float("inf")
    best_first_action = 0.0

    initial_state = torch.tensor(
        current_state, dtype=torch.float32, device=device
    ).unsqueeze(0)

    for _ in range(num_samples):
        state = initial_state.clone()
        total_return = 0.0
        action_seq = np.random.uniform(-1.0, 1.0, size=(horizon, 1)).astype(
            np.float32
        )

        for action_np in action_seq:
            action = torch.tensor(action_np, dtype=torch.float32, device=device).unsqueeze(
                0
            )
            next_state, reward = model(state, action)
            total_return += float(reward.item())
            state = next_state

        if total_return > best_return:
            best_return = total_return
            best_first_action = float(action_seq[0, 0])

    return best_first_action


@torch.no_grad()
def infer_hidden_from_history(
    model: GRURewardWorldModel,
    obs_history: np.ndarray,
    action_history: np.ndarray,
    device: str = "cpu",
) -> torch.Tensor | None:
    """Infer the current recurrent hidden state from real interaction history."""

    if len(obs_history) == 0:
        return None

    model.eval()
    obs = torch.tensor(obs_history, dtype=torch.float32, device=device).unsqueeze(0)
    action = torch.tensor(action_history, dtype=torch.float32, device=device).unsqueeze(0)
    _, _, hidden = model(obs, action)
    return hidden


@torch.no_grad()
def imagine_action_sequence(
    model: GRURewardWorldModel,
    current_obs: np.ndarray,
    current_hidden: torch.Tensor | None,
    action_sequence: np.ndarray,
    device: str = "cpu",
) -> tuple[float, np.ndarray, np.ndarray]:
    """Roll out one candidate action sequence inside the learned world model."""

    model.eval()
    obs = torch.tensor(current_obs, dtype=torch.float32, device=device).unsqueeze(0)
    hidden = None if current_hidden is None else current_hidden.clone()
    imagined_obs: list[np.ndarray] = [obs.squeeze(0).cpu().numpy()]
    imagined_rewards: list[float] = []
    total_reward = 0.0

    for action_np in action_sequence:
        action = torch.tensor(action_np, dtype=torch.float32, device=device).unsqueeze(0)
        pred_next_obs, pred_reward, hidden = model.step(obs, action, hidden)
        reward_value = float(pred_reward.item())
        total_reward += reward_value
        imagined_rewards.append(reward_value)
        obs = pred_next_obs
        imagined_obs.append(obs.squeeze(0).cpu().numpy())

    return total_reward, np.stack(imagined_obs), np.array(imagined_rewards)


@torch.no_grad()
def plan_action_with_world_model(
    model: GRURewardWorldModel,
    current_obs: np.ndarray,
    current_hidden: torch.Tensor | None,
    horizon: int = 15,
    num_samples: int = 300,
    action_low: float = -1.0,
    action_high: float = 1.0,
    top_k: int = 8,
    device: str = "cpu",
) -> dict[str, np.ndarray | float]:
    """Random-shooting planner over imagined recurrent world-model rollouts."""

    top_rollouts: list[tuple[float, np.ndarray, np.ndarray, np.ndarray]] = []

    for _ in range(num_samples):
        action_sequence = np.random.uniform(
            action_low, action_high, size=(horizon, 1)
        ).astype(np.float32)
        total_reward, imagined_obs, imagined_rewards = imagine_action_sequence(
            model=model,
            current_obs=current_obs,
            current_hidden=current_hidden,
            action_sequence=action_sequence,
            device=device,
        )
        top_rollouts.append(
            (total_reward, action_sequence, imagined_obs, imagined_rewards)
        )
        top_rollouts.sort(key=lambda item: item[0], reverse=True)
        top_rollouts = top_rollouts[:top_k]

    best_return, best_sequence, best_obs, best_rewards = top_rollouts[0]
    return {
        "first_action": best_sequence[0].copy(),
        "best_sequence": best_sequence.copy(),
        "best_return": float(best_return),
        "best_imagined_obs": best_obs.copy(),
        "best_imagined_rewards": best_rewards.copy(),
        "top_returns": np.array([item[0] for item in top_rollouts], dtype=np.float32),
        "top_imagined_obs": [item[2].copy() for item in top_rollouts],
    }


def run_receding_horizon_episode(
    env: PartialPointMass1DEnv,
    model: GRURewardWorldModel,
    horizon: int = 15,
    num_samples: int = 300,
    device: str = "cpu",
) -> dict[str, np.ndarray | list[np.ndarray]]:
    """Control the real environment by repeatedly replanning in the world model."""

    obs = env.reset()
    done = False
    obs_history: list[np.ndarray] = []
    action_history: list[np.ndarray] = []
    true_states = [env.get_true_state()]
    real_rewards: list[float] = []
    imagined_returns: list[float] = []
    first_plan_rollouts: list[np.ndarray] | None = None
    first_best_plan: np.ndarray | None = None

    while not done:
        current_hidden = infer_hidden_from_history(
            model=model,
            obs_history=np.stack(obs_history) if obs_history else np.empty((0, 1)),
            action_history=np.stack(action_history) if action_history else np.empty((0, 1)),
            device=device,
        )
        plan = plan_action_with_world_model(
            model=model,
            current_obs=obs,
            current_hidden=current_hidden,
            horizon=horizon,
            num_samples=num_samples,
            device=device,
        )
        first_action = np.asarray(plan["first_action"], dtype=np.float32)

        if first_plan_rollouts is None:
            first_plan_rollouts = list(plan["top_imagined_obs"])
            first_best_plan = np.asarray(plan["best_imagined_obs"])

        next_obs, reward, done, _ = env.step(float(first_action[0]))
        obs_history.append(obs.copy())
        action_history.append(first_action.copy())
        real_rewards.append(float(reward))
        imagined_returns.append(float(plan["best_return"]))
        obs = next_obs
        true_states.append(env.get_true_state())

    return {
        "observations": np.vstack([item.reshape(1, -1) for item in obs_history] + [obs.reshape(1, -1)]),
        "actions": np.vstack(action_history),
        "true_states": np.vstack(true_states),
        "real_rewards": np.array(real_rewards, dtype=np.float32),
        "imagined_returns": np.array(imagined_returns, dtype=np.float32),
        "first_plan_rollouts": first_plan_rollouts or [],
        "first_best_plan": first_best_plan if first_best_plan is not None else np.empty((0, 1)),
    }
