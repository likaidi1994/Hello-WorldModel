from __future__ import annotations

import numpy as np
import torch

from .models import WorldModel


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
