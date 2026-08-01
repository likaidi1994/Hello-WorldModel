from __future__ import annotations

import numpy as np


class PointMass1DEnv:
    """A tiny 1D point-mass control environment.

    State: [position, velocity]
    Action: acceleration clipped to [-1, 1]
    Reward: negative distance-to-goal penalty and action penalty
    """

    def __init__(self, dt: float = 0.1, horizon: int = 100, goal: float = 0.0):
        self.dt = dt
        self.horizon = horizon
        self.goal = goal
        self.t = 0
        self.pos = 0.0
        self.vel = 0.0

    def reset(self) -> np.ndarray:
        self.t = 0
        self.pos = float(np.random.uniform(-2.0, 2.0))
        self.vel = float(np.random.uniform(-0.5, 0.5))
        return self._get_obs()

    def step(self, action: float) -> tuple[np.ndarray, float, bool, dict]:
        action = float(np.clip(action, -1.0, 1.0))

        self.vel += action * self.dt
        self.pos += self.vel * self.dt
        self.t += 1

        reward = -((self.pos - self.goal) ** 2 + 0.1 * action**2)
        done = self.t >= self.horizon
        return self._get_obs(), float(reward), done, {}

    def _get_obs(self) -> np.ndarray:
        return np.array([self.pos, self.vel], dtype=np.float32)


class PartialPointMass1DEnv(PointMass1DEnv):
    """Point-mass environment where the agent only observes position."""

    def _get_obs(self) -> np.ndarray:
        return np.array([self.pos], dtype=np.float32)

    def get_true_state(self) -> np.ndarray:
        return np.array([self.pos, self.vel], dtype=np.float32)


class NoisyPointMass1DEnv(PointMass1DEnv):
    """Point-mass environment with random process noise in the dynamics."""

    def __init__(
        self,
        dt: float = 0.1,
        horizon: int = 100,
        goal: float = 0.0,
        process_noise_std: float = 0.15,
    ):
        super().__init__(dt=dt, horizon=horizon, goal=goal)
        self.process_noise_std = process_noise_std

    def step(self, action: float) -> tuple[np.ndarray, float, bool, dict]:
        action = float(np.clip(action, -1.0, 1.0))
        noise = float(np.random.normal(0.0, self.process_noise_std))

        self.vel += (action + noise) * self.dt
        self.pos += self.vel * self.dt
        self.t += 1

        reward = -((self.pos - self.goal) ** 2 + 0.1 * action**2)
        done = self.t >= self.horizon
        return self._get_obs(), float(reward), done, {"process_noise": noise}
