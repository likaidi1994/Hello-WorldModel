from __future__ import annotations

import torch
from torch import nn


class WorldModel(nn.Module):
    """Predict next state and reward from current state-action pair."""

    def __init__(self, state_dim: int = 2, action_dim: int = 1, hidden_dim: int = 128):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.next_state_head = nn.Linear(hidden_dim, state_dim)
        self.reward_head = nn.Linear(hidden_dim, 1)

    def forward(
        self, state: torch.Tensor, action: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat([state, action], dim=-1)
        h = self.backbone(x)
        return self.next_state_head(h), self.reward_head(h)


class GRUWorldModel(nn.Module):
    """Predict next observation from a sequence of observations and actions."""

    def __init__(self, obs_dim: int = 1, action_dim: int = 1, hidden_dim: int = 32):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.gru = nn.GRU(
            input_size=obs_dim + action_dim,
            hidden_size=hidden_dim,
            batch_first=True,
        )
        self.next_obs_head = nn.Linear(hidden_dim, obs_dim)

    def forward(
        self,
        obs_seq: torch.Tensor,
        action_seq: torch.Tensor,
        h0: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat([obs_seq, action_seq], dim=-1)
        hidden_seq, hn = self.gru(x, h0)
        pred_next_obs = self.next_obs_head(hidden_seq)
        return pred_next_obs, hn


class GRURewardWorldModel(nn.Module):
    """Predict next observation and reward from recurrent world-model state."""

    def __init__(self, obs_dim: int = 1, action_dim: int = 1, hidden_dim: int = 32):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.gru = nn.GRU(
            input_size=obs_dim + action_dim,
            hidden_size=hidden_dim,
            batch_first=True,
        )
        self.next_obs_head = nn.Linear(hidden_dim, obs_dim)
        self.reward_head = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        obs_seq: torch.Tensor,
        action_seq: torch.Tensor,
        h0: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = torch.cat([obs_seq, action_seq], dim=-1)
        hidden_seq, hn = self.gru(x, h0)
        pred_next_obs = self.next_obs_head(hidden_seq)
        pred_reward = self.reward_head(hidden_seq)
        return pred_next_obs, pred_reward, hn

    def step(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        h: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Roll the recurrent world model forward by one imagined step."""

        x = torch.cat([obs, action], dim=-1).unsqueeze(1)
        hidden_seq, hn = self.gru(x, h)
        hidden = hidden_seq[:, 0, :]
        pred_next_obs = self.next_obs_head(hidden)
        pred_reward = self.reward_head(hidden)
        return pred_next_obs, pred_reward, hn


class LatentDynamicsWorldModel(nn.Module):
    """Roll future states in a learned latent space instead of observation space."""

    def __init__(
        self,
        obs_dim: int = 2,
        action_dim: int = 1,
        latent_dim: int = 8,
        hidden_dim: int = 64,
    ):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.transition = nn.Sequential(
            nn.Linear(latent_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, obs_dim),
        )
        self.reward_head = nn.Sequential(
            nn.Linear(latent_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self, obs: torch.Tensor, action: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        latent = self.encoder(obs)
        next_latent, pred_reward = self.step_latent(latent, action)
        pred_next_obs = self.decoder(next_latent)
        return pred_next_obs, pred_reward, next_latent

    def step_latent(
        self, latent: torch.Tensor, action: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        transition_input = torch.cat([latent, action], dim=-1)
        latent_delta = self.transition(transition_input)
        next_latent = latent + latent_delta
        pred_reward = self.reward_head(transition_input)
        return next_latent, pred_reward
