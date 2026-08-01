from __future__ import annotations

import random
from collections.abc import Sequence

import numpy as np
import torch
from torch import nn, optim

from .models import StochasticLatentWorldModel
from .training import Transition


def train_stochastic_latent_model(
    model: StochasticLatentWorldModel,
    data: Sequence[Transition],
    epochs: int = 50,
    batch_size: int = 128,
    lr: float = 1e-3,
    kl_weight: float = 0.02,
    device: str = "cpu",
) -> dict[str, list[float]]:
    """Train a minimal Gaussian stochastic latent dynamics model."""

    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    mse = nn.MSELoss()
    shuffled_data = list(data)
    losses = {"total": [], "obs": [], "reward": [], "kl": []}

    for epoch in range(epochs):
        random.shuffle(shuffled_data)
        totals = {"total": 0.0, "obs": 0.0, "reward": 0.0, "kl": 0.0}

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

            output = model(obs, action, next_obs=next_obs, sample=True)
            obs_loss = mse(output["pred_next_obs"], next_obs)
            reward_loss = mse(output["pred_reward"], reward)
            kl_loss = gaussian_kl(
                output["posterior_next_mean"],
                output["posterior_next_log_std"],
                output["prior_mean"],
                output["prior_log_std"],
            ).mean()
            loss = obs_loss + reward_loss + kl_weight * kl_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_size_actual = len(batch)
            totals["total"] += loss.item() * batch_size_actual
            totals["obs"] += obs_loss.item() * batch_size_actual
            totals["reward"] += reward_loss.item() * batch_size_actual
            totals["kl"] += kl_loss.item() * batch_size_actual

        n = len(shuffled_data)
        for key in losses:
            losses[key].append(totals[key] / n)

        print(
            f"Epoch {epoch + 1:03d} | "
            f"total={losses['total'][-1]:.6f} | "
            f"obs={losses['obs'][-1]:.6f} | "
            f"reward={losses['reward'][-1]:.6f} | "
            f"kl={losses['kl'][-1]:.6f}"
        )

    return losses


def gaussian_kl(
    q_mean: torch.Tensor,
    q_log_std: torch.Tensor,
    p_mean: torch.Tensor,
    p_log_std: torch.Tensor,
) -> torch.Tensor:
    """KL(q || p) for diagonal Gaussian distributions."""

    q_var = torch.exp(2.0 * q_log_std)
    p_var = torch.exp(2.0 * p_log_std)
    return 0.5 * (
        2.0 * (p_log_std - q_log_std)
        + (q_var + (q_mean - p_mean) ** 2) / p_var
        - 1.0
    ).sum(dim=-1)


@torch.no_grad()
def sample_one_step_futures(
    model: StochasticLatentWorldModel,
    obs: np.ndarray,
    action: np.ndarray,
    num_samples: int = 80,
    device: str = "cpu",
) -> np.ndarray:
    """Sample many possible next observations for one state-action pair."""

    model.eval()
    obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
    action_tensor = torch.tensor(action, dtype=torch.float32, device=device).unsqueeze(0)
    samples: list[np.ndarray] = []

    for _ in range(num_samples):
        output = model(obs_tensor, action_tensor, sample=True)
        samples.append(output["pred_next_obs"].squeeze(0).cpu().numpy())

    return np.stack(samples)


@torch.no_grad()
def rollout_stochastic_latent_model(
    model: StochasticLatentWorldModel,
    initial_obs: np.ndarray,
    action_sequence: np.ndarray,
    num_rollouts: int = 40,
    device: str = "cpu",
) -> np.ndarray:
    """Sample multiple open-loop futures from stochastic latent dynamics."""

    model.eval()
    all_rollouts: list[np.ndarray] = []

    for _ in range(num_rollouts):
        obs = torch.tensor(initial_obs, dtype=torch.float32, device=device).unsqueeze(0)
        latent_mean, latent_log_std = model.encode(obs)
        latent = model.sample_latent(latent_mean, latent_log_std, sample=True)
        observations = [obs.squeeze(0).cpu().numpy()]

        for action_np in action_sequence:
            action = torch.tensor(action_np, dtype=torch.float32, device=device).unsqueeze(
                0
            )
            prior_mean, prior_log_std = model.transition_distribution(latent, action)
            latent = model.sample_latent(prior_mean, prior_log_std, sample=True)
            decoded_obs = model.decoder(latent)
            observations.append(decoded_obs.squeeze(0).cpu().numpy())

        all_rollouts.append(np.stack(observations))

    return np.stack(all_rollouts)


@torch.no_grad()
def sample_prior_latents(
    model: StochasticLatentWorldModel,
    obs: np.ndarray,
    action: np.ndarray,
    num_samples: int = 250,
    device: str = "cpu",
) -> np.ndarray:
    """Sample latent next states from p(z_{t+1} | z_t, a_t)."""

    model.eval()
    obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
    action_tensor = torch.tensor(action, dtype=torch.float32, device=device).unsqueeze(0)
    latent_mean, latent_log_std = model.encode(obs_tensor)
    latent = model.sample_latent(latent_mean, latent_log_std, sample=False)
    prior_mean, prior_log_std = model.transition_distribution(latent, action_tensor)
    samples = [
        model.sample_latent(prior_mean, prior_log_std, sample=True)
        .squeeze(0)
        .cpu()
        .numpy()
        for _ in range(num_samples)
    ]
    return np.stack(samples)


@torch.no_grad()
def analyze_prior_posterior_alignment(
    model: StochasticLatentWorldModel,
    data: Sequence[Transition],
    max_points: int = 500,
    device: str = "cpu",
) -> dict[str, np.ndarray]:
    """Compare prior and posterior distributions on real transitions."""

    model.eval()
    sample = list(data[:max_points])
    obs = torch.tensor(
        np.stack([transition[0] for transition in sample]),
        dtype=torch.float32,
        device=device,
    )
    action = torch.tensor(
        np.stack([transition[1] for transition in sample]),
        dtype=torch.float32,
        device=device,
    )
    next_obs = torch.tensor(
        np.stack([transition[2] for transition in sample]),
        dtype=torch.float32,
        device=device,
    )

    output = model(obs, action, next_obs=next_obs, sample=False)
    prior_mean = output["prior_mean"]
    prior_log_std = output["prior_log_std"]
    posterior_mean = output["posterior_next_mean"]
    posterior_log_std = output["posterior_next_log_std"]
    kl = gaussian_kl(
        posterior_mean,
        posterior_log_std,
        prior_mean,
        prior_log_std,
    )
    mean_distance = torch.linalg.norm(posterior_mean - prior_mean, dim=-1)
    prior_decoded_next_obs = model.decoder(prior_mean)
    posterior_decoded_next_obs = model.decoder(posterior_mean)

    return {
        "obs": obs.cpu().numpy(),
        "action": action.cpu().numpy(),
        "true_next_obs": next_obs.cpu().numpy(),
        "prior_mean": prior_mean.cpu().numpy(),
        "prior_std": torch.exp(prior_log_std).cpu().numpy(),
        "posterior_mean": posterior_mean.cpu().numpy(),
        "posterior_std": torch.exp(posterior_log_std).cpu().numpy(),
        "kl": kl.cpu().numpy(),
        "mean_distance": mean_distance.cpu().numpy(),
        "prior_decoded_next_obs": prior_decoded_next_obs.cpu().numpy(),
        "posterior_decoded_next_obs": posterior_decoded_next_obs.cpu().numpy(),
    }


@torch.no_grad()
def sample_prior_and_posterior_latents(
    model: StochasticLatentWorldModel,
    obs: np.ndarray,
    action: np.ndarray,
    next_obs: np.ndarray,
    num_samples: int = 300,
    device: str = "cpu",
) -> dict[str, np.ndarray]:
    """Sample prior and posterior latent distributions for one transition."""

    model.eval()
    obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
    action_tensor = torch.tensor(action, dtype=torch.float32, device=device).unsqueeze(0)
    next_obs_tensor = torch.tensor(next_obs, dtype=torch.float32, device=device).unsqueeze(
        0
    )

    latent_mean, latent_log_std = model.encode(obs_tensor)
    latent = model.sample_latent(latent_mean, latent_log_std, sample=False)
    prior_mean, prior_log_std = model.transition_distribution(latent, action_tensor)
    posterior_mean, posterior_log_std = model.encode(next_obs_tensor)

    prior_samples = [
        model.sample_latent(prior_mean, prior_log_std, sample=True)
        .squeeze(0)
        .cpu()
        .numpy()
        for _ in range(num_samples)
    ]
    posterior_samples = [
        model.sample_latent(posterior_mean, posterior_log_std, sample=True)
        .squeeze(0)
        .cpu()
        .numpy()
        for _ in range(num_samples)
    ]

    return {
        "prior_samples": np.stack(prior_samples),
        "posterior_samples": np.stack(posterior_samples),
        "prior_mean": prior_mean.squeeze(0).cpu().numpy(),
        "posterior_mean": posterior_mean.squeeze(0).cpu().numpy(),
        "prior_std": torch.exp(prior_log_std).squeeze(0).cpu().numpy(),
        "posterior_std": torch.exp(posterior_log_std).squeeze(0).cpu().numpy(),
    }
