from __future__ import annotations

import random
from collections.abc import Callable, Sequence

import numpy as np
import torch
from torch import nn, optim

from .envs import PointMass1DEnv
from .models import LatentDynamicsWorldModel
from .training import Transition

DreamerEpisode = dict[str, np.ndarray]


class DreamerActor(nn.Module):
    """A tiny deterministic actor that maps latent state to bounded action."""

    def __init__(
        self,
        latent_dim: int = 8,
        action_dim: int = 1,
        hidden_dim: int = 64,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh(),
        )

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        return self.net(latent)


class DreamerValue(nn.Module):
    """A tiny value model that estimates imagined return from latent state."""

    def __init__(self, latent_dim: int = 8, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        return self.net(latent)


class DreamerRSSMWorldModel(nn.Module):
    """A compact RSSM with deterministic memory and stochastic latent state."""

    def __init__(
        self,
        obs_dim: int = 2,
        action_dim: int = 1,
        deter_dim: int = 32,
        stoch_dim: int = 8,
        hidden_dim: int = 64,
        min_log_std: float = -5.0,
        max_log_std: float = 2.0,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.deter_dim = deter_dim
        self.stoch_dim = stoch_dim
        self.feature_dim = deter_dim + stoch_dim
        self.min_log_std = min_log_std
        self.max_log_std = max_log_std

        self.recurrent = nn.GRUCell(stoch_dim + action_dim, deter_dim)
        self.prior_net = nn.Sequential(
            nn.Linear(deter_dim, hidden_dim),
            nn.ReLU(),
        )
        self.prior_mean = nn.Linear(hidden_dim, stoch_dim)
        self.prior_log_std = nn.Linear(hidden_dim, stoch_dim)
        self.posterior_net = nn.Sequential(
            nn.Linear(deter_dim + obs_dim, hidden_dim),
            nn.ReLU(),
        )
        self.posterior_mean = nn.Linear(hidden_dim, stoch_dim)
        self.posterior_log_std = nn.Linear(hidden_dim, stoch_dim)
        self.decoder = nn.Sequential(
            nn.Linear(self.feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, obs_dim),
        )
        self.reward_head = nn.Sequential(
            nn.Linear(self.feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.continue_head = nn.Sequential(
            nn.Linear(self.feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def initial(
        self, batch_size: int, device: str | torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        deter = torch.zeros(batch_size, self.deter_dim, device=device)
        stoch = torch.zeros(batch_size, self.stoch_dim, device=device)
        return deter, stoch

    def feature(self, deter: torch.Tensor, stoch: torch.Tensor) -> torch.Tensor:
        return torch.cat([deter, stoch], dim=-1)

    def prior_distribution(
        self, deter: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.prior_net(deter)
        return self.prior_mean(hidden), self._clip_log_std(self.prior_log_std(hidden))

    def posterior_distribution(
        self, deter: torch.Tensor, obs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.posterior_net(torch.cat([deter, obs], dim=-1))
        return self.posterior_mean(hidden), self._clip_log_std(
            self.posterior_log_std(hidden)
        )

    def imagine_step(
        self,
        deter: torch.Tensor,
        stoch: torch.Tensor,
        action: torch.Tensor,
        sample: bool = True,
    ) -> dict[str, torch.Tensor]:
        next_deter = self.recurrent(torch.cat([stoch, action], dim=-1), deter)
        prior_mean, prior_log_std = self.prior_distribution(next_deter)
        next_stoch = self.sample_stoch(prior_mean, prior_log_std, sample=sample)
        return {
            "deter": next_deter,
            "stoch": next_stoch,
            "prior_mean": prior_mean,
            "prior_log_std": prior_log_std,
        }

    def observe_step(
        self,
        deter: torch.Tensor,
        stoch: torch.Tensor,
        action: torch.Tensor,
        obs: torch.Tensor,
        sample: bool = True,
    ) -> dict[str, torch.Tensor]:
        imagined = self.imagine_step(deter, stoch, action, sample=sample)
        posterior_mean, posterior_log_std = self.posterior_distribution(
            imagined["deter"], obs
        )
        posterior_stoch = self.sample_stoch(
            posterior_mean, posterior_log_std, sample=sample
        )
        imagined["stoch"] = posterior_stoch
        imagined["posterior_mean"] = posterior_mean
        imagined["posterior_log_std"] = posterior_log_std
        return imagined

    def observe(
        self,
        obs_seq: torch.Tensor,
        action_seq: torch.Tensor,
        sample: bool = True,
    ) -> dict[str, torch.Tensor]:
        batch_size, obs_steps, _ = obs_seq.shape
        device = obs_seq.device
        deter, stoch = self.initial(batch_size, device)
        zero_action = torch.zeros(batch_size, self.action_dim, device=device)

        deters: list[torch.Tensor] = []
        stochs: list[torch.Tensor] = []
        prior_means: list[torch.Tensor] = []
        prior_log_stds: list[torch.Tensor] = []
        posterior_means: list[torch.Tensor] = []
        posterior_log_stds: list[torch.Tensor] = []

        for t in range(obs_steps):
            previous_action = zero_action if t == 0 else action_seq[:, t - 1]
            step = self.observe_step(
                deter,
                stoch,
                previous_action,
                obs_seq[:, t],
                sample=sample,
            )
            deter = step["deter"]
            stoch = step["stoch"]
            deters.append(deter)
            stochs.append(stoch)
            prior_means.append(step["prior_mean"])
            prior_log_stds.append(step["prior_log_std"])
            posterior_means.append(step["posterior_mean"])
            posterior_log_stds.append(step["posterior_log_std"])

        deter_seq = torch.stack(deters, dim=1)
        stoch_seq = torch.stack(stochs, dim=1)
        features = self.feature(deter_seq, stoch_seq)
        return {
            "deter": deter_seq,
            "stoch": stoch_seq,
            "features": features,
            "prior_mean": torch.stack(prior_means, dim=1),
            "prior_log_std": torch.stack(prior_log_stds, dim=1),
            "posterior_mean": torch.stack(posterior_means, dim=1),
            "posterior_log_std": torch.stack(posterior_log_stds, dim=1),
            "obs_pred": self.decoder(features),
            "reward_pred": self.reward_head(features[:, 1:]),
            "continue_logits": self.continue_head(features[:, 1:]),
        }

    def encode_observation(
        self, obs: torch.Tensor, sample: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = obs.shape[0]
        deter, stoch = self.initial(batch_size, obs.device)
        zero_action = torch.zeros(batch_size, self.action_dim, device=obs.device)
        step = self.observe_step(deter, stoch, zero_action, obs, sample=sample)
        feature = self.feature(step["deter"], step["stoch"])
        return step["deter"], step["stoch"], feature

    def sample_stoch(
        self,
        mean: torch.Tensor,
        log_std: torch.Tensor,
        sample: bool = True,
    ) -> torch.Tensor:
        if not sample:
            return mean
        return mean + torch.randn_like(mean) * torch.exp(log_std)

    def _clip_log_std(self, log_std: torch.Tensor) -> torch.Tensor:
        return log_std.clamp(self.min_log_std, self.max_log_std)


class DreamerReplayBuffer:
    """Small episode replay buffer for sequence sampling."""

    def __init__(self, capacity: int = 100):
        self.capacity = capacity
        self.episodes: list[DreamerEpisode] = []

    def add(self, episode: DreamerEpisode) -> None:
        self.episodes.append(episode)
        if len(self.episodes) > self.capacity:
            self.episodes.pop(0)

    def sample_sequences(
        self,
        batch_size: int,
        sequence_length: int,
    ) -> dict[str, np.ndarray]:
        eligible = [
            episode
            for episode in self.episodes
            if len(episode["actions"]) >= sequence_length
        ]
        if not eligible:
            raise ValueError("Replay buffer does not contain enough long episodes.")

        obs_batch = []
        action_batch = []
        reward_batch = []
        continue_batch = []

        for _ in range(batch_size):
            episode = random.choice(eligible)
            max_start = len(episode["actions"]) - sequence_length
            start = random.randint(0, max_start)
            end = start + sequence_length
            obs_batch.append(episode["observations"][start : end + 1])
            action_batch.append(episode["actions"][start:end])
            reward_batch.append(episode["rewards"][start:end])
            continue_batch.append(episode["continues"][start:end])

        return {
            "observations": np.stack(obs_batch).astype(np.float32),
            "actions": np.stack(action_batch).astype(np.float32),
            "rewards": np.stack(reward_batch).astype(np.float32),
            "continues": np.stack(continue_batch).astype(np.float32),
        }

    def __len__(self) -> int:
        return len(self.episodes)


def train_dreamer_actor_critic(
    world_model: LatentDynamicsWorldModel,
    actor: DreamerActor,
    value: DreamerValue,
    data: Sequence[Transition],
    epochs: int = 250,
    batch_size: int = 128,
    imagination_horizon: int = 15,
    discount: float = 0.97,
    actor_lr: float = 1e-3,
    value_lr: float = 1e-3,
    device: str = "cpu",
) -> dict[str, list[float]]:
    """Train actor and value inside a frozen latent world model."""

    world_model.to(device)
    actor.to(device)
    value.to(device)
    world_model.eval()
    _set_requires_grad(world_model, requires_grad=False)

    actor_optimizer = optim.Adam(actor.parameters(), lr=actor_lr)
    value_optimizer = optim.Adam(value.parameters(), lr=value_lr)
    mse = nn.MSELoss()
    observations = np.stack([transition[0] for transition in data])

    losses = {
        "actor": [],
        "value": [],
        "imagined_return": [],
        "mean_abs_action": [],
    }

    for epoch in range(epochs):
        batch_obs = _sample_observations(observations, batch_size, device)

        actor.eval()
        value.train()
        with torch.no_grad():
            critic_rollout = imagine_latent_actor_rollout(
                world_model,
                actor,
                batch_obs,
                horizon=imagination_horizon,
                device=device,
            )
            critic_latents = critic_rollout["latents"].detach()
            critic_rewards = critic_rollout["rewards"].detach()
            bootstrap = value(critic_latents[:, -1]).detach()
            target_returns = discounted_returns(
                critic_rewards,
                bootstrap,
                discount=discount,
            )

        value_pred = value(
            critic_latents[:, :-1].reshape(-1, critic_latents.shape[-1])
        ).reshape(batch_size, imagination_horizon, 1)
        value_loss = mse(value_pred, target_returns)

        value_optimizer.zero_grad()
        value_loss.backward()
        value_optimizer.step()

        actor.train()
        value.eval()
        _set_requires_grad(value, requires_grad=False)
        actor_rollout = imagine_latent_actor_rollout(
            world_model,
            actor,
            batch_obs,
            horizon=imagination_horizon,
            device=device,
        )
        actor_rewards = actor_rollout["rewards"]
        actor_actions = actor_rollout["actions"]
        actor_bootstrap = value(actor_rollout["latents"][:, -1])
        actor_returns = discounted_returns(
            actor_rewards,
            actor_bootstrap,
            discount=discount,
        )
        actor_loss = -actor_returns.mean()

        actor_optimizer.zero_grad()
        actor_loss.backward()
        actor_optimizer.step()
        _set_requires_grad(value, requires_grad=True)

        losses["actor"].append(float(actor_loss.item()))
        losses["value"].append(float(value_loss.item()))
        losses["imagined_return"].append(float(actor_returns[:, 0].mean().item()))
        losses["mean_abs_action"].append(float(actor_actions.abs().mean().item()))

        if (epoch + 1) % 25 == 0 or epoch == 0:
            print(
                f"Actor-critic epoch {epoch + 1:03d} | "
                f"actor={losses['actor'][-1]:.4f} | "
                f"value={losses['value'][-1]:.4f} | "
                f"imagined_return={losses['imagined_return'][-1]:.4f}"
            )

    return losses


def imagine_latent_actor_rollout(
    world_model: LatentDynamicsWorldModel,
    actor: DreamerActor,
    initial_obs: torch.Tensor,
    horizon: int = 15,
    device: str = "cpu",
) -> dict[str, torch.Tensor]:
    """Roll the actor forward inside the latent world model."""

    obs = initial_obs.to(device)
    latent = world_model.encoder(obs)
    latents = [latent]
    decoded_observations = [world_model.decoder(latent)]
    actions: list[torch.Tensor] = []
    rewards: list[torch.Tensor] = []

    for _ in range(horizon):
        action = actor(latent)
        latent, reward = world_model.step_latent(latent, action)
        decoded_obs = world_model.decoder(latent)
        actions.append(action)
        rewards.append(reward)
        latents.append(latent)
        decoded_observations.append(decoded_obs)

    return {
        "latents": torch.stack(latents, dim=1),
        "decoded_observations": torch.stack(decoded_observations, dim=1),
        "actions": torch.stack(actions, dim=1),
        "rewards": torch.stack(rewards, dim=1),
    }


def discounted_returns(
    rewards: torch.Tensor,
    bootstrap: torch.Tensor,
    discount: float = 0.97,
) -> torch.Tensor:
    """Compute simple bootstrapped discounted returns for imagined rewards."""

    returns: list[torch.Tensor] = []
    running = bootstrap

    for t in reversed(range(rewards.shape[1])):
        running = rewards[:, t] + discount * running
        returns.append(running)

    returns.reverse()
    return torch.stack(returns, dim=1)


@torch.no_grad()
def imagine_actor_rollout_numpy(
    world_model: LatentDynamicsWorldModel,
    actor: DreamerActor,
    initial_obs: np.ndarray,
    horizon: int = 35,
    device: str = "cpu",
) -> dict[str, np.ndarray]:
    """Create one imagined actor rollout and return NumPy arrays for plotting."""

    world_model.eval()
    actor.eval()
    obs = torch.tensor(initial_obs, dtype=torch.float32, device=device).unsqueeze(0)
    rollout = imagine_latent_actor_rollout(
        world_model,
        actor,
        obs,
        horizon=horizon,
        device=device,
    )
    return {
        "decoded_observations": rollout["decoded_observations"]
        .squeeze(0)
        .cpu()
        .numpy(),
        "actions": rollout["actions"].squeeze(0).cpu().numpy(),
        "rewards": rollout["rewards"].squeeze(0).cpu().numpy(),
    }


@torch.no_grad()
def sample_actor_action_grid(
    world_model: LatentDynamicsWorldModel,
    actor: DreamerActor,
    position_range: tuple[float, float] = (-2.0, 2.0),
    velocity_range: tuple[float, float] = (-1.0, 1.0),
    grid_size: int = 25,
    device: str = "cpu",
) -> dict[str, np.ndarray]:
    """Evaluate the actor on a grid of point-mass states."""

    world_model.eval()
    actor.eval()
    positions = np.linspace(position_range[0], position_range[1], grid_size)
    velocities = np.linspace(velocity_range[0], velocity_range[1], grid_size)
    grid_position, grid_velocity = np.meshgrid(positions, velocities)
    observations = np.stack(
        [grid_position.reshape(-1), grid_velocity.reshape(-1)],
        axis=-1,
    ).astype(np.float32)
    obs_tensor = torch.tensor(observations, dtype=torch.float32, device=device)
    latent = world_model.encoder(obs_tensor)
    actions = actor(latent).cpu().numpy().reshape(grid_size, grid_size)

    return {
        "positions": grid_position,
        "velocities": grid_velocity,
        "actions": actions,
    }


@torch.no_grad()
def rollout_actor_in_env(
    env: PointMass1DEnv,
    world_model: LatentDynamicsWorldModel,
    actor: DreamerActor,
    device: str = "cpu",
) -> dict[str, np.ndarray]:
    """Run the learned actor in the real environment with posterior re-encoding."""

    world_model.eval()
    actor.eval()
    obs = env.reset()
    observations = [obs.copy()]
    actions: list[float] = []
    rewards: list[float] = []
    done = False

    while not done:
        obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        latent = world_model.encoder(obs_tensor)
        action = float(actor(latent).squeeze(0).cpu().numpy()[0])
        obs, reward, done, _ = env.step(action)
        observations.append(obs.copy())
        actions.append(action)
        rewards.append(reward)

    return {
        "observations": np.stack(observations),
        "actions": np.array(actions, dtype=np.float32).reshape(-1, 1),
        "rewards": np.array(rewards, dtype=np.float32).reshape(-1, 1),
    }


def rollout_random_policy(
    env: PointMass1DEnv,
) -> dict[str, np.ndarray]:
    """Run a random policy as a lightweight baseline."""

    obs = env.reset()
    observations = [obs.copy()]
    actions: list[float] = []
    rewards: list[float] = []
    done = False

    while not done:
        action = random.uniform(-1.0, 1.0)
        obs, reward, done, _ = env.step(action)
        observations.append(obs.copy())
        actions.append(action)
        rewards.append(reward)

    return {
        "observations": np.stack(observations),
        "actions": np.array(actions, dtype=np.float32).reshape(-1, 1),
        "rewards": np.array(rewards, dtype=np.float32).reshape(-1, 1),
    }


def train_dreamer_like_agent(
    env_factory: Callable[[], PointMass1DEnv],
    world_model: DreamerRSSMWorldModel,
    actor: DreamerActor,
    value: DreamerValue,
    iterations: int = 8,
    seed_episodes: int = 8,
    collect_episodes_per_iteration: int = 2,
    sequence_length: int = 24,
    batch_size: int = 32,
    world_updates: int = 25,
    actor_updates: int = 25,
    imagination_horizon: int = 15,
    discount: float = 0.97,
    lambda_: float = 0.95,
    kl_weight: float = 0.05,
    continue_weight: float = 0.5,
    exploration_std: float = 0.25,
    device: str ="cpu",
) -> tuple[DreamerReplayBuffer, dict[str, list[float]]]:
    """Train a compact Dreamer-like agent with RSSM, lambda-return, and replay."""

    world_model.to(device)
    actor.to(device)
    value.to(device)
    replay = DreamerReplayBuffer(capacity=80)

    for _ in range(seed_episodes):
        replay.add(collect_dreamer_episode(env_factory(), random_actions=True))

    world_optimizer = optim.Adam(world_model.parameters(), lr=1e-3)
    actor_optimizer = optim.Adam(actor.parameters(), lr=1e-4)
    value_optimizer = optim.Adam(value.parameters(), lr=1e-4)
    history = {
        "world_total": [],
        "world_obs": [],
        "world_reward": [],
        "world_continue": [],
        "world_kl": [],
        "actor": [],
        "value": [],
        "imagined_return": [],
        "real_return": [],
        "replay_episodes": [],
    }

    for iteration in range(iterations):
        world_metrics = []
        actor_metrics = []

        _set_requires_grad(world_model, requires_grad=True)
        for _ in range(world_updates):
            batch = replay.sample_sequences(batch_size, sequence_length)
            world_metrics.append(
                train_rssm_world_model_step(
                    world_model,
                    batch,
                    world_optimizer,
                    kl_weight=kl_weight,
                    continue_weight=continue_weight,
                    device=device,
                )
            )

        _set_requires_grad(world_model, requires_grad=False)
        for _ in range(actor_updates):
            batch = replay.sample_sequences(batch_size, sequence_length)
            actor_metrics.append(
                train_rssm_actor_critic_step(
                    world_model,
                    actor,
                    value,
                    batch,
                    actor_optimizer,
                    value_optimizer,
                    imagination_horizon=imagination_horizon,
                    discount=discount,
                    lambda_=lambda_,
                    device=device,
                )
            )

        current_exploration = exploration_std * (0.85**iteration)
        for _ in range(collect_episodes_per_iteration):
            replay.add(
                collect_dreamer_episode(
                    env_factory(),
                    world_model=world_model,
                    actor=actor,
                    exploration_std=current_exploration,
                    random_actions=False,
                    device=device,
                )
            )

        eval_episode = collect_dreamer_episode(
            env_factory(),
            world_model=world_model,
            actor=actor,
            exploration_std=0.0,
            random_actions=False,
            device=device,
        )
        world_summary = _mean_metrics(world_metrics)
        actor_summary = _mean_metrics(actor_metrics)
        history["world_total"].append(world_summary["total"])
        history["world_obs"].append(world_summary["obs"])
        history["world_reward"].append(world_summary["reward"])
        history["world_continue"].append(world_summary["continue"])
        history["world_kl"].append(world_summary["kl"])
        history["actor"].append(actor_summary["actor"])
        history["value"].append(actor_summary["value"])
        history["imagined_return"].append(actor_summary["imagined_return"])
        history["real_return"].append(float(eval_episode["rewards"].sum()))
        history["replay_episodes"].append(float(len(replay)))

        print(
            f"Dreamer-like iteration {iteration + 1:02d} | "
            f"world={history['world_total'][-1]:.4f} | "
            f"actor={history['actor'][-1]:.4f} | "
            f"value={history['value'][-1]:.4f} | "
            f"real_return={history['real_return'][-1]:.2f}"
        )

    _set_requires_grad(world_model, requires_grad=True)
    return replay, history


def train_rssm_world_model_step(
    world_model: DreamerRSSMWorldModel,
    batch: dict[str, np.ndarray],
    optimizer: optim.Optimizer,
    kl_weight: float = 0.05,
    continue_weight: float = 0.5,
    device: str = "cpu",
) -> dict[str, float]:
    """One supervised RSSM update on replay sequences."""

    world_model.train()
    obs = torch.tensor(batch["observations"], dtype=torch.float32, device=device)
    actions = torch.tensor(batch["actions"], dtype=torch.float32, device=device)
    rewards = torch.tensor(batch["rewards"], dtype=torch.float32, device=device)
    continues = torch.tensor(batch["continues"], dtype=torch.float32, device=device)
    output = world_model.observe(obs, actions, sample=True)

    obs_loss = nn.functional.mse_loss(output["obs_pred"], obs)
    reward_loss = nn.functional.mse_loss(output["reward_pred"], rewards)
    continue_loss = nn.functional.binary_cross_entropy_with_logits(
        output["continue_logits"], continues
    )
    kl_loss = _diagonal_gaussian_kl(
        output["posterior_mean"][:, 1:],
        output["posterior_log_std"][:, 1:],
        output["prior_mean"][:, 1:],
        output["prior_log_std"][:, 1:],
    ).mean()
    total_loss = (
        obs_loss
        + reward_loss
        + continue_weight * continue_loss
        + kl_weight * kl_loss
    )

    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    return {
        "total": float(total_loss.item()),
        "obs": float(obs_loss.item()),
        "reward": float(reward_loss.item()),
        "continue": float(continue_loss.item()),
        "kl": float(kl_loss.item()),
    }


def train_rssm_actor_critic_step(
    world_model: DreamerRSSMWorldModel,
    actor: DreamerActor,
    value: DreamerValue,
    batch: dict[str, np.ndarray],
    actor_optimizer: optim.Optimizer,
    value_optimizer: optim.Optimizer,
    imagination_horizon: int = 15,
    discount: float = 0.97,
    lambda_: float = 0.95,
    device: str = "cpu",
) -> dict[str, float]:
    """One actor/value update using imagined RSSM rollouts and lambda-return."""

    world_model.eval()
    actor.to(device)
    value.to(device)
    obs = torch.tensor(batch["observations"], dtype=torch.float32, device=device)
    actions = torch.tensor(batch["actions"], dtype=torch.float32, device=device)

    with torch.no_grad():
        posterior = world_model.observe(obs, actions, sample=True)
        start_deter = posterior["deter"][:, -1].detach()
        start_stoch = posterior["stoch"][:, -1].detach()
        critic_rollout = imagine_rssm_actor_rollout(
            world_model,
            actor,
            start_deter,
            start_stoch,
            horizon=imagination_horizon,
            sample=False,
        )
        critic_features = critic_rollout["features"].detach()
        values_next = value(critic_features[:, 1:].reshape(-1, world_model.feature_dim))
        values_next = values_next.reshape(obs.shape[0], imagination_horizon, 1)
        value_targets = lambda_returns(
            critic_rollout["rewards"].detach(),
            values_next.detach(),
            critic_rollout["continues"].detach(),
            discount=discount,
            lambda_=lambda_,
        ).detach()

    value.train()
    value_pred = value(critic_features[:, :-1].reshape(-1, world_model.feature_dim))
    value_pred = value_pred.reshape(obs.shape[0], imagination_horizon, 1)
    value_loss = nn.functional.mse_loss(value_pred, value_targets)
    value_optimizer.zero_grad()
    value_loss.backward()
    value_optimizer.step()

    actor.train()
    value.eval()
    _set_requires_grad(value, requires_grad=False)
    actor_rollout = imagine_rssm_actor_rollout(
        world_model,
        actor,
        start_deter,
        start_stoch,
        horizon=imagination_horizon,
        sample=False,
    )
    actor_values_next = value(
        actor_rollout["features"][:, 1:].reshape(-1, world_model.feature_dim)
    ).reshape(obs.shape[0], imagination_horizon, 1)
    actor_returns = lambda_returns(
        actor_rollout["rewards"],
        actor_values_next,
        actor_rollout["continues"],
        discount=discount,
        lambda_=lambda_,
    )
    actor_loss = -actor_returns.mean()
    actor_optimizer.zero_grad()
    actor_loss.backward()
    actor_optimizer.step()
    _set_requires_grad(value, requires_grad=True)

    return {
        "actor": float(actor_loss.item()),
        "value": float(value_loss.item()),
        "imagined_return": float(actor_returns[:, 0].mean().item()),
    }


def imagine_rssm_actor_rollout(
    world_model: DreamerRSSMWorldModel,
    actor: DreamerActor,
    start_deter: torch.Tensor,
    start_stoch: torch.Tensor,
    horizon: int = 15,
    sample: bool = False,
) -> dict[str, torch.Tensor]:
    """Imagine an actor-controlled rollout from a posterior RSSM state."""

    deter = start_deter
    stoch = start_stoch
    features = [world_model.feature(deter, stoch)]
    decoded_observations = [world_model.decoder(features[-1])]
    actions: list[torch.Tensor] = []
    rewards: list[torch.Tensor] = []
    continues: list[torch.Tensor] = []

    for _ in range(horizon):
        feature = world_model.feature(deter, stoch)
        action = actor(feature)
        imagined = world_model.imagine_step(deter, stoch, action, sample=sample)
        deter = imagined["deter"]
        stoch = imagined["stoch"]
        next_feature = world_model.feature(deter, stoch)
        actions.append(action)
        rewards.append(world_model.reward_head(next_feature))
        continues.append(torch.sigmoid(world_model.continue_head(next_feature)))
        features.append(next_feature)
        decoded_observations.append(world_model.decoder(next_feature))

    return {
        "features": torch.stack(features, dim=1),
        "decoded_observations": torch.stack(decoded_observations, dim=1),
        "actions": torch.stack(actions, dim=1),
        "rewards": torch.stack(rewards, dim=1),
        "continues": torch.stack(continues, dim=1),
    }


def lambda_returns(
    rewards: torch.Tensor,
    values_next: torch.Tensor,
    continues: torch.Tensor,
    discount: float = 0.97,
    lambda_: float = 0.95,
) -> torch.Tensor:
    """Compute Dreamer-style lambda returns over imagined rewards."""

    next_return = values_next[:, -1]
    returns: list[torch.Tensor] = []

    for t in reversed(range(rewards.shape[1])):
        next_value = values_next[:, t]
        next_return = rewards[:, t] + discount * continues[:, t] * (
            (1.0 - lambda_) * next_value + lambda_ * next_return
        )
        returns.append(next_return)

    returns.reverse()
    return torch.stack(returns, dim=1)


def collect_dreamer_episode(
    env: PointMass1DEnv,
    world_model: DreamerRSSMWorldModel | None = None,
    actor: DreamerActor | None = None,
    exploration_std: float = 0.25,
    random_actions: bool = False,
    device: str = "cpu",
) -> DreamerEpisode:
    """Collect one episode for the Dreamer replay buffer."""

    obs = env.reset()
    observations = [obs.copy()]
    actions: list[float] = []
    rewards: list[float] = []
    continues: list[float] = []
    done = False

    if world_model is not None and actor is not None and not random_actions:
        world_model.eval()
        actor.eval()
        obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            deter, stoch, feature = world_model.encode_observation(
                obs_tensor, sample=False
            )
    else:
        deter = stoch = feature = None

    while not done:
        if random_actions or world_model is None or actor is None or feature is None:
            action = random.uniform(-1.0, 1.0)
        else:
            with torch.no_grad():
                action = float(actor(feature).squeeze(0).cpu().numpy()[0])
            if exploration_std > 0.0:
                action += float(np.random.normal(0.0, exploration_std))
            action = float(np.clip(action, -1.0, 1.0))

        next_obs, reward, done, _ = env.step(action)
        observations.append(next_obs.copy())
        actions.append(action)
        rewards.append(reward)
        continues.append(0.0 if done else 1.0)

        if world_model is not None and actor is not None and not random_actions:
            with torch.no_grad():
                action_tensor = torch.tensor(
                    [[action]], dtype=torch.float32, device=device
                )
                next_obs_tensor = torch.tensor(
                    next_obs, dtype=torch.float32, device=device
                ).unsqueeze(0)
                step = world_model.observe_step(
                    deter,
                    stoch,
                    action_tensor,
                    next_obs_tensor,
                    sample=False,
                )
                deter = step["deter"]
                stoch = step["stoch"]
                feature = world_model.feature(deter, stoch)

        obs = next_obs

    return {
        "observations": np.stack(observations).astype(np.float32),
        "actions": np.array(actions, dtype=np.float32).reshape(-1, 1),
        "rewards": np.array(rewards, dtype=np.float32).reshape(-1, 1),
        "continues": np.array(continues, dtype=np.float32).reshape(-1, 1),
    }


@torch.no_grad()
def evaluate_rssm_reconstruction(
    world_model: DreamerRSSMWorldModel,
    replay: DreamerReplayBuffer,
    sequence_length: int = 40,
    device: str = "cpu",
) -> dict[str, np.ndarray]:
    """Decode one replay sequence through the posterior RSSM for diagnostics."""

    world_model.eval()
    batch = replay.sample_sequences(batch_size=1, sequence_length=sequence_length)
    obs = torch.tensor(batch["observations"], dtype=torch.float32, device=device)
    actions = torch.tensor(batch["actions"], dtype=torch.float32, device=device)
    output = world_model.observe(obs, actions, sample=False)
    return {
        "observations": batch["observations"][0],
        "obs_pred": output["obs_pred"].squeeze(0).cpu().numpy(),
        "rewards": batch["rewards"][0],
        "reward_pred": output["reward_pred"].squeeze(0).cpu().numpy(),
        "continues": batch["continues"][0],
        "continue_prob": torch.sigmoid(output["continue_logits"])
        .squeeze(0)
        .cpu()
        .numpy(),
    }


@torch.no_grad()
def imagine_rssm_actor_rollout_numpy(
    world_model: DreamerRSSMWorldModel,
    actor: DreamerActor,
    initial_obs: np.ndarray,
    horizon: int = 35,
    device: str = "cpu",
) -> dict[str, np.ndarray]:
    """Create one RSSM imagined actor rollout as NumPy arrays."""

    world_model.eval()
    actor.eval()
    obs = torch.tensor(initial_obs, dtype=torch.float32, device=device).unsqueeze(0)
    deter, stoch, _ = world_model.encode_observation(obs, sample=False)
    rollout = imagine_rssm_actor_rollout(
        world_model,
        actor,
        deter,
        stoch,
        horizon=horizon,
        sample=False,
    )
    return {
        "decoded_observations": rollout["decoded_observations"]
        .squeeze(0)
        .cpu()
        .numpy(),
        "actions": rollout["actions"].squeeze(0).cpu().numpy(),
        "rewards": rollout["rewards"].squeeze(0).cpu().numpy(),
        "continues": rollout["continues"].squeeze(0).cpu().numpy(),
    }


@torch.no_grad()
def sample_rssm_actor_action_grid(
    world_model: DreamerRSSMWorldModel,
    actor: DreamerActor,
    position_range: tuple[float, float] = (-2.0, 2.0),
    velocity_range: tuple[float, float] = (-1.0, 1.0),
    grid_size: int = 25,
    device: str = "cpu",
) -> dict[str, np.ndarray]:
    """Evaluate an RSSM actor on a grid of fully observed point-mass states."""

    world_model.eval()
    actor.eval()
    positions = np.linspace(position_range[0], position_range[1], grid_size)
    velocities = np.linspace(velocity_range[0], velocity_range[1], grid_size)
    grid_position, grid_velocity = np.meshgrid(positions, velocities)
    observations = np.stack(
        [grid_position.reshape(-1), grid_velocity.reshape(-1)],
        axis=-1,
    ).astype(np.float32)
    obs_tensor = torch.tensor(observations, dtype=torch.float32, device=device)
    _, _, features = world_model.encode_observation(obs_tensor, sample=False)
    actions = actor(features).cpu().numpy().reshape(grid_size, grid_size)
    return {
        "positions": grid_position,
        "velocities": grid_velocity,
        "actions": actions,
    }


@torch.no_grad()
def rollout_rssm_actor_in_env(
    env: PointMass1DEnv,
    world_model: DreamerRSSMWorldModel,
    actor: DreamerActor,
    device: str = "cpu",
) -> dict[str, np.ndarray]:
    """Run an RSSM actor in the real environment."""

    return collect_dreamer_episode(
        env,
        world_model=world_model,
        actor=actor,
        exploration_std=0.0,
        random_actions=False,
        device=device,
    )


def _sample_observations(
    observations: np.ndarray,
    batch_size: int,
    device: str,
) -> torch.Tensor:
    indices = np.random.randint(0, len(observations), size=batch_size)
    return torch.tensor(
        observations[indices],
        dtype=torch.float32,
        device=device,
    )


def _diagonal_gaussian_kl(
    q_mean: torch.Tensor,
    q_log_std: torch.Tensor,
    p_mean: torch.Tensor,
    p_log_std: torch.Tensor,
) -> torch.Tensor:
    q_var = torch.exp(2.0 * q_log_std)
    p_var = torch.exp(2.0 * p_log_std)
    return 0.5 * (
        2.0 * (p_log_std - q_log_std)
        + (q_var + (q_mean - p_mean) ** 2) / p_var
        - 1.0
    ).sum(dim=-1)


def _mean_metrics(metrics: list[dict[str, float]]) -> dict[str, float]:
    if not metrics:
        return {}
    return {
        key: float(np.mean([metric[key] for metric in metrics]))
        for key in metrics[0]
    }


def _set_requires_grad(module: nn.Module, requires_grad: bool) -> None:
    for parameter in module.parameters():
        parameter.requires_grad_(requires_grad)
