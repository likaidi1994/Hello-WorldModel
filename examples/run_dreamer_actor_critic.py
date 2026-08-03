
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from hello_world_model import (  # noqa: E402
    DreamerActor,
    DreamerRSSMWorldModel,
    DreamerValue,
    PointMass1DEnv,
    evaluate_rssm_reconstruction,
    imagine_rssm_actor_rollout_numpy,
    plot_actor_policy_field,
    plot_dreamer_actor_episode,
    plot_dreamer_like_imagined_rollout,
    plot_dreamer_like_training_curves,
    plot_rssm_reconstruction,
    rollout_random_policy,
    rollout_rssm_actor_in_env,
    sample_rssm_actor_action_grid,
    train_dreamer_like_agent,
)


def make_env() -> PointMass1DEnv:
    return PointMass1DEnv(horizon=80)


def main() -> None:
    np.random.seed(10)
    torch.manual_seed(10)

    output_dir = PROJECT_ROOT / "outputs" / "dreamer_actor_critic"

    print("1. Building a compact RSSM world model...")
    world_model = DreamerRSSMWorldModel(
        obs_dim=2,
        action_dim=1,
        deter_dim=32,
        stoch_dim=8,
        hidden_dim=64,
    )
    actor = DreamerActor(
        latent_dim=world_model.feature_dim,
        action_dim=1,
        hidden_dim=64,
    )
    value = DreamerValue(latent_dim=world_model.feature_dim, hidden_dim=64)

    print("2. Training Dreamer-like agent with replay, RSSM, and lambda-return...")
    replay, history = train_dreamer_like_agent(
        make_env,
        world_model,
        actor,
        value,
        iterations=200,
        seed_episodes=200,
        collect_episodes_per_iteration=10,
        sequence_length=24,
        batch_size=32,
        world_updates=25,
        actor_updates=25,
        imagination_horizon=15,
        discount=0.97,
        lambda_=0.95,
        kl_weight=0.5,
        continue_weight=0.5,
        exploration_std=0.25,
        device="cpu",
    )
    plot_dreamer_like_training_curves(
        history,
        output_dir / "01_dreamer_like_training_curves.png",
    )

    print("3. Inspecting RSSM reconstruction, reward, and continuation heads...")
    reconstruction = evaluate_rssm_reconstruction(
        world_model,
        replay,
        sequence_length=40,
    )
    plot_rssm_reconstruction(
        reconstruction,
        output_dir / "02_rssm_reconstruction.png",
    )

    print("4. Visualizing an imagined actor rollout...")
    initial_obs = replay.episodes[-1]["observations"][0]
    imagined_rollout = imagine_rssm_actor_rollout_numpy(
        world_model,
        actor,
        initial_obs,
        horizon=35,
    )
    plot_dreamer_like_imagined_rollout(
        imagined_rollout,
        output_dir / "03_rssm_imagined_actor_rollout.png",
    )

    print("5. Visualizing the learned actor field...")
    policy_grid = sample_rssm_actor_action_grid(world_model, actor, grid_size=31)
    plot_actor_policy_field(policy_grid, output_dir / "04_actor_policy_field.png")

    print("6. Running the learned actor in the real environment...")
    actor_log = rollout_rssm_actor_in_env(make_env(), world_model, actor)
    random_log = rollout_random_policy(make_env())
    plot_dreamer_actor_episode(
        actor_log,
        random_log,
        output_dir / "05_actor_vs_random_episode.png",
    )

    actor_return = float(actor_log["rewards"].sum())
    random_return = float(random_log["rewards"].sum())
    print(f"Actor return: {actor_return:.2f}")
    print(f"Random return: {random_return:.2f}")
    print(f"Saved visualizations to: {output_dir}")


if __name__ == "__main__":
    main()
