from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from hello_world_model import (  # noqa: E402
    LatentDynamicsWorldModel,
    PointMass1DEnv,
    WorldModel,
    collect_random_data,
    collect_real_rollout,
    make_probe_action_sequence,
    plot_latent_model_losses,
    plot_latent_space_rollout,
    plot_rollout_error_growth,
    plot_rollout_prediction_comparison,
    plot_training_loss,
    rollout_latent_dynamics_model,
    rollout_observation_model,
    train_latent_dynamics_model,
    train_world_model,
)


def main() -> None:
    output_dir = PROJECT_ROOT / "outputs" / "latent_dynamics_world_model"

    print("1. Collecting full-state point-mass data...")
    data_env = PointMass1DEnv(horizon=60)
    data = collect_random_data(data_env, episodes=260)

    print("2. Training observation-space dynamics baseline...")
    observation_model = WorldModel(state_dim=2, action_dim=1, hidden_dim=128)
    observation_losses = train_world_model(
        observation_model, data, epochs=45, batch_size=128, lr=1e-3
    )
    plot_training_loss(
        observation_losses, output_dir / "01_observation_model_loss.png"
    )

    print("3. Training latent dynamics model...")
    latent_model = LatentDynamicsWorldModel(
        obs_dim=2, action_dim=1, latent_dim=8, hidden_dim=64
    )
    latent_losses = train_latent_dynamics_model(
        latent_model, data, epochs=45, batch_size=128, lr=1e-3
    )
    plot_latent_model_losses(latent_losses, output_dir / "02_latent_model_losses.png")

    print("4. Comparing open-loop rollout behavior...")
    action_sequence = make_probe_action_sequence(horizon=40)
    rollout_env = PointMass1DEnv(horizon=60)
    real_obs, _ = collect_real_rollout(rollout_env, action_sequence)
    initial_obs = real_obs[0]
    obs_rollout, _ = rollout_observation_model(
        observation_model, initial_obs, action_sequence
    )
    latent_rollout, _, latents = rollout_latent_dynamics_model(
        latent_model, initial_obs, action_sequence
    )

    plot_rollout_prediction_comparison(
        real_obs,
        obs_rollout,
        latent_rollout,
        output_dir / "03_open_loop_rollout_comparison.png",
    )
    plot_rollout_error_growth(
        real_obs,
        obs_rollout,
        latent_rollout,
        output_dir / "04_rollout_error_growth.png",
    )
    plot_latent_space_rollout(latents, output_dir / "05_latent_space_rollout.png")

    print(f"Saved visualizations to: {output_dir}")


if __name__ == "__main__":
    main()
