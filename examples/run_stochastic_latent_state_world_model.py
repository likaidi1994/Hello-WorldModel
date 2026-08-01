from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from hello_world_model import (  # noqa: E402
    NoisyPointMass1DEnv,
    StochasticLatentWorldModel,
    collect_random_data,
    collect_real_rollout,
    make_probe_action_sequence,
    plot_one_step_future_distribution,
    plot_prior_latent_samples,
    plot_stochastic_model_losses,
    plot_stochastic_rollout_fan,
    plot_stochastic_uncertainty_growth,
    rollout_stochastic_latent_model,
    sample_one_step_futures,
    sample_prior_latents,
    train_stochastic_latent_model,
)


def main() -> None:
    output_dir = PROJECT_ROOT / "outputs" / "stochastic_latent_state_world_model"

    print("1. Collecting noisy point-mass data...")
    data_env = NoisyPointMass1DEnv(horizon=60, process_noise_std=0.18)
    data = collect_random_data(data_env, episodes=320)

    print("2. Training stochastic latent world model...")
    model = StochasticLatentWorldModel(
        obs_dim=2, action_dim=1, latent_dim=8, hidden_dim=64
    )
    losses = train_stochastic_latent_model(
        model, data, epochs=50, batch_size=128, lr=1e-3, kl_weight=0.02
    )
    plot_stochastic_model_losses(losses, output_dir / "01_stochastic_model_losses.png")

    print("3. Sampling one-step futures from the learned latent distribution...")
    obs, action, true_next_obs, _ = data[0]
    next_samples = sample_one_step_futures(model, obs, action, num_samples=120)
    plot_one_step_future_distribution(
        obs,
        action,
        true_next_obs,
        next_samples,
        output_dir / "02_one_step_future_distribution.png",
    )

    print("4. Sampling open-loop stochastic futures...")
    action_sequence = make_probe_action_sequence(horizon=40)
    rollout_env = NoisyPointMass1DEnv(horizon=60, process_noise_std=0.18)
    real_obs, _ = collect_real_rollout(rollout_env, action_sequence)
    sampled_rollouts = rollout_stochastic_latent_model(
        model, real_obs[0], action_sequence, num_rollouts=60
    )
    plot_stochastic_rollout_fan(
        real_obs, sampled_rollouts, output_dir / "03_stochastic_rollout_fan.png"
    )
    plot_stochastic_uncertainty_growth(
        sampled_rollouts, output_dir / "04_uncertainty_growth.png"
    )

    print("5. Visualizing samples from the next-latent prior...")
    latent_samples = sample_prior_latents(model, obs, action, num_samples=300)
    plot_prior_latent_samples(
        latent_samples, output_dir / "05_prior_latent_samples.png"
    )

    print(f"Saved visualizations to: {output_dir}")


if __name__ == "__main__":
    main()
