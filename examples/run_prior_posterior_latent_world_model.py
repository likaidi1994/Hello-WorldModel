from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from hello_world_model import (  # noqa: E402
    NoisyPointMass1DEnv,
    StochasticLatentWorldModel,
    analyze_prior_posterior_alignment,
    collect_random_data,
    plot_prior_decoded_next_state,
    plot_prior_posterior_alignment,
    plot_prior_posterior_latent_samples,
    plot_prior_vs_posterior_latent_means,
    plot_stochastic_model_losses,
    sample_prior_and_posterior_latents,
    train_stochastic_latent_model,
)


def main() -> None:
    np.random.seed(7)
    torch.manual_seed(7)

    output_dir = PROJECT_ROOT / "outputs" / "prior_posterior_latent_world_model"

    print("1. Collecting noisy transitions...")
    data_env = NoisyPointMass1DEnv(horizon=60, process_noise_std=0.18)
    data = collect_random_data(data_env, episodes=320)

    print("2. Training stochastic latent world model...")
    model = StochasticLatentWorldModel(
        obs_dim=2,
        action_dim=1,
        latent_dim=8,
        hidden_dim=64,
    )
    losses = train_stochastic_latent_model(
        model,
        data,
        epochs=50,
        batch_size=128,
        lr=1e-3,
        kl_weight=0.02,
    )
    plot_stochastic_model_losses(
        losses,
        output_dir / "01_prior_posterior_training_losses.png",
    )

    print("3. Measuring prior/posterior alignment on real transitions...")
    analysis = analyze_prior_posterior_alignment(model, data, max_points=800)
    plot_prior_vs_posterior_latent_means(
        analysis,
        output_dir / "02_prior_vs_posterior_latent_means.png",
    )
    plot_prior_posterior_alignment(
        analysis,
        output_dir / "04_prior_posterior_alignment.png",
    )
    plot_prior_decoded_next_state(
        analysis,
        output_dir / "05_prior_decoded_next_state.png",
    )

    print("4. Sampling prior and posterior latent clouds for one transition...")
    obs, action, next_obs, _ = data[0]
    latent_samples = sample_prior_and_posterior_latents(
        model,
        obs,
        action,
        next_obs,
        num_samples=350,
    )
    plot_prior_posterior_latent_samples(
        latent_samples,
        output_dir / "03_prior_posterior_latent_samples.png",
    )

    print(f"Saved visualizations to: {output_dir}")


if __name__ == "__main__":
    main()
