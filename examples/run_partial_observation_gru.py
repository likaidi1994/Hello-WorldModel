from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from hello_world_model import (  # noqa: E402
    GRUWorldModel,
    PartialPointMass1DEnv,
    collect_partial_episodes,
    plot_gru_sequence_prediction,
    plot_hidden_state_velocity_probe,
    plot_partial_observation_ambiguity,
    plot_training_loss,
    predict_episode,
    train_gru_world_model,
)


def main() -> None:
    output_dir = PROJECT_ROOT / "outputs" / "partial_observation_gru"

    env = PartialPointMass1DEnv()
    print("1. Collecting partial-observation episodes...")
    episodes = collect_partial_episodes(env, episodes=300)
    plot_partial_observation_ambiguity(
        episodes, output_dir / "01_partial_observation_ambiguity.png"
    )

    print("2. Training GRU world model...")
    model = GRUWorldModel(obs_dim=1, action_dim=1, hidden_dim=32)
    losses = train_gru_world_model(model, episodes, epochs=60, lr=1e-3)
    plot_training_loss(losses, output_dir / "02_gru_training_loss.png")

    print("3. Plotting sequence predictions and hidden-state probe...")
    true_next_obs, pred_next_obs, hidden_states = predict_episode(model, episodes[0])
    plot_gru_sequence_prediction(
        true_next_obs, pred_next_obs, output_dir / "03_gru_sequence_prediction.png"
    )
    plot_hidden_state_velocity_probe(
        hidden_states, episodes[0], output_dir / "04_hidden_state_probe.png"
    )

    print(f"Saved visualizations to: {output_dir}")


if __name__ == "__main__":
    main()
