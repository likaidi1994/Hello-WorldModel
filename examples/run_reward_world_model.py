from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from hello_world_model import (  # noqa: E402
    GRURewardWorldModel,
    PartialPointMass1DEnv,
    collect_partial_episodes,
    plot_action_reward_scores,
    plot_reward_landscape,
    plot_reward_model_losses,
    plot_reward_predictions,
    predict_episode_with_reward,
    score_candidate_actions,
    train_gru_reward_world_model,
)


def main() -> None:
    output_dir = PROJECT_ROOT / "outputs" / "reward_world_model"

    env = PartialPointMass1DEnv()
    print("1. Showing the true reward landscape...")
    plot_reward_landscape(output_dir / "01_reward_landscape.png", goal=env.goal)

    print("2. Collecting partial-observation episodes with reward...")
    episodes = collect_partial_episodes(env, episodes=300)

    print("3. Training GRU reward world model...")
    model = GRURewardWorldModel(obs_dim=1, action_dim=1, hidden_dim=32)
    losses = train_gru_reward_world_model(model, episodes, epochs=60, lr=1e-3)
    plot_reward_model_losses(losses, output_dir / "02_reward_model_losses.png")

    print("4. Comparing true rewards with predicted rewards...")
    _, _, true_reward, pred_reward = predict_episode_with_reward(model, episodes[0])
    plot_reward_predictions(
        true_reward, pred_reward, output_dir / "03_reward_prediction.png"
    )

    print("5. Scoring candidate actions with the learned reward head...")
    action_values = np.linspace(-1.0, 1.0, 101, dtype=np.float32)
    scores = score_candidate_actions(
        model, episodes[0], step_index=20, action_values=action_values
    )
    plot_action_reward_scores(
        action_values, scores, output_dir / "04_reward_guided_action_scores.png"
    )

    print(f"Saved visualizations to: {output_dir}")


if __name__ == "__main__":
    main()
