from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from hello_world_model import (  # noqa: E402
    GRURewardWorldModel,
    PartialPointMass1DEnv,
    collect_partial_episodes,
    plot_first_planning_rollouts,
    plot_imagined_vs_real_feedback,
    plot_receding_horizon_control,
    plot_reward_model_losses,
    run_receding_horizon_episode,
    train_gru_reward_world_model,
)


def main() -> None:
    output_dir = PROJECT_ROOT / "outputs" / "planning_world_model"

    print("1. Collecting data for the recurrent reward world model...")
    training_env = PartialPointMass1DEnv(horizon=60)
    episodes = collect_partial_episodes(training_env, episodes=220)

    print("2. Training the model used for imagination and reward scoring...")
    model = GRURewardWorldModel(obs_dim=1, action_dim=1, hidden_dim=32)
    losses = train_gru_reward_world_model(model, episodes, epochs=50, lr=1e-3)
    plot_reward_model_losses(losses, output_dir / "01_planning_model_losses.png")

    print("3. Running receding-horizon planning in the real environment...")
    control_env = PartialPointMass1DEnv(horizon=60)
    planning_log = run_receding_horizon_episode(
        control_env,
        model,
        horizon=12,
        num_samples=160,
    )

    print("4. Saving planning visualizations...")
    plot_first_planning_rollouts(
        planning_log["first_plan_rollouts"],
        planning_log["first_best_plan"],
        output_dir / "02_first_imagined_rollouts.png",
    )
    plot_receding_horizon_control(
        planning_log,
        output_dir / "03_receding_horizon_control.png",
    )
    plot_imagined_vs_real_feedback(
        planning_log,
        output_dir / "04_imagined_vs_real_feedback.png",
    )

    print(f"Saved visualizations to: {output_dir}")


if __name__ == "__main__":
    main()
