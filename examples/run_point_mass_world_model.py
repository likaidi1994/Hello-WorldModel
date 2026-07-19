from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from hello_world_model import (  # noqa: E402
    PointMass1DEnv,
    WorldModel,
    collect_random_data,
    evaluate_planner,
    plot_dataset_phase_space,
    plot_one_step_predictions,
    plot_planned_trajectory,
    plot_training_loss,
    rollout_planner_episode,
    train_world_model,
)


def main() -> None:
    output_dir = PROJECT_ROOT / "outputs" / "point_mass_world_model"

    env = PointMass1DEnv()
    print("1. Collecting random interaction data...")
    data = collect_random_data(env, episodes=300)
    plot_dataset_phase_space(data, output_dir / "01_random_data_phase_space.png")

    model = WorldModel()
    print("2. Training the world model...")
    losses = train_world_model(model, data, epochs=60, batch_size=128, lr=1e-3)
    plot_training_loss(losses, output_dir / "02_training_loss.png")
    plot_one_step_predictions(model, data, output_dir / "03_one_step_predictions.png")

    print("3. Evaluating model-based planning...")
    evaluate_planner(env, model, episodes=10)

    print("4. Saving one planned trajectory...")
    states, actions, rewards = rollout_planner_episode(env, model)
    plot_planned_trajectory(states, actions, rewards, output_dir / "04_planned_trajectory.png")

    print(f"Saved visualizations to: {output_dir}")


if __name__ == "__main__":
    main()
