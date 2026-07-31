# Examples

这里存放可以直接运行的小实验。

计划中的第一个实验：

- 用一个简单环境收集 `(state, action, next_state)` 数据。
- 训练一个小模型预测 `next_state`。
- 对比真实下一状态和模型预测结果。

每个例子建议包含：

- 运行命令
- 依赖说明
- 预期输出
- 对应笔记

## 当前示例

### 1D point mass world model

```bash
pip install -r requirements.txt
python examples/run_point_mass_world_model.py
```

运行后会在 `outputs/point_mass_world_model` 下生成教学图示：

- `01_random_data_phase_space.png`：随机交互数据覆盖了哪些位置和速度。
- `02_training_loss.png`：world model 的训练误差是否下降。
- `03_one_step_predictions.png`：模型预测的下一状态是否接近真实下一状态。
- `04_planned_trajectory.png`：planner 是否把点质量系统推向目标位置。

对应笔记：

- [MDP、POMDP 与最小世界模型](../docs/notes/world-models-2-basics.md)

### Partial observation GRU world model

```bash
pip install -r requirements.txt
python examples/run_partial_observation_gru.py
```

运行后会在 `outputs/partial_observation_gru` 下生成教学图示：

- `01_partial_observation_ambiguity.png`：同一个位置观测可能对应不同隐藏速度。
- `02_gru_training_loss.png`：GRU world model 的训练误差。
- `03_gru_sequence_prediction.png`：真实位置序列与预测位置序列对比。
- `04_hidden_state_probe.png`：GRU hidden state 是否编码了隐藏速度信息。

对应笔记：

- [为什么世界模型需要 memory 与 latent state](../docs/notes/world-models-3-latent-memory.md)

### Reward world model

```bash
pip install -r requirements.txt
python examples/run_reward_world_model.py
```

运行后会在 `outputs/reward_world_model` 下生成教学图示：

- `01_reward_landscape.png`：toy 环境里 reward 如何由目标距离和动作代价决定。
- `02_reward_model_losses.png`：total / observation / reward 三种训练误差。
- `03_reward_prediction.png`：真实 reward 与预测 reward 的对比。
- `04_reward_guided_action_scores.png`：reward head 如何给候选动作排序。

对应笔记：

- [为什么世界模型需要预测 reward](../docs/notes/world-models-4-reward.md)

### Planning with a world model

```bash
pip install -r requirements.txt
python examples/run_planning_world_model.py
```

运行后会在 `outputs/planning_world_model` 下生成教学图示：

- `01_planning_model_losses.png`：用于 planning 的 recurrent reward world model 训练误差。
- `02_first_imagined_rollouts.png`：第一次规划时，模型想象出的多条未来轨迹。
- `03_receding_horizon_control.png`：滚动规划在真实环境中的控制轨迹。
- `04_imagined_vs_real_feedback.png`：模型预测的最佳 imagined return 与真实 reward 的对照。

对应笔记：

- [用世界模型做 planning](../docs/notes/world-models-5-planning.md)

### Latent dynamics world model

```bash
pip install -r requirements.txt
python examples/run_latent_dynamics_world_model.py
```

运行后会在 `outputs/latent_dynamics_world_model` 下生成教学图示：

- `01_observation_model_loss.png`：observation-space baseline 的训练误差。
- `02_latent_model_losses.png`：latent dynamics model 的训练误差。
- `03_open_loop_rollout_comparison.png`：真实轨迹、observation rollout、latent rollout 的对比。
- `04_rollout_error_growth.png`：open-loop 误差如何随 rollout horizon 增长。
- `05_latent_space_rollout.png`：模型在 latent space 中想象出的轨迹。

对应笔记：

- [为什么要转向 latent dynamics](../docs/notes/world-models-6-latent-dynamics.md)
