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

- [MDP、POMDP 与最小世界模型](../docs/notes/world-models-basics.md)
