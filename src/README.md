# Source Code

这里存放世界模型相关的可复用代码。

初期可以先从最小模块开始：

- 环境交互代码
- 数据收集代码
- 状态预测模型
- 训练脚本
- 评估脚本

建议原则：

- 每个实验都能从 README 或注释中看懂入口。
- 优先写小而清楚的代码，再逐步抽象。
- 每次实验都记录对应的笔记链接。

## 当前代码

- [hello_world_model](hello_world_model)：最小 world model 教学代码。

它包含：

- `envs.py`：一维点质量环境。
- `models.py`：MLP world model 和 GRU world model。
- `training.py`：完整状态下的数据收集、训练和评估。
- `memory.py`：partial observation 序列数据、GRU 训练和 reward head 训练。
- `planning.py`：random shooting 规划器。
- `visualization.py`：训练过程和规划结果的可视化。

对应笔记：

- [MDP、POMDP 与最小世界模型](../docs/notes/world-models-basics.md)
- [为什么世界模型需要 memory 与 latent state](../docs/notes/world-models-latent-memory.md)
- [为什么世界模型需要预测 reward](../docs/notes/world-models-reward.md)
