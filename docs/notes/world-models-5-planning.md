什么是world model planning ? ans:让世界模型“在脑中试动作”，再决定现实里做什么；不直接盲目执行动作，而是在模型里先模拟几种未来，再选最好的那个。

这就是最朴素的 planning with a world model。
先把这件事说成人话

假设你现在在某个状态，手里有 3 种可能动作：

往左一点
不动
往右一点

你不急着马上做，而是先在脑子里想：

如果我连续往左几步，会发生什么？
如果我先右后左，会怎样？
哪种动作序列累计 reward 更高？

最后你只执行那个最好序列的第一步动作。
走完这一步，再重新观察、重新想。

这叫：

receding horizon planning
也就是“滚动规划”。
为什么只执行第一步，而不是整串都执行

因为模型不是完美的。

如果你现在直接把未来 10 步都押上去，后面可能越来越偏。
更稳妥的做法是：

用模型想象未来 10 步
选出最好的一条动作序列
只执行第一步
回到真实环境重新观察
再规划下一步

这是 model-based control 里非常经典的思路。

4. 先给模型加一个“单步 rollout”接口

我们现在的 GRU 模型是按整段序列跑的。
为了 planning，更方便的做法是加一个单步接口：

输入：

当前观测 obs_t
当前动作 a_t
当前 hidden h_t

输出：

预测下一观测
预测 reward
下一 hidden
import torch
import torch.nn as nn

class GRUWorldModelWithReward(nn.Module):
    def __init__(self, obs_dim=1, act_dim=1, hidden_dim=32):
        super().__init__()
        self.input_dim = obs_dim + act_dim
        self.hidden_dim = hidden_dim

        self.gru = nn.GRU(
            input_size=self.input_dim,
            hidden_size=hidden_dim,
            batch_first=True
        )
        self.next_obs_head = nn.Linear(hidden_dim, obs_dim)
        self.reward_head = nn.Linear(hidden_dim, 1)

    def forward(self, obs_seq, act_seq, h0=None):
        x = torch.cat([obs_seq, act_seq], dim=-1)   # [B, T, obs_dim+act_dim]
        out, hn = self.gru(x, h0)
        pred_next_obs = self.next_obs_head(out)
        pred_reward = self.reward_head(out)
        return pred_next_obs, pred_reward, hn

    def step(self, obs, act, h):
        """
        obs: [B, obs_dim]
        act: [B, act_dim]
        h:   [1, B, hidden_dim] or None
        """
        x = torch.cat([obs, act], dim=-1).unsqueeze(1)  # [B, 1, input_dim]
        out, hn = self.gru(x, h)                        # out: [B, 1, hidden_dim]
        out = out[:, 0, :]                              # [B, hidden_dim]

        pred_next_obs = self.next_obs_head(out)         # [B, obs_dim]
        pred_reward = self.reward_head(out)             # [B, 1]
        return pred_next_obs, pred_reward, hn
5. 当前 hidden state 从哪里来

这个问题很重要。

模型做 planning 时，不能从空白脑子开始。
它需要一个已经编码了历史的 hidden state。

所以我们先写一个函数：
把到当前为止的真实历史喂进去，得到当前 hidden。

@torch.no_grad()
def infer_hidden_from_history(model, obs_hist, act_hist, device="cpu"):
    """
    obs_hist: numpy array [T, obs_dim]
    act_hist: numpy array [T, act_dim]
    return: hidden state [1, 1, hidden_dim]
    """
    model.eval()

    obs = torch.tensor(obs_hist, dtype=torch.float32, device=device).unsqueeze(0)  # [1, T, obs_dim]
    act = torch.tensor(act_hist, dtype=torch.float32, device=device).unsqueeze(0)  # [1, T, act_dim]

    _, _, hn = model(obs, act, h0=None)
    return hn

这个 hn 就是：

模型基于真实历史形成的当前内部信念。

6. 在模型里评估一条动作序列

这是 planning 的核心函数。

import numpy as np

@torch.no_grad()
def evaluate_action_sequence(model, current_obs, current_hidden, action_seq, device="cpu"):
    """
    current_obs: numpy array [obs_dim]
    current_hidden: torch tensor [1, 1, hidden_dim]
    action_seq: numpy array [H, act_dim]
    """
    model.eval()

    obs = torch.tensor(current_obs, dtype=torch.float32, device=device).unsqueeze(0)  # [1, obs_dim]
    h = current_hidden.clone()
    total_reward = 0.0

    for t in range(len(action_seq)):
        act = torch.tensor(action_seq[t], dtype=torch.float32, device=device).unsqueeze(0)  # [1, act_dim]
        pred_next_obs, pred_reward, h = model.step(obs, act, h)

        total_reward += pred_reward.item()
        obs = pred_next_obs   # 关键：把预测出来的 next_obs 继续喂回去

    return total_reward

这里最关键的一句是：

obs = pred_next_obs

这意味着：

你已经不依赖真实环境了，而是在模型里“想象未来”。

这就是世界模型真正开始工作的时刻。

7. 随机采样很多动作序列，选最好的那条

这是最简单的 random shooting。

@torch.no_grad()
def plan_action_with_world_model(
    model,
    current_obs,
    current_hidden,
    horizon=15,
    num_samples=300,
    action_low=-1.0,
    action_high=1.0,
    device="cpu"
):
    best_return = -1e18
    best_first_action = None
    best_seq = None

    for _ in range(num_samples):
        action_seq = np.random.uniform(
            action_low, action_high, size=(horizon, 1)
        ).astype(np.float32)

        total_reward = evaluate_action_sequence(
            model=model,
            current_obs=current_obs,
            current_hidden=current_hidden,
            action_seq=action_seq,
            device=device
        )

        if total_reward > best_return:
            best_return = total_reward
            best_first_action = action_seq[0].copy()
            best_seq = action_seq.copy()

    return best_first_action, best_seq, best_return
8. 把它接回真实环境：滚动规划控制

现在我们把 planning 真正接进环境控制闭环。

def run_planning_episode(env, model, horizon=15, num_samples=300, device="cpu"):
    obs = env.reset()
    done = False
    total_real_return = 0.0

    obs_hist = []
    act_hist = []

    step_idx = 0

    while not done:
        # 先根据到当前为止的真实历史得到 hidden
        if len(obs_hist) == 0:
            current_hidden = None
        else:
            current_hidden = infer_hidden_from_history(
                model,
                obs_hist=np.stack(obs_hist),
                act_hist=np.stack(act_hist),
                device=device
            )

        # 用世界模型选动作
        first_action, best_seq, imagined_return = plan_action_with_world_model(
            model=model,
            current_obs=obs,
            current_hidden=current_hidden,
            horizon=horizon,
            num_samples=num_samples,
            device=device
        )

        # 执行第一步真实动作
        next_obs, reward, done, _ = env.step(float(first_action[0]))
        total_real_return += reward

        # 把真实 transition 写入历史
        obs_hist.append(obs.copy())
        act_hist.append(first_action.copy())

        obs = next_obs
        step_idx += 1

    return total_real_return
9. 这里有一个非常值得你盯住的细节

刚才的流程里，规划用的是模型 imagined reward，
但真正累计成绩时，用的是真实环境 reward。

这两者不一样。

这正是世界模型方法最核心的张力：

你靠模型思考
但最后成败，要由真实世界裁判

所以如果模型不准，planning 就会被带偏。

10. 这一步你必须真正理解的核心

这不是“多了个搜索函数”那么简单。
这里发生了一个质变：

之前

模型只是：

给定输入
预测输出
现在

模型变成了：

内部模拟器
可用来比较未来
可用来选动作

也就是说，世界模型第一次真正具备了“思考未来”的功能。

11. 但这个最小版本有一个明显限制

你大概率已经能感觉到了：

我们在 imagined rollout 时，把 pred_next_obs 继续当成下一步输入。
这样一滚很多步，误差会越来越大。

这就是 compounding error。

也是为什么真正高级 world model 通常不直接在 observation 空间里硬滚很多步，而更喜欢：

在 latent space 里滚
用 stochastic latent 建模不确定性
用更稳的训练目标做多步一致性约束

所以你现在已经站在一个很重要的分界点：

你会发现“最小 world model 能工作”，
也会发现“它为什么很容易出问题”。

这正是研究的入口。

12. 这一步和 PlaNet / Dreamer 的关系

你现在这套最小系统，本质上已经有了 PlaNet/Dreamer 的雏形：

有内部状态
有 dynamics
有 reward prediction
有 imagined rollout

只是它还很初级：

hidden state 是纯确定性的
rollout 在 observation 空间进行
planning 很粗糙，用 random shooting
还没有 actor / value learning

但方向已经完全对了。

13. 你这轮只要抓住这三句话
第一句

planning 的本质，就是在模型内部比较不同动作序列带来的未来累计 reward。

第二句

真正执行时，通常只执行最好序列的第一步，然后重新规划。

第三句

world model 的价值，不在于“会预测下一帧”，而在于“能帮助你选择动作”