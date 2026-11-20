# 如何用于训练

这套数据的价值是把 RGB、相机、世界状态和代理对齐，并明确哪些信息可以作为输入、哪些只能作为监督。当前脚本准备数组，不训练模型权重。

## 推荐顺序

1. 人工复核10–20条，核对相机、目标身份、骨骼、粗静态误遮挡与画外返回。RDR源训练资格为false的原因需要先处理。
2. **先按场景、角色身份、采集批次分组，再切窗**，避免同一录像的相邻帧跨训练 / 验证 / 测试集。
3. 单案例过拟合条件渲染器，确认结构能控制生成。
4. 独立训练 / 评价画外动力学，与保持位置、历史匀速比较。
5. 上述接口成立后，再接视频模型内部记忆与联合目标。

## A：条件视频渲染

输入为历史RGB / 外观、**已知未来代理**和显式相机条件；监督为对应未来RGB。这个任务允许未来代理，因为要检验的是“给定结构能否生成画面”。

选定基础视频模型后，建议先冻结其主体和VAE，训练新增代理编码器 / 条件适配层，沿用基础模型原本的扩散或flow目标。具体条件注入位置、帧数、尺寸与优化器必须按模型配置确定，目前没有假设一个通用训练器。

先用actors / control减少粗静态误遮挡对运动条件的影响，同时保留white用于诊断。将无损深度、实例、类别作为额外条件时，各通道需按其语义预处理：离散ID和掩码用最近邻，浮点深度结合有效掩码处理无穷背景。

改写轨迹生成新代理后，原RGB不再是该轨迹的配对监督；需要重新采集或设计无配对目标。

## B：画外动力学

学生只读取历史截止 `t0` 之前的数据。默认输入为历史RGB和时间。可另做显式特权状态实验，使用历史引擎位置 / 速度，但应与仅RGB初始化分开报告。

未来世界关节、根位置 / 速度、点轨迹和RGB是监督。**未来真实状态、速度、事件、代理、monitor / follow等辅助画面不能进入学生输入。** 辅助视角可用于离线教师建标签。

目标可从masked Huber位置损失开始：

```python
valid = target_valid & torch.isfinite(target_xyz).all(dim=-1)
# 避免先对NaN计算再乘0；对有效位置索引计算。
if valid.any():
    loss_position = torch.nn.functional.smooth_l1_loss(
        prediction[valid], target_xyz[valid], reduction="mean"
    )
```

此片段展示损失约定，并非完整训练器。可逐步加入速度 / 加速度一致性、身份与返回位置目标。损失对有效元素归一化；画外但仍有有效世界标签的点可以参与监督。

先使用共同20关节与根位置。CP/GTA 256代理点与RDR 19骨骼点不能被当作相同真实表面对应。未来相机若外部给定，应单独标记已知条件；若任务要求预测相机，则也需预测。未来查询时间可显式给定，不等于允许未来实体状态进入输入。

## C：内部动态记忆

将显式对象 / 世界状态作为监督，设计持久记忆token、历史写入、画外推进、未来时间 / 位置查询及视频生成读取。

K负责注意力寻址的一部分，V承载内容。仅说“训练KV中的K”不能确定记忆内容、更新规则和生成接口。需根据选定基础模型和论文结构核对编码器 / 查询头，再明确K/V投影、记忆适配器或状态模块的训练范围。

可先冻结基础视频模型，训练记忆读写与3D查询头，再逐步联合轨迹目标和生成目标。当前仓库没有接入D4RT，也没有证明K-only训练有效。

## 实际准备样本

在网页选择历史起始帧、截止帧和目标终帧，导出窗口JSON。要求：

```text
0 ≤ start ≤ cutoff < end < episode.frame_count
context = [start, cutoff]        # 两端包含
target  = [cutoff + 1, end]       # 两端包含
```

从仓库根目录执行：

```powershell
python scripts/prepare_window.py --recipe "window.json" --out prepared/sample_001
# 显式特权历史状态基线，使用不同输出目录：
python scripts/prepare_window.py --recipe "window.json" --out prepared/sample_001_oracle --oracle-history
```

输出目录必须新建或为空，脚本不覆盖已有样本。

| 文件 | 内容 |
|---|---|
| `student_inputs.npz` | 默认 `past_rgb/past_time_us`；renderer额外含未来proxy / 相机；oracle额外含历史引擎数组 |
| `supervision.npz` | 未来RGB、关节、根位置 / 速度、点轨迹、掩码和ID |
| `provenance.json` | 配方、解析后的本地案例路径、输入模式、形状、剩余事项 |

动力学默认输入断言不含 `future_*` 键。不要把整个监督字典拼回模型输入。脚本默认按384×216准备，完整视频模型仍需自己的时空预处理。

`training_ready=false` 表示窗口尚需视觉质量与划分审核，不是服务异常。准备数组可用于调试，但不自动覆盖源质量状态。

## 直接读取帧配对

```python
from pathlib import Path
from worldbridge.data import iter_pairs
import numpy as np

episode = Path("data/unified_worlds/gta5/your_case")
for pair in iter_pairs(episode, condition="control", stride=2):
    rgb, proxy, time_us = pair["rgb"], pair["proxy"], pair["time_us"]
    # covered是代理覆盖，不是原生可见性。

with np.load(episode / "joints.npz", allow_pickle=False) as z:
    xyz = z["world_xyz"]
    valid = z["valid"] & np.isfinite(xyz).all(axis=-1)
```

`target_size` 只调整RGB目标尺寸，代理保持原生分辨率。条件与目标若需相同空间尺寸，应在选定模型的数据预处理里明确一起转换。数值缓冲不通过MP4编码，直接读取NPZ。

## 评价与采集方向

- 动力学：画外ADE/FDE、重现位置误差、身份保持；使用原生时间间隔，区分视锥外与场景遮挡。
- 生成：结构对齐、身份 / 外观保持、时间一致性与视频质量；对照实验使用相同相机和数据划分。
- 记忆：无记忆、静态记忆、匀速推进、学习推进的消融；按画外时长和交互类型分桶。
- 需要物理 / 行为能力时，补采碰撞、接触、动作、目标、导航约束和状态转移；不能把粗代理几何当原生物理真值。

下一轮训练前需确定基础视频模型、窗口帧数 / 帧率 / 尺寸、条件注入模块、显存预算及通过复核的分组清单。

