# 复现、验证与目录要求

## 环境

Python 3.10+，numpy / Pillow / OpenCV；网页另需FastAPI / uvicorn / Pydantic2，测试需pytest / httpx。ffmpeg编码代理，ffprobe读取视频元数据与presentation PTS。

当前GPU渲染实现只支持Windows WGL与OpenGL4.3+驱动。历史批次在NVIDIA RTX5060 Laptop GPU上运行；不声称其他GPU或系统已经验证。Linux可以使用转换、读取与网页，但不能直接调用当前WGL渲染器。

## 原始数据发现

适配器发现案例的标志文件：

| 游戏 | 搜索标志 | 主要输入 |
|---|---|---|
| cp2077 | `code/data/episode.json` | `code/data/states.jsonl`、`video/*_present.jsonl`、`video/*.mp4`、`result.json` |
| gta5 | `video/main.mp4` | `code/data/episode.json`、`code/data/frames.jsonl`、`video/*_present.jsonl`、场景布局 / 膜面、result等 |
| rdr2 | `code/dynamics.npz` | runtime observations、cameras、static_compiled、actions / transitions / state_machine、videos等 |

标志文件用于发现，不表示缺失其他文件仍能转换。完整字段与读取路径以 `worldbridge/adapters.py` 为准；缺少依赖数据会明确报错，不自动编造。

## 小批处理

```powershell
Copy-Item configs/example.json configs/local.json
# 编辑三个源根目录；保持output指向新的可写目录。
python -m worldbridge scan --config configs/local.json
python -m worldbridge batch --config configs/local.json --game gta5 --limit 1 --workers 1 --render-workers 1
python -m worldbridge validate --root data/unified_worlds --workers 1
```

确认一条RGB / 代理对齐后去掉game/limit做全量。Windows进程池的入口在可导入的 `workers.py`，不要把worker函数搬回 `__main__`，否则spawn可能无法反序列化函数。

`convert`、`batch`会复用已有完成案例。修改适配规则后使用新的 `--out data/unified_worlds_v2`，不要假定旧目录会自动失效。未完成阶段写入 `.partial`，旧阶段目录会改名保留供排错，不递归删除源文件。

需要其他相机代理时：

```powershell
python -m worldbridge render --episode "data/unified_worlds/gta5/your_case" --view follow --width 384 --height 216
python -m worldbridge index --root data/unified_worlds
```

## 显式程序推进

```powershell
python examples/predict_proxy.py --episode "data/unified_worlds/gta5/your_case" --out runs/constant_velocity --frame 0 --steps 90 --dt 0.0333333333
```

输出记录初始化帧、推进步长与假设。它从一个已观测引擎状态初始化，保持相机与关节姿态，仅平移根和实测关节，不读取未来实体状态。它是特权初始化基线，不是仅RGB预测或学习式物理系统。

## 测试与报告

```powershell
python -m pytest -m "not gpu" -q
python -m pytest -q
```

GPU测试检查真实深度遮挡、实例ID、语义和无覆盖背景。非Windows会跳过；Windows未具备图形驱动时用 `-m "not gpu"`。其余测试检查投影原点不变性、缺失与false、类别映射、画外显式推进、因果窗口与本地网页接入。

真实数据全量验证：

```powershell
python -m worldbridge validate --root "D:/datasets/unified_worlds" --workers 3
```

检查范围为全部统一帧流与轨迹、全部缓冲头部 / 时间、每案例首帧深度ID和源视频元数据；不代表全量视觉验收或真实表面真值。历史汇总见 `reports/`。

## 提交资产的可复现边界

代码与数据契约、示例窗口可直接检查；统计与截图来自已完成本地批次。完整源数据未放到Git，第三方仅克隆仓库不能重建305案例，需要提供相同格式的本地交付。没有模型权重、训练结果或生成质量指标被隐含发布。

