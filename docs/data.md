# 数据契约与资产范围

原始游戏视频和完整统一输出留在本地。仓库包含代码、示例、数据契约、聚合统计与少量展示图，不包含完整游戏资源、模型权重或个人路径配置。

## 一个统一案例

```text
data/unified_worlds/<game>/<case>/
  episode.json              # 世界约定、源视频、视角、点轨迹语义
  entities.json             # 共同实例ID与原生身份 / 来源
  frames.jsonl.gz           # 帧状态、相机、变换、关节、有效状态
  first_frame.json
  static.npz                # 粗静态网格、每面实例 / 类别
  events.jsonl              # 命令 / 观测事件
  timeline.npz              # 原生相对time_us、源帧ID、视频帧编号
  program.json              # 重放入口、操作符、控制器证据
  qa.json                   # 来源质量与覆盖限制
  joints.npz                # 共同20关节、根位置 / 速度、掩码
  points.npz                # 源点 / 骨骼驱动代理点轨迹
  proxy/main/
    white.mp4
    semantic.mp4
    actors.mp4
    control.mp4
    preview.jpg
    frames.jsonl
    render_report.json
    buffers/chunk_00000.npz  # 默认32帧一块
```

根目录 `training_pairs.jsonl` 列出每个案例 / 相机的引用，`generated_proxy_available` 标识代理是否完整。源 RGB 通常只保留本地路径，不复制。

## 共同约定

| 字段 | 含义 |
|---|---|
| `schema` | `worldbridge.scene.v1` |
| `game/case/native_id` | 实体命名空间；每案例整数 `instance_id` 从1开始 |
| `time_us` | 以案例起始为基准的原生仿真时间，严格递增 |
| `index` | 连续视频配对帧编号；不要把它与原生游戏帧ID混淆 |
| `world` | 局部原点位于首个主相机，Z向上，保留原生水平轴 |
| `distance_unit` | 按1游戏单位=1米解释，尚未独立标定真实尺度 |
| `camera_axes` | X向右 / Y向上 / Z向前 |
| `ndc_depth` | `minus_one_to_one`，供OpenGL投影使用 |
| `frame_semantics` | 有范围的观测，并非完整世界清单 |

缺少实体表示未知，不等于销毁。缺少布尔状态为 null，不等于 false；数值轨迹缺失为 NaN 并附有效掩码。JSON 帧流不写 NaN；NaN 仅在数值数组中使用。

`entities.json` 承担整数ID到来源身份的映射。不要硬编码 focus ID=1：历史 CP2077 首案例的目标实例为165。帧记录和轨迹都应通过实例表关联。

## 数组

`joints.npz`：

| 键 | 形状 / 类型 |
|---|---|
| `world_xyz` | `[T,E,20,3] float32` |
| `valid` | `[T,E,20] bool` |
| `root_world`, `velocity_world` | `[T,E,3] float32` |
| `root_valid`, `velocity_valid` | `[T,E] bool` |
| `instance_ids` | `[E]` |
| `joint_names`, `parent_joint` | 共同关节名称与父节点索引 |
| `time_us` | `[T] int64` |

规范关节包括 pelvis、spine、neck、head、左右臂与腿。并非每游戏每关节都有数据；只监督 `valid & finite`，不能把未采到的关节补为零坐标。

`points.npz` 包含 `world_xyz[T,E,P,3]`、`valid[T,E,P]`、`instance_ids`、`point_ids`、`time_us` 和 `tracking_kind`。CP / GTA 的 P=256 为骨骼驱动代理点，RDR 的 P=19 为源骨骼点。它们不是相同真实表面点。RDR 中只保留轨迹、缺完整状态的实体标为 `track_only`。

`buffers/chunk_*.npz`：

| 键 | 含义 |
|---|---|
| `depth` | `[T,H,W] float32`，全几何相机Z，背景 +inf |
| `instance` | `[T,H,W] uint32`，0表示无代理覆盖 |
| `semantic` | `[T,H,W] uint16`，0为unknown |
| `covered` | `[T,H,W] bool`，代理覆盖而非原生可见性 |
| `actor_depth/instance/semantic` | 独立动态层对应量 |
| `time_us`, `source_frame` | 仿真时间与源视频帧编号 |

白模、语义与深度在全几何下使用正常深度测试。actors 不受静态代理遮挡，动态实体之间仍深度测试。control 为动态优先合成，不能与全几何 depth 当成相同遮挡约定。

## 历史批次规模与限制

305案例 / 78,409主视角时间步，代理分辨率384×216，生成结果约17.25GiB，RGB未复制。GTA5 另有241个空目录未计入完整案例。

- CP2077：缺共享静态几何与原生深度，主要是目标骨骼和近邻边界。
- GTA5：碰撞墙段 / 平面与粗边界；水平面按平均高度处理，不能当原生精确表面。
- RDR2：粗模板 / 地形与部分覆盖，源 `training_eligible=false` 不能因成功转换被提升。
- 聚合 `track_only_entities=1906` 为逐案例身份之和，不是跨案例唯一人物数。
- 点轨迹记录数包含带有效掩码的记录，不等于全部都是真实有效表面标签。

## 本地路径与迁移

转换产物保留源路径，移动原始视频后 `episode.json` 中的 `source_video` 引用会失效。推荐在新机器用本机源目录和新输出根重新转换；不要只修改网页入口后假定所有嵌套引用已更新。

窗口JSON里的相对 `episode` 按JSON文件目录解析。网页导出的绝对路径方便本机使用，分享窗口前应改为本机可移植的相对路径。示例使用 `../data/unified_worlds/...`。

## 文件级验证范围

结构校验读取全部帧流、全部轨迹和掩码、所有缓冲头部 / 时间、每案例首帧深度与实例约束以及源视频元数据。它没有逐像素检查全部缓冲，也没有完成全部RGB与代理视觉验收。聚合历史结果见 `reports/`。

