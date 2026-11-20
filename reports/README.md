# 验证记录

这里的JSON是已完成本地批次的聚合与验收快照。它们不包含完整数据集，也不能替代新机器、新数据或新代码的重新验证。

| 文件 | 范围 |
|---|---|
| `dataset_summary.json` | 305案例、四相机索引、已渲染main视角数量 |
| `tracks_summary.json` | 三游戏帧数、点记录数、逐案例track_only身份计数 |
| `validation_summary.json` | 历史305案例结构校验聚合；原始逐案例绝对路径未发布 |
| `acceptance.json` | 历史本地网页63项接口 / 浏览器 / 样本准备检查 |
| `decoded_frame_validation.json` | 三游戏RGB与white首/中/末共18项浏览器 / OpenCV解码帧比较 |
| `repository_validation.json` | 仓库整理后本地测试与实际数据接入复查，推送前生成 |

历史批次结构失败0，`visual_quality_approval_inferred=false`。网页复核记录不提升原始训练资格。解码帧检查只证明浏览器选中了正确视频帧，不证明粗代理与RGB内容已经几何精确对齐。

