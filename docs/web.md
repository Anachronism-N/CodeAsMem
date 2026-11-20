# 本地网页

前端为本地HTML/CSS/JavaScript，后端为FastAPI+uvicorn，没有CDN、外部字体或模型调用。服务绑定127.0.0.1，源RGB通过HTTP字节范围按需读取。

## 启动与停止

从仓库根目录：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File web/start.ps1 -DataRoot "D:/datasets/unified_worlds"
# 端口已占用时：
powershell -NoProfile -ExecutionPolicy Bypass -File web/start.ps1 -DataRoot "D:/datasets/unified_worlds" -Port 8878
powershell -NoProfile -ExecutionPolicy Bypass -File web/stop.ps1
```

也可双击 `web/Open_WorldBridge.cmd` / `web/Stop_WorldBridge.cmd`。未指定DataRoot时使用环境变量或仓库 `data/unified_worlds`。

其他平台已有预生成数据时：

```bash
CODEASMEM_DATA_ROOT=/datasets/unified_worlds python web/app.py --port 8877
```

后台启动状态与日志保存在 `web/.local/`，该目录被Git忽略。停止脚本核对服务标识、PID、工作目录与进程参数，避免停止其他进程。页面关闭后部署服务继续运行，机器重启后需重新启动。

## 页面

| 页面 | 功能 |
|---|---|
| 案例浏览 | 游戏筛选、搜索、RGB/white/control对照、actors/semantic切换、播放与逐帧定位 |
| 世界轨迹与缓冲 | 主体XY轨迹、原生时间、代理视锥状态、深度/实例图与像素数值 |
| 人工复核 | 待检查 / 条件训练候选 / 需修复及备注 |
| 实现过程 | 处理链路、游戏差异、共同IR、代理语义和真实源码 |
| 训练准备 | A/B/C任务边界、窗口JSON下载、数组准备方式 |
| 数据与验证 | 当前动态统计、历史批次验证范围、文件下载和本地路径 |

统计从实际数据计算。缺少 `batch_validation.json` 时不自动视为结构通过；缺少可选CP采集审计时，用规范关节估计视锥状态，并标注其来源。当前浏览器只列出具有完整主视角 `render_report.json` 的案例。

## 数据位置

`CODEASMEM_DATA_ROOT` 应指向包含 `<game>/<case>/episode.json` 的统一结果根。只选择根目录并不能修复 `episode.json` 中失效的源视频路径。移动原始数据后，建议按新路径重新转换。

`CODEASMEM_AUDIT` 可选，指向本地CP采集审计JSON，其中 `cases` 可含 `case_id/location_id/segments`。没有该文件不阻止其他功能，不在仓库发布完整原始审计。

人工记录写到 `web/.local/reviews.local.json`，不修改E盘源文件、统一qa.json或源训练资格。标记“候选”不等于将原始 `training_eligible=false` 自动改成true。

## 时间同步

原始视频可能VFR，代理为CFR预览。暂停 / 拖动按帧编号选择，再分别映射源presentation PTS和代理时间。保留MP4非零起点；末帧可能小于1ms，使用微秒级边界。仿真 `time_us` 独立用于动力学时间。

播放中存在浏览器解码延迟，近似同步；做精确检查应暂停逐帧。历史验证另用浏览器Canvas与OpenCV相邻解码帧比较，不能只检查currentTime数值相等。

## 缓冲含义

深度可切完整几何与独立动态。显示色阶按当前帧2–98%分位，点击查询原数值。训练直接读取NPZ，不能从彩色深度截图逆推精确深度。

视锥条仅表示采集记录或关节估计在不在相机视锥内，不表示真实像素遮挡。control忽略静态代理遮挡，covered也不是原生可见性。

## 常见问题

- 空案例库：DataRoot不正确、尚未渲染main、或转换没有完成。先scan / batch，再重启服务。
- RGB打不开：检查episode里的本地源视频，ffprobe是否在PATH，以及浏览器是否支持该视频编码。
- 深度首次稍慢：服务按需解压NPZ并缓存最近数块。
- 修改源代码或加入新案例：服务启动时建目录索引，需要重启。
- 换DataRoot：先停止现有服务再启动；重复启动相同目录会复用服务。
- 数据量变了但历史报告仍为305：历史快照与当前动态统计分别展示，历史报告不代表新数据已通过。

