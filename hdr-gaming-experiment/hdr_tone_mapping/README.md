# HDR tone mapping

从 ColorVideoVDP 项目根目录运行，使用 cvvdp 环境。

## 运行命令

```powershell
conda activate cvvdp

# 实验用：前 600 帧（10 秒）的 HDR1000 / HDR400 校准 spline + SDR ACES Filmic
python -B hdr-gaming-experiment/hdr_tone_mapping/run.py "C:\Users\xw506\Videos\OBS\HDR\clips\returnal\returnal_clip_005.mp4"

# 前 12 帧测试；覆盖已有结果
python -B hdr-gaming-experiment/hdr_tone_mapping/run.py "C:\Users\xw506\Videos\OBS\HDR\clips\returnal\returnal_clip_005.mp4" --max-frames 12 --overwrite

# 只画曲线（不处理视频）
python -B hdr-gaming-experiment/hdr_tone_mapping/run.py --plot-only

# 参考图：旧 simple spline 与校准 spline 对比（仅供讨论，不生成视频）
python -B hdr-gaming-experiment/hdr_tone_mapping/plot_simple_spline.py

# 实测显示器响应图（白、红、绿、蓝四条测量曲线）
python -B hdr-gaming-experiment/hdr_tone_mapping/asushdr_display/plot_measurements.py
```

命令行参数是要处理的视频文件的完整路径（一次一个视频；之后批量处理可以由 CSV 循环逐个传入路径）。输出文件名取输入文件名，如 `returnal_clip_005_hdr1000_calibrated.mkv`。

输入须为 limited-range BT.2020/PQ（OBS HDR 录制符合）。`--max-frames` 默认 600，即只处理前 600 帧（60 fps 下 10 秒），和 `compression/compress.py` 的默认值一致，两边的文件逐帧对应。原因：`extract_clips.py` 用 stream copy 剪出的片段解码后是 601 帧，前 600 帧与原片逐帧一致，第 601 帧跳过了原片的 2 帧。`--device auto|cpu|cuda` 选择映射设备，默认有 CUDA 就用 CUDA。`-B` 避免生成 Python 缓存。

视频读写继承 ColorVideoVDP 自带的 [`video_reader`](../../pycvvdp/video_source_file.py) 和 [`VideoWriter`](../../pycvvdp/video_writer.py)，原库不做修改，需要改的地方都在 `video_io.py` 里替换。

## 目录及脚本

```text
hdr_tone_mapping/
  run.py                  # 运行入口：画曲线、转换视频、验证、写报告
  hdr_spline.py           # 校准 HDR spline：WhiteResponse 由 white CSV 求拐点和弯平形状；HDRSpline 逐帧映射
  sdr_aces.py             # SDR ACES Filmic（Narkowicz 拟合）：转 BT.709 后逐通道应用，sRGB 编码
  color_encoding.py       # 共用：PQ/sRGB 传递函数、BT.2020 常数
  video_io.py             # 继承 pycvvdp 的 video_reader / VideoWriter，替换 ffmpeg 命令和写帧方式；输出完整解码验证
  plot_simple_spline.py   # 参考图：simple spline vs 校准 spline
  tone_map_output/        # 所有输出
  asushdr_display/        # ASUS HDR 显示器实测数据
    resp_asushdr_pq1000_{white,red,green,blue}.csv  # 测量 CSV；HDR spline 只用 white
    plot_measurements.py  # 画测量曲线 -> asus_hdr_response_{log,linear}.png
```

每个映射器都是一个可调用对象：输入线性 BT.2020 RGB（nits，PyTorch 张量，CPU 或 CUDA），输出编码后的帧。新增一种映射时，在 `run.py` 的 `build_mappers()` 里加一项即可；输出名以 `sdr` 开头的按 SDR（BT.709/sRGB、8-bit）编码，其余按 HDR（BT.2020/PQ、10-bit）编码。

## 实验条件

| 输出 | 映射 |
|---|---|
| `hdr1000_calibrated` / `hdr400_calibrated` | 实测拐点（约 85% 峰值）以下保持 1:1，之后按实测弯平形状到 1000/400 nits；同一曲线分别作用于线性 BT.2020 的 R、G、B |
| `sdr100_aces` | 转线性 BT.709，乘曝光后把 Narkowicz ACES Filmic 分别作用于 R、G、B，再 sRGB 编码 |

测量来源：`hdr-gaming-experiment/hdr_tone_mapping/asushdr_display/resp_asushdr_pq1000_white.csv`（`--white-csv` 可替换）。HDR spline 由 CSV 决定两点：

- 拐点：实测曲线最后一次 log-log 斜率由 ≥1 变为 <1 的位置，约为平台亮度（292.9 nits）的 85%。HDR400 约 339 nits，HDR1000 约 848 nits。拐点以下保持 1:1。
- 弯平形状：拐点以上直接使用实测曲线从拐点到平台的那一段，按比例放到 400/1000 nits；约在输入 445/1113 nits 时到顶，之后保持峰值。接缝处斜率连续，整条曲线不超过 1:1。

每个通道都不超过峰值，高亮饱和色会向白色靠拢。曲线图中虚线（Calibrated simulation）是按中间调对齐、缩放到 400/1000 nits 的实测形状，仅作对照。实测峰值约 293.3 nits；400/1000 是模拟条件，曲线跟随实测显示器的拐点和弯平方式，不是显示器逆校准。CSV pixel 的物理单位仍需采集程序确认。

SDR 按 [Narkowicz ACES Filmic](https://knarkowicz.wordpress.com/2016/01/06/aces-filmic-tone-mapping-curve/) 原文用法：线性 Rec.709 输入先乘曝光，再逐通道套 `ACESFilm()`，之后做 gamma（sRGB）编码；HDR 和 SDR 因此都是逐通道映射。该拟合已预曝光（输入 1 → 输出约 0.8），不另乘原文为匹配原始 ACES 曲线而提的 0.6。超出 BT.709 色域的负分量在曲线前截为 0。输入曝光尺度暂为 100（`--exposure-nits`，即 100 nits 对应 x=1），SDR 白色按 100 nits 解释，仍需正式确认曝光与播放校准。

Simple spline（亮度 Y 上的设计曲线，200/500 nits 以下 1:1）已不再用于生成视频，只保留在 `plot_simple_spline.py` 中画图对比。

## 输出与编码

`tone_map_output/`：三个 MKV、主曲线图 `tone_curves_with_aces.png`（校准 HDR spline + SDR ACES）、`calibration_summary.json`（拐点、峰值等标定结果，只取决于 white CSV，每次运行都会重写这一个文件）；`plot_simple_spline.py` 另存 `simple_vs_calibrated_spline.png`。覆盖同名视频须显式加 `--overwrite`。

流程：

- 读：`HDRReader`（继承 `video_reader`，只替换 ffmpeg 命令，明确按 limited-range BT.2020 转 RGB48）。`get_frame()` 取一帧，`unpack()` 得到 [0, 1] 的 PQ 编码 RGB 张量，再用 `pq_decode` 转成线性 nits。
- 映射：PyTorch（默认 CUDA）完成 tone mapping 和 PQ/sRGB 编码。
- 写：`HDRExperimentWriter`（继承 `VideoWriter`，替换 ffmpeg 命令和 `write_frame_rgb()`），帧四舍五入成整数 RGB 送入 ffmpeg（HDR 16-bit，SDR 8-bit），在 CPU 上用 FFV1 无损编码（level 3、4:2:0、MKV）。HDR 为 BT.2020/PQ、10-bit；SDR 为 BT.709/sRGB、8-bit，和游戏先渲染 8-bit RGB、再由编码器转 YUV 4:2:0 的顺序一致。

替换 writer 命令的原因：原版 `VideoWriter` 没有指定 RGB→YUV 矩阵，ffmpeg 默认用 BT.601，但 HDR 文件标成 BT.2020、SDR 不写标签，播放时饱和颜色会偏（灰色不受影响）。替换后的命令明确指定矩阵（HDR BT.2020、SDR BT.709）和颜色标签，已用纯色块验证。替换 `write_frame_rgb()` 的原因：原版把帧截断成整数，8-bit SDR 整体约暗半个色阶；这里改为四舍五入。原库暂不修改。

验证包括帧数（按 packet 计数）、分辨率、帧率、颜色标签，以及完整解码一遍；全部通过后才改成最终文件名，`.partial.mkv` 为未完成文件。

为什么无损：这些文件既是路线 1（先 TM 再压缩）的压缩输入，也是参考视频。之前用 CRF 12 有损保存时，ColorVideoVDP 测得 HDR 输出比无损版低约 0.4 JOD；拿它当压缩输入和参考时，路线 1 在 2160p / 25M 下被高估约 0.1 JOD（前 120 帧，`standard_hdr_pq`）。无损文件每 10 秒约 2.6 GB（HDR）/ 1.4 GB（SDR），一般播放器播不流畅，只用于后续处理和打分；主观实验播放用的文件另外统一转码。保留分辨率、标称帧率和解码帧顺序，不保留音频或原始可变时间戳。
