# HDR tone mapping

从 ColorVideoVDP 项目根目录运行，使用 cvvdp 环境。只需选择下列两个入口之一。

## 运行命令

```powershell
conda activate cvvdp

# Simple spline：完整视频，GPU 映射
python -B hdr_tone_mapping/simple_spline/run.py --device cuda

# Calibrated simulation：完整视频，CUDA 映射
python -B hdr_tone_mapping/calibrated_simulation/run_calibrated.py --device cuda
```

测试前 12 帧，在命令后加 `--max-frames 12`；覆盖已有结果加 `--overwrite`；只画曲线加 `--plot-only`。不指定帧数则处理到视频结束。`-B` 避免生成 Python 缓存。

默认输入：

```text
C:\Users\xw506\Videos\Elgato Studio\HDR\test\clips\input05\input05_clip_002.mp4
```

`--input "视频路径"` 可替换为另一个单视频，目前不遍历文件夹。

## 目录及脚本

```text
hdr_tone_mapping/
  video_io.py                    # 共用视频读写及输出验证
  sdr_mapping.py                 # 两版统一的亮度ACES（CPU/CUDA）
  simple_spline/
    run.py                       # Simple 运行入口
    tone_mapping.py              # Simple spline、ACES、颜色转换
    frame_mapping.py             # Simple CUDA逐帧映射
    README.md
    tone_map_output/             # Simple 输出
  calibrated_simulation/
    run_calibrated.py            # Simulation 运行入口
    white_response.py            # White CSV 拟合、尺度调整、暗部修正
    frame_mapping.py             # CUDA 逐帧映射
    README.md
    tone_map_output/             # Simulation 输出
```

测试脚本已删除。其余 Python 文件均参与运行，支持模块不需要单独执行。`.gitignore` 仅防止把生成输出和缓存提交到 Git，不影响实验。

## 实验条件

| 版本 | HDR400 / HDR1000 | SDR ACES Filmic |
|---|---|---|
| Simple | 亮度 Y 的设计 spline；200/500 nits 以下保持 1:1，高亮压缩到目标峰值 | 亮度 Y 上 ACES，再缩放 RGB、转 BT.709 |
| Simulation | 根据 white CSV 缩放响应形状至 400/1000 nits，保留高亮走势，只修正暗部 | 在亮度 Y 上应用 ACES，再缩放 RGB、转 BT.709 |

两种 HDR 都按 Y_out/Y_in 等比例缩放线性 RGB。两版 SDR 共用 sdr_mapping.py，使用相同的亮度算子；目前输入曝光尺度为 100，SDR 白色按 100 nits 解释，仍需正式确认曝光与播放校准。

Simulation 使用 `asushdr_display/resp_asushdr_pq1000_white.csv`。400-nit 曲线在 4 nits 以下保持 1:1，4–40 nits 平滑接回实测形状；1000-nit 曲线对应 10、10–100 nits。实测峰值约293.3 nits；400/1000 是模拟条件，不代表显示器实际完成逆校准。CSV pixel 的物理单位仍需采集程序确认。

## 输出与编码

各版本只写入自身的 `tone_map_output/`：三个 MP4、曲线 PNG、JSON 报告，不生成 PDF/CSV。覆盖同名结果须显式加 `--overwrite`。

两版共用 `video_io.py`：HDR 为 HEVC 10-bit、BT.2020/PQ，SDR 为 H.264 8-bit、BT.709/sRGB；CRF12、fast、bf=0、HDR hdr-opt、MP4 faststart。映射可使用 GPU，但编码仍为 CPU。验证帧数、分辨率、帧率、颜色标签和完整解码后才生成最终文件名；`.partial.mp4` 为未完成文件。

编码仍有损，高码率 4K 视频可能在 Media Player 中卡顿，不能仅凭格式验证认定为无损实验参考。保留分辨率、标称帧率和解码帧顺序，不保留音频或原始可变时间戳。


