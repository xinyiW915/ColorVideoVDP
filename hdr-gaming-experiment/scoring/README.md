# Scoring

用 ColorVideoVDP 给 `playback/test_output/` 里的播放版打分，并画图。从 ColorVideoVDP 项目根目录运行，使用 cvvdp 环境。

## 运行命令

```powershell
conda activate cvvdp

# 一个片段的全部 54 个测试视频（600 帧，约 2 分钟一个，共约 1.8 小时）
python -B hdr-gaming-experiment/scoring/score.py returnal_clip_005

# 快速预览：每个视频只算前 10 帧（约 2 分钟全部算完）
python -B hdr-gaming-experiment/scoring/score.py returnal_clip_005 --frames 10

# 只重画图
python -B hdr-gaming-experiment/scoring/score.py returnal_clip_005 --plot-only
```

参数是片段名。`--targets`、`--heights`、`--bitrates` 可以只算一部分条件。每算完一个就写进 CSV；中断后再运行，会跳过已经用同样帧数算过的条件，`--overwrite` 强制重算。

## 比较什么

每个测试视频和同一目标的参考视频比较：

| 测试视频 | 参考视频 |
|---|---|
| `test_output/lanczos/{tonemap_then_compress,compress_then_tonemap}/hdr1000/` | `test_output/reference/<片段>_hdr1000.mp4` |
| `.../hdr400/` | `test_output/reference/<片段>_hdr400.mp4` |
| `.../sdr/` | `test_output/reference/<片段>_sdr.mp4` |

## 显示器模型

`display_models_asus.json`：同一台 ASUS PA32UCX、同一个房间，HDR 和 SDR 只差显示的格式和亮度，所以三个目标的分数可以直接比较。

| 参数 | `asus_pa32ucx_pq1000`（HDR1000 / HDR400） | `asus_pa32ucx_sdr100`（SDR） | 来源 |
|---|---|---|---|
| 颜色 | BT.2020 + PQ | BT.709 + sRGB | |
| 分辨率 | 3840×2160 | 同左 | |
| 每度像素数 | 60（约 0.63 m） | 同左 | 实验程序 `pix_per_deg = 60` |
| 尺寸 | 32 寸 | 同左 | `HDRgamingExperiment/profiles/display_profile_xyz-linear_pa32ucs.mat` |
| 峰值亮度 | 1000 nits | 100 nits（SDR 白色） | PQ1000 规格；SDR 白色是 tone map 的假设 |
| 黑色 | 0.001 nits | 同左 | 估计（mini-LED 局部调光） |
| 环境光 | 5 lux | 同左 | 估计（暗室） |

峰值、黑色、环境光和 SDR 白色还没有实测，有测量值后改这个文件即可。

## 输出

`scoring_output/`：

| 文件 | 内容 |
|---|---|
| `<片段>_cvvdp.csv` | 每行一个测试视频：路线、目标、分辨率、码率、帧数、JOD、显示器模型、文件路径 |
| `<片段>_jod_vs_bitrate.png` | JOD 随码率的变化；每个目标一张小图，实线先 TM，虚线先压缩，颜色是分辨率 |
| `<片段>_route_difference.png` | 先 TM 的 JOD 减先压缩的 JOD；0 线上方表示先 TM 更好，灰带是 ±0.2 JOD 的参考范围（不是严格阈值） |
| `<片段>_route_scatter.png` | 同一个比较的散点图：横轴先压缩，纵轴先 TM，对角线上方表示先 TM 更好；形状是码率 |
| `<片段>_peak_luminance.png` | JOD 随峰值亮度（SDR 100 / HDR400 / HDR1000）的变化；行是路线，列是码率 |

运行结束后终端还会打印每个目标的汇总表（先 TM / 先压缩 / 差值）。

解读时注意：HDR1000 和 HDR400 用同一套 tone map 曲线，主要只差峰值亮度；SDR 和 HDR 之间还有 tone map 曲线（ACES）、位深（8-bit）和颜色格式的不同。播放版用 CRF 12 保存，这一步本身约损失 0.4 JOD（HDR），参考和测试视频都经过这一步。
