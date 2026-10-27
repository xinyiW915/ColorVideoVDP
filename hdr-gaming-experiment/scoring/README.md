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

## 检查显示器模型（dm-preview）

打分前先确认 ColorVideoVDP 读到的视频经过显示器模型后亮度是对的：

```powershell
python -B hdr-gaming-experiment/scoring/preview.py returnal_clip_005
```

每个目标抽一个测试视频（默认先 TM、2160p、25 Mbps，`--route`、`--height`、`--mbps` 可改），和 `score.py` 用同样的方式读入，用 ColorVideoVDP 的 `dm-preview` 把前 3 帧（`--frames`）存成 OpenEXR：绝对线性 BT.709 RGB，单位 cd/m²，上半是测试视频，下半是参考视频。文件在 `scoring/dm_preview/<片段>/`，已在 `.gitignore` 里，不进 git。

终端会打印测试和参考视频的亮度（最小、中位数、99.9%、最大），以及显示器模型能输出的范围：

| | 含义 | HDR1000 | HDR400 | SDR |
|---|---|---|---|---|
| floor | 黑位 + 环境光反射（PQ 先在 0.005 nits 截断） | ≈ 0.014 | ≈ 0.014 | ≈ 0.009 |
| ceiling | 目标峰值 + 同样的偏移 | ≈ 1000 | ≈ 400 | ≈ 100 |

以下情况会打印 `WARNING`：最小值低于 floor、超过 0.1% 的像素比 ceiling 高出 10% 以上（传输函数或范围错了会让整幅画面按比例偏亮）、分辨率和显示器不一致、测试和参考视频中位数差超过 10%。像素只是略高于 ceiling 时只打印 `note`，这是编码造成的，真实显示器也会这样显示：

- 大片区域被 tone map 压在目标峰值上时，编码噪声和 10-bit PQ 的量化（400 nits 附近每级约 1%）会让它们解码后在峰值上下几个百分点内浮动。例如 Spider-Man HDR400 约 0.8% 的像素（天空）在 400 附近，其中约 0.3% 超过 408，但几乎没有超过 440 的。
- 4:2:0 色度子采样和量化在高光边缘造成零星过冲，例如 Returnal HDR400 约 0.002% 的像素达到 470–690 nits。参考视频峰值远低于 ceiling 也只是提示（这几帧可能没有高光）。SDR 画面很暗时，pycvvdp 会打印 `The mean color value is less than 1`，这是它的经验性提醒，只要白色约 100 nits 就可以忽略。EXR 可以用 [tev](https://github.com/Tom94/tev) 打开，鼠标悬停可看到每个像素的 nits。

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
