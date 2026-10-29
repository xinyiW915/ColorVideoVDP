# Scoring

给 `playback/test_output/` 里的播放版打分，并画图。从 ColorVideoVDP 项目根目录运行，使用 cvvdp 环境。

| 脚本 | 做什么 |
|---|---|
| `score_cvvdp.py` | ColorVideoVDP（JOD）打分和画图 |
| `score_vmaf.py` | VMAF / PU21-VMAF 打分，画和 JOD 对比的图（见 [VMAF 对照](#vmaf-对照)） |
| `preview.py` | 打分前检查显示器模型 |

## 运行命令

```powershell
conda activate cvvdp

# 一个片段的全部 90 个测试视频（Lanczos 和 ESRGAN 各 36 个，2160p 共 18 个；600 帧，约 80 秒一个，共约 2 小时）
python -B hdr-gaming-experiment/scoring/score_cvvdp.py returnal_clip_005

# 可以一次给多个片段，逐个打分；多于一个片段时另外画所有片段并排的图
python -B hdr-gaming-experiment/scoring/score_cvvdp.py returnal_clip_005 horizon5_2_clip_005 spider_man_1_clip_001 the_last_of_us_2_clip_001

# 只算一种放大方式（2160p 仍然从 native/ 读）
python -B hdr-gaming-experiment/scoring/score_cvvdp.py returnal_clip_005 --upscalers lanczos

# 快速预览：每个视频只算前 10 帧（约 2 分钟全部算完）
python -B hdr-gaming-experiment/scoring/score_cvvdp.py returnal_clip_005 --frames 10

# 只重画图（4 个片段的单独图和并排图）
python -B hdr-gaming-experiment/scoring/score_cvvdp.py returnal_clip_005 horizon5_2_clip_005 spider_man_1_clip_001 the_last_of_us_2_clip_001 --plot-only
```

参数是片段名（一个或多个）。`--targets`、`--heights`、`--bitrates`、`--upscalers` 可以只算一部分条件。每算完一个就写进 CSV；中断后再运行，会跳过已经用同样帧数算过的条件，`--overwrite` 强制重算。`--test-output` 改读别的播放版文件夹（例如 `playback.py --output` 写出的测试结果），`--results` 改 CSV 和图的输出文件夹。

ESRGAN 播放版和打分都要用 GPU：不要和 `playback.py` 的 ESRGAN 同时跑（Windows 显存不够时会借用内存，慢很多倍）。

## 检查显示器模型（dm-preview）

打分前先确认 ColorVideoVDP 读到的视频经过显示器模型后亮度是对的：

```powershell
python -B hdr-gaming-experiment/scoring/preview.py returnal_clip_005
```

每个目标抽一个测试视频（默认先 TM、2160p、25 Mbps，`--route`、`--height`、`--mbps`、`--upscaler` 可改；2160p 从 `native/` 读），和 `score_cvvdp.py` 用同样的方式读入，用 ColorVideoVDP 的 `dm-preview` 把前 3 帧（`--frames`）存成 OpenEXR：绝对线性 BT.709 RGB，单位 cd/m²，上半是测试视频，下半是参考视频。文件在 `scoring/dm_preview/<片段>/`，已在 `.gitignore` 里，不进 git。

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
| `test_output/{lanczos,esrgan,native}/{tonemap_then_compress,compress_then_tonemap}/hdr1000/` | `test_output/reference/<片段>_hdr1000.mp4` |
| `.../hdr400/` | `test_output/reference/<片段>_hdr400.mp4` |
| `.../sdr/` | `test_output/reference/<片段>_sdr.mp4` |

720p 和 1080p 在 `lanczos/`、`esrgan/` 里各有一份。2160p 不放大，只在 `native/` 里有一份，只打一次分（CSV 里放大方式记为 `native`），两种放大方式的汇总表和图都用这个分数。

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
| `<片段>_cvvdp.csv` | 每行一个测试视频：放大方式（`lanczos` / `esrgan` / `native`）、路线、目标、分辨率、码率、帧数、JOD、显示器模型、文件路径 |
| `<片段>_jod_vs_bitrate.png` | JOD 随码率的变化；行是路线，列是目标 |
| `<片段>_peak_luminance.png` | JOD 随峰值亮度（SDR 100 / HDR400 / HDR1000）的变化；行是路线，列是码率 |
| `all_clips_jod_vs_bitrate.png` | 所有给出的片段并排：每列一个片段，每行一个路线 × 目标（先 TM 三行、先压缩三行），所有小图共用 JOD 轴，可以直接比较片段之间的高低（给多个片段时才画） |
| `all_clips_peak_luminance.png` | 同样并排：每列一个片段，每行一个路线 × 码率（先 TM 三行、先压缩三行），横轴是峰值亮度（给多个片段时才画） |
| `all_clips_route_scatter.png` | 两条路线的散点图，同样并排：每列一个片段，每行一个目标；横轴先压缩，纵轴先 TM，对角线上方表示先 TM 更好；形状是码率；所有子图的横轴和纵轴范围都相同（给多个片段时才画，不再单独画每个片段的散点图） |

两种放大方式画在同一张图里：颜色是分辨率，Lanczos-5 是实线、实心点，Real-ESRGAN 是虚线、空心点；2160p 不放大，画成一条绿色点线（散点图里是绿色实心点）。每个片段 2 张图。所有图的 JOD 轴固定为 5–10（`score_cvvdp.py` 里的 `JOD_RANGE`），散点图的横轴和纵轴也是这个范围，VMAF 图的 JOD 轴也一样；换片段、只跑一个片段，图都可以直接对比。只有某个 JOD 超出 5–10 时才会把范围放宽到能装下它（按 0.5 取整），并在终端打印提醒，这时的图和其他图不一致。运行结束后终端还会打印每个片段、每种放大方式、每个目标的汇总表（先 TM / 先压缩 / 差值）。

## VMAF 对照

`score_vmaf.py` 用 ffmpeg 的 libvmaf 给同样的测试视频打 VMAF 分（参考视频、2160p 的处理和 `score_cvvdp.py` 一样），再和 ColorVideoVDP 的 JOD 对比。有两种方式：

| | VMAF（默认） | PU21-VMAF（`--pu21`） |
|---|---|---|
| VMAF 读到的是什么 | 解码后的码值：HDR 是 10-bit PQ，SDR 是 8-bit sRGB（4:2:0） | 组里 metric_calibration 项目的做法（见下） |
| 模型 | `vmaf_4k_v0.6.1` 和 `vmaf_4k_v0.6.1neg` | `vmaf_b_v0.6.3`（和组里一样，没有 NEG） |
| 速度（600 帧） | 约 50 秒一个，一个片段约 75 分钟 | 约 1.5–2 分钟一个，一个片段约 2.5–3 小时 |
| 硬件 | 只用 CPU | 读视频和显示器模型用 GPU，不要和 `playback.py` 或 `score_cvvdp.py` 同时跑 |

PU21-VMAF 严格按组里 `metric_calibration` 项目的 `cpp_metrics/vmaf_metric.py` 计算：

1. 用 pycvvdp 读视频，显示器模型和 `score_cvvdp.py` 相同，取 `colorspace='display_encoded_01'`：HDR（PQ）是显示器线性 RGB 每个通道的 PU21 值除以 PU21(10000 cd/m²)，范围 0–1；SDR 不做 PU21，就是 sRGB 码值本身（pycvvdp 的规定）。
2. 用 BT.709 矩阵转成 YCbCr，10-bit limited range（Y 64–940），4:4:4（同项目 `utils/yuv_utils.py` 的 `YUVWriter`）。
3. 用 `vmaf_b_v0.6.3` 打分，取 `vmaf` 的平均值。

`vmaf_b_v0.6.3` 是 21 个模型的 bootstrap 集合，libvmaf 报告的 `vmaf` 就是其中第 0 个模型的分数（其余是 `vmaf_0001` 等和 `vmaf_bagging`）。这个 ffmpeg 没有内置它，也不能读模型集合，所以从 `vmaf_b_v0.6.3.json`（Netflix/vmaf 仓库 `model/` 里的原文件）取出第 0 个模型单独加载。用官方 `vmaf` 命令行（v3.2.1，组里的命令）在同样的 5 帧输入上核对过：两边的 `vmaf` 都是 39.462613。测试视频和参考视频相同时，这个模型给出约 99.3（不是 100）。和组里 numpy 版转换相比，送进 VMAF 的码值只有不到 0.01% 差 1（GPU/CPU 浮点误差）。`scoring/vmaf.exe`（如果有）只是用来这样核对的，打分不用它，也不进 git。

```powershell
# 可以一次给多个片段，逐个打分，最后画所有片段合并的图
python -B hdr-gaming-experiment/scoring/score_vmaf.py returnal_clip_005 horizon5_2_clip_005 spider_man_1_clip_001 the_last_of_us_2_clip_001 --pu21
python -B hdr-gaming-experiment/scoring/score_vmaf.py returnal_clip_005 horizon5_2_clip_005 spider_man_1_clip_001 the_last_of_us_2_clip_001

# 只重画图
python -B hdr-gaming-experiment/scoring/score_vmaf.py returnal_clip_005 horizon5_2_clip_005 spider_man_1_clip_001 the_last_of_us_2_clip_001 --pu21 --plot-only
```

参数和 `score_cvvdp.py` 相同，中断后再运行同样会跳过已经算过的视频。分数是所有帧的平均值；默认方式的 `vmaf_neg`（`vmaf_4k_v0.6.1neg`）不奖励锐化等增强，PU21-VMAF 的 CSV 里这一列是空的。

下表是默认方式的文件名；`--pu21` 时把文件名里的 `vmaf` 换成 `pu21_vmaf`（例如 `<片段>_pu21_vmaf.csv`、`all_clips_pu21_vmaf_vs_bitrate.png`），两种结果互不覆盖。

图和 `score_cvvdp.py` 的 all 图一模一样（同一个画图函数），只是 JOD 换成 VMAF，可以和右边一列的 JOD 图左右对比：

| 文件 | 内容 | 对应的 JOD 图 |
|---|---|---|
| `<片段>_vmaf.csv` | 每行一个测试视频的 VMAF 和 VMAF NEG | |
| `all_clips_vmaf_vs_bitrate.png` | VMAF 随码率的变化；每列一个片段，每行一个路线 × 目标 | `all_clips_jod_vs_bitrate.png` |
| `all_clips_vmaf_peak_luminance.png` | VMAF 随峰值亮度的变化；每列一个片段，每行一个路线 × 码率 | `all_clips_peak_luminance.png` |
| `all_clips_vmaf_route_scatter.png` | 两条路线的 VMAF 散点图；每列一个片段，每行一个目标 | `all_clips_route_scatter.png` |

VMAF 轴固定为 20–100（`score_vmaf.py` 里的 `VMAF_RANGE`），超出时才放宽到能装下（按 5 取整）并打印提醒。只给一个片段也会画（只有一列）。不需要 JOD 的 CSV 就能画图；有同样帧数的 `<片段>_cvvdp.csv` 时，终端还会打印每个片段和所有片段合起来的 VMAF–JOD 相关系数，以及两个比较的平均 ΔJOD、ΔVMAF（默认方式还有 ΔNEG）和两者符号相同的个数。

VMAF 是在 SDR 上训练的，只看亮度、不看颜色。两种方式下不同目标之间的 VMAF 都不能比较（JOD 可以）：默认方式 HDR 是 PQ 码值、SDR 是 sRGB 码值；PU21-VMAF 只有 HDR 做了 PU21，SDR 仍是 sRGB 码值。HDR1000 和 HDR400 的 PU21-VMAF 用同一个尺度。

解读时注意：HDR1000 和 HDR400 用同一套 tone map 曲线，主要只差峰值亮度；SDR 和 HDR 之间还有 tone map 曲线（ACES）、位深（8-bit）和颜色格式的不同。两条路线的 tone map 位置不同：先 TM 是服务器端 tone map，在缩小和压缩之前；先压缩是客户端或显示器 tone map，在解码和放大到 4K 之后（见 `playback/README.md`）。播放版用 x265 CRF 10 保存，参考和测试视频都经过这一步；只看前 30 帧时它相对无损版约为 9.92 JOD（HDR）、9.98 JOD（SDR），完整 600 帧还没测。
