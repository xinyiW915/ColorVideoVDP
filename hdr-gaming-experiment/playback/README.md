# Playback

生成给被试看、也用来给 ColorVideoVDP 打分的最终视频（播放版），放在 `playback/test_output/`。这一步模拟串流的客户端：解码压缩视频、放大到 4K，在先压缩再 TM 的路线里再做 tone map。从 ColorVideoVDP 项目根目录运行，使用 cvvdp 环境。

## 运行命令

```powershell
conda activate cvvdp

# 一个片段的全部播放版：3 个参考 + 18 个 2160p + Lanczos 和 ESRGAN 各 36 个，共 93 个（约 3.5 小时，大部分是 ESRGAN）
python -B hdr-gaming-experiment/playback/playback.py returnal_clip_005

# 只做部分；覆盖已有结果
python -B hdr-gaming-experiment/playback/playback.py returnal_clip_005 --parts reference --overwrite
python -B hdr-gaming-experiment/playback/playback.py returnal_clip_005 --upscalers lanczos --heights 720 --bitrates 5

# 快速检查：每个视频 60 帧，写到别的文件夹，再用 score_cvvdp.py 打分
python -B hdr-gaming-experiment/playback/playback.py returnal_clip_005 --frames 60 --bitrates 12 --output hdr-gaming-experiment/playback/test_output/_check
python -B hdr-gaming-experiment/scoring/score_cvvdp.py returnal_clip_005 --frames 60 --bitrates 12 --test-output hdr-gaming-experiment/playback/test_output/_check --results <另一个文件夹>
```

参数是片段名，即原片片段的文件名（不含扩展名）。输入从前两步的输出文件夹里按名字找，缺任何一个会先报错：

- `hdr_tone_mapping/tone_map_output/<片段>_{hdr1000_calibrated,hdr400_calibrated,sdr100_aces}.mkv`
- `compression/compression_output/tonemap_then_compress/{hdr1000,hdr400,sdr}/<片段>_<目标>_<分辨率>p_<码率>M.mp4`
- `compression/compression_output/compress_then_tonemap/<片段>_<分辨率>p_<码率>M.mp4`

ESRGAN 要用约 12 GB 显存，不要和 `score_cvvdp.py` 同时跑：空闲显存不够时程序会直接报错。Windows 显存不够时不会报 OOM，而是借用内存，速度会慢很多倍。

## 两条路线

服务器端（`hdr_tone_mapping/run.py` 和 `compression/compress.py`）先做好 tone map 和实验压缩，这里是客户端：

```text
参考视频：  [ffmpeg] 解码无损 MKV → 播放编码 → reference/

先 TM，再压缩（tonemap_then_compress，服务器端 tone map）：
  [ffmpeg] 解码 → RGB48
  [torch]  放大到 4K → 截到目标峰值 → 取整（HDR 16-bit，SDR 8-bit）
  [ffmpeg] RGB → YUV 4:2:0 → 播放编码

先压缩，再 TM（compress_then_tonemap，客户端或显示器 tone map）：
  [ffmpeg] 解码 HDR → RGB48
  [torch]  放大到 4K（只做一次）→ 截到 [0,1] → PQ 解码 → tone map 到三个目标（在 4K 上）→ 截到 [0,1] → 取整
  [ffmpeg] 三个写入进程：RGB → YUV 4:2:0 → 播放编码

2160p 不放大：先 TM 的直接转码（不经过 RGB），先压缩的在 4K 上 tone map，都写到 native/
```

- 先压缩再 TM 的 tone map 放在放大之后，对应显示器（或客户端软件）在最终画面上 tone map 的情况；tone map 用 `run.py` 的同一套映射（`build_mappers`）。
- 解码统一用 `accurate_rnd+full_chroma_int`（HDR BT.2020，SDR BT.709），避免 8-bit SDR 偏暗。
- 截断：先 TM 路线放大后截到不超过目标峰值（HDR1000 / HDR400）的最大 16-bit 码值，SDR 截到白色，因为被模拟的显示器亮不过它的峰值；Lanczos 的过冲和 ESRGAN 推亮的边缘都在这里截掉。先压缩路线由 tone map 保证不超过峰值。送进编码器的 RGB 都不超过峰值；编码成 YUV 4:2:0 再解码后，颜色交界处会有极少数像素略超峰值，参考视频也一样。
- 放大、tone map 的中间结果不写硬盘，算完直接进播放编码。

## 放大方式

`upscale.py`，都在 RGB 上用 torch 在 GPU 上计算：

| 压缩分辨率 | `lanczos` | `esrgan` |
|---|---|---|
| 1080p | Lanczos-5 ×2 | `RealESRGAN_x2plus`，×2 直接到 2160p |
| 720p | Lanczos-5 ×3 | `RealESRGAN_x4plus` ×4 到 2880p，再用 Lanczos-5 缩小到 2160p（即 Real-ESRGAN 官方 `--outscale` 的做法） |
| 2160p | 不放大（`native/`） | 同左 |

**Lanczos-5**：`sinc(x)·sinc(x/5)`，|x| < 5，像素中心对齐，缩小时核按比例展宽，边缘像素复制，权重归一化，和 ffmpeg 的 `lanczos:param0=5` 相同。不用 ffmpeg 是因为它用 16.16 定点数表示采样步长：1080p ×2 和服务器端的 ÷2、÷3 是精确的，但 720p ×3（步长 1/3）和 2880p → 2160p（4/3）会逐像素累积误差，到画面边缘约偏 0.02 个原图像素。实测 torch 版和 ffmpeg 在 1080p ×2 上几乎一样（HDR PSNR 96 dB），720p ×3 平均只差约 0.15 个 10-bit 码值。

**Real-ESRGAN**：

- 两个分辨率用不同模型：实际的游戏串流也会按分辨率选择合适的放大方式，所以不把这当作混杂变量。
- 网络输入是 full-range RGB 码值 [0, 1]：HDR 为 BT.2020/PQ，SDR 为 BT.709/sRGB，不转线性光。网络在 SDR 上训练，用在 PQ 上时会把 UI 文字等亮边缘推到码值 1.0（10000 nits），所以先 TM 路线要截到峰值。
- 整帧计算，fp16，channels_last，不切块。
- RRDBNet 网络代码复制自 BasicSR（Apache 2.0，见 `upscale.py` 文件头），参数名不变，可以直接加载官方权重，不依赖 basicsr、realesrgan、spandrel 这些包。
- 权重第一次运行时自动下载到 `playback/models/`，并校验 SHA-256（这个文件夹不进 git）。
- 显存：x4plus 整帧约 11–12 GB，x2plus 约 4 GB。
- 速度：x4plus 720p 约 0.75 秒/帧，x2plus 1080p 约 0.4 秒/帧；一个片段 ESRGAN 共运行 24 次（先 TM 18 次，先压缩 6 次）。

## 播放编码

| 参数 | 取值 | 命令行 |
|---|---|---|
| 编码器 | libx265，preset fast | `--preset` |
| 画质 | CRF 10（不是实验条件，只为原样保存画面） | `--crf` |
| 帧结构 | `bframes=0:keyint=99999:scenecut=0`，和 `compress.py` 一致 | |
| 帧数 | 600（60 fps 下 10 秒） | `--frames` |
| HDR | BT.2020/PQ，10-bit 4:2:0，limited range | |
| SDR | BT.709/sRGB，8-bit 4:2:0，limited range | |
| 分辨率 | 3840×2160 | |

不写 HDR10 静态元数据：实验程序用 `PsychHDR('HDRMetadata', win, 0, 0, 0)` 统一设置。用前 30 帧和无损 MKV 比较（returnal_clip_005）：

| CRF | HDR1000 | HDR400 | SDR | 参考视频码率（HDR） |
|---|---|---|---|---|
| 12（旧版） | 9.890 | 9.895 | 9.963 | 188 Mbps |
| **10** | **9.918** | **9.921** | **9.977** | 250 Mbps |
| 8 | 9.934 | 9.938 | 9.986 | 325 Mbps |

完整 600 帧的结果还没测（旧版 CRF 12 在 600 帧上约 9.6）。CRF 12 已确认能在 Psychtoolbox 里播放；CRF 10 需要在实验电脑上用码率最高的 HDR1000 参考视频试播一次，确认不掉帧。也测过 SVT-AV1 作为播放编码：HDR 上 CRF 越低分数反而越低（逐帧画质跳动），所以没有采用。SDR 视频在 Psychtoolbox HDR 窗口里的亮度还需在实验电脑上实测。

## 输出

```text
test_output/
  reference/                      <片段>_{hdr1000,hdr400,sdr}.mp4
  native/                         2160p，两种放大方式共用
    tonemap_then_compress/
      hdr1000/ hdr400/ sdr/       <片段>_<目标>_2160p_<码率>M.mp4，各 3 个
    compress_then_tonemap/
      hdr1000/ hdr400/ sdr/       同上
  lanczos/                        720p 和 1080p
    tonemap_then_compress/
      hdr1000/ hdr400/ sdr/       <片段>_<目标>_<分辨率>p_<码率>M.mp4，各 6 个
    compress_then_tonemap/
      hdr1000/ hdr400/ sdr/       同上
  esrgan/                         同 lanczos/
```

一个片段约 14 GB（按 returnal_clip_005 前 60 帧估算）。覆盖已有文件须显式加 `--overwrite`。`test_output/` 下的完整路径最长约 230 个字符，Windows 默认上限是 260；片段名或 `--output` 太长时改名一步会失败。

验证包括帧数（按 packet 计数）、分辨率、编码格式、像素格式、颜色标签，以及完整解码一遍；全部通过后才改成最终文件名，`.partial.mp4` 为未完成文件。每个视频完成后打印实际码率。
