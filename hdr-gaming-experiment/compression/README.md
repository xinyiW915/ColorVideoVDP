# Compression

从 ColorVideoVDP 项目根目录运行，使用 cvvdp 环境。

## 运行命令

```powershell
conda activate cvvdp

# 先压缩，再 TM：压原片片段
python -B hdr-gaming-experiment/compression/compress.py "C:\Users\xw506\Videos\OBS\HDR\clips\returnal\returnal_clip_005.mp4"

# 先 TM，再压缩：压 tone map 后的三个无损 MKV（一次可传多个文件）
python -B hdr-gaming-experiment/compression/compress.py `
  hdr-gaming-experiment/hdr_tone_mapping/tone_map_output/returnal_clip_005_hdr1000_calibrated.mkv `
  hdr-gaming-experiment/hdr_tone_mapping/tone_map_output/returnal_clip_005_hdr400_calibrated.mkv `
  hdr-gaming-experiment/hdr_tone_mapping/tone_map_output/returnal_clip_005_sdr100_aces.mkv

# 只压部分条件；覆盖已有结果
python -B hdr-gaming-experiment/compression/compress.py "C:\Users\xw506\Videos\OBS\HDR\clips\returnal\returnal_clip_005.mp4" --heights 1080 --bitrates 12 --overwrite
```

命令行参数是要压缩的视频文件的完整路径，可以一次传多个。输入须为 limited-range BT.2020/PQ（原片和 HDR tone map 输出）或 BT.709/sRGB（SDR tone map 输出），按颜色标签自动判断。`-B` 避免生成 Python 缓存。

`--max-frames` 默认 600，即只压前 600 帧（60 fps 下 10 秒），和 `hdr_tone_mapping/run.py` 的默认值一致，两边的文件逐帧对应。原因：`extract_clips.py` 用 stream copy 剪出的片段解码后是 601 帧，前 600 帧与原片逐帧一致，第 601 帧跳过了原片的 2 帧。

## 实验条件

| 参数 | 取值 | 命令行 |
|---|---|---|
| 分辨率 | 2160p（3840×2160）、1080p（1920×1080）、720p（1280×720） | `--heights` |
| 码率 | 5、12、25 Mbps | `--bitrates` |

每个输入得到 9 个文件。

## 编码设置

每个文件由两个 ffmpeg 进程经管道连接完成，不在磁盘上写原始 YUV：

```text
解码（前 600 帧） -> Lanczos-5 缩小 -> raw YUV 4:2:0 -> x265
```

编码命令按导师给的 SVT-AV1 配方：

```text
ffmpeg -y -f rawvideo -pixel_format yuv420p -video_size WIDTHxHEIGHT -framerate 60 \
  -i input.yuv -c:v libsvtav1 -preset 12 -pix_fmt yuv420p -b:v 20M \
  -svtav1-params 'pred-struct=1:rc=1:keyint=99999' -loglevel error output.mp4
```

| 配方 | 本项目 | 说明 |
|---|---|---|
| `libsvtav1` | `libx265` | 按原项目计划用 H.265 |
| `-preset 12` | `-preset fast`（`--preset`） | x265 没有对应档位，fast 近似游戏串流的硬件编码器 |
| `yuv420p` | HDR `yuv420p10le`；SDR `yuv420p` | HDR 需要 10-bit；SDR 和配方一样用 8-bit |
| `-b:v 20M` | `-b:v` 5 / 12 / 25M | 码率换成实验条件 |
| `pred-struct=1`（低延迟） | `bframes=0` | 只参考之前的帧 |
| `rc=1`（VBR） | 单遍 `-b:v`（x265 平均码率模式） | |
| `keyint=99999` | `keyint=99999:scenecut=0` | x265 默认在场景切换时插关键帧，`scenecut=0` 关掉，只有第一帧是关键帧 |
| 无 | 颜色标签 | HDR 为 BT.2020/PQ，SDR 为 BT.709/sRGB，limited range；写在原始输入上，码流随之带上标签。不用 hdr-opt，不写 HDR10 静态元数据 |

缩小：Lanczos-5（`flags=lanczos:param0=5`），直接作用在 YUV 码值上（HDR 为 PQ，SDR 为 sRGB）；2160p 不缩放。全程保持 YUV，不经过 RGB 转换。

## 输出与验证

输出文件夹按输入文件名自动决定：

```text
compression_output/
  compress_then_tonemap/          原片压缩，如 returnal_clip_005_1080p_12M.mp4
  tonemap_then_compress/
    hdr1000/                      输入 *_hdr1000_calibrated.mkv
    hdr400/                       输入 *_hdr400_calibrated.mkv
    sdr/                          输入 *_sdr100_aces.mkv，如 returnal_clip_005_sdr100_aces_1080p_12M.mp4
```

`--output` 改的是 `compression_output/` 这一层，子文件夹仍按上面规则生成。覆盖同名文件须显式加 `--overwrite`。

验证包括帧数（按 packet 计数）、分辨率、帧率、编码格式、像素格式、颜色标签，以及完整解码一遍；全部通过后才改成最终文件名，`.partial.mp4` 为未完成文件。每个文件完成后打印实际码率。开始前会完整解码一遍输入来数帧，无损 4K 输入约需 1 分钟。
