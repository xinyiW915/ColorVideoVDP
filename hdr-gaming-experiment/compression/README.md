# Compression

从 ColorVideoVDP 项目根目录运行，使用 cvvdp 环境。

## 运行命令

```powershell
conda activate cvvdp

# 路线 2（后 TM）：原片片段的前 600 帧，3 个分辨率 × 3 个码率
python -B hdr-gaming-experiment/compression/compress.py "C:\Users\xw506\Videos\OBS\HDR\clips\returnal\returnal_clip_005.mp4"

# 只压部分条件；覆盖已有结果
python -B hdr-gaming-experiment/compression/compress.py "C:\Users\xw506\Videos\OBS\HDR\clips\returnal\returnal_clip_005.mp4" --heights 1080 --bitrates 12 --overwrite
```

命令行参数是要压缩的视频文件的完整路径（一次一个视频）。输入须为 limited-range BT.2020/PQ（OBS HDR 录制符合）。`-B` 避免生成 Python 缓存。

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
解码 -> Lanczos-5 缩小 -> raw yuv420p10le -> x265
```

- 缩小：Lanczos-5（`flags=lanczos:param0=5`），直接作用在 limited-range BT.2020/PQ 的 YUV 码值上；2160p 不缩放。
- 编码器：libx265（H.265），10-bit 4:2:0，单遍 `-b:v` 目标码率。
- 低延迟：`bframes=0:keyint=99999:scenecut=0`，对应参考配方的 pred-struct=1（无 B 帧）、只有第一帧是关键帧。
- preset `fast`（`--preset`）：x265 没有 SVT-AV1 preset 12 的对应档位，fast 用来近似游戏串流的硬件编码器。
- 颜色标签：BT.2020/PQ、limited range，写在原始输入上，码流随之带上标签。不用 hdr-opt，不写 HDR10 静态元数据。

## 输出与验证

`compression_output/original/`：`<输入文件名>_<分辨率>p_<码率>M.mp4`，如 `returnal_clip_005_1080p_12M.mp4`。覆盖同名文件须显式加 `--overwrite`。

验证包括帧数（按 packet 计数）、分辨率、帧率、编码格式、像素格式、颜色标签，以及完整解码一遍；全部通过后才改成最终文件名，`.partial.mp4` 为未完成文件。每个文件完成后打印实际码率。

## 目前范围

现在只做路线 2（先压缩，再 tone map）的压缩：输入是原片片段。路线 1（先 tone map，再压缩）的输入是 `hdr_tone_mapping/tone_map_output/` 里的无损 MKV，其中 SDR 是 BT.709/sRGB、8-bit，压缩时也用 8-bit；`compress.py` 目前只接受 BT.2020/PQ 输入，做路线 1 时再扩展。
