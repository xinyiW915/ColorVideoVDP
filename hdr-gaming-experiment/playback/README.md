# Playback

生成给被试看、也用来给 ColorVideoVDP 打分的最终视频（播放版），放在 `playback/test_output/`。从 ColorVideoVDP 项目根目录运行，使用 cvvdp 环境。

## 运行命令

```powershell
conda activate cvvdp

# 一个片段的全部播放版：3 个参考 + 两条路线各 27 个测试视频
python -B hdr-gaming-experiment/playback/playback.py returnal_clip_005

# 只做部分；覆盖已有结果
python -B hdr-gaming-experiment/playback/playback.py returnal_clip_005 --parts reference --overwrite
python -B hdr-gaming-experiment/playback/playback.py returnal_clip_005 --parts compress_then_tonemap --heights 720 --bitrates 5
```

参数是片段名，即原片片段的文件名（不含扩展名）。输入从前两步的输出文件夹里按名字找，缺任何一个会先报错：

- `hdr_tone_mapping/tone_map_output/<片段>_{hdr1000_calibrated,hdr400_calibrated,sdr100_aces}.mkv`
- `compression/compression_output/tonemap_then_compress/{hdr1000,hdr400,sdr}/<片段>_<目标>_<分辨率>p_<码率>M.mp4`
- `compression/compression_output/compress_then_tonemap/<片段>_<分辨率>p_<码率>M.mp4`

## 流程

| 输出 | 处理 |
|---|---|
| 参考视频 | tone map 无损 MKV → 播放编码 |
| 先 TM，再压缩 | 压缩 MP4 → Lanczos-5 放大到 4K → 播放编码 |
| 先压缩，再 TM | 压缩 MP4 → tone map（HDR1000 / HDR400 / SDR）→ Lanczos-5 放大到 4K → 播放编码 |

- 放大：Lanczos-5（`flags=lanczos:param0=5`），作用在 limited-range YUV 4:2:0 码值上，和 `compress.py` 的缩小一致；2160p 不放大。
- 先压缩再 TM：tone map 用 `hdr_tone_mapping/run.py` 的同一套映射（`build_mappers`）和同样的取整，先在压缩分辨率下转成 YUV 4:2:0，再放大。这样两条路线放大的都是同一种画面，唯一区别是 tone map 在压缩前还是后。
- 放大、tone map 的中间结果不写硬盘，算完直接进播放编码。
- Lanczos 放大在边缘附近会有轻微过冲，个别像素可能略超目标峰值；两条路线都在 tone map 之后放大，影响相同，目前不另做截断。

## 播放编码

| 参数 | 取值 | 命令行 |
|---|---|---|
| 编码器 | libx265，preset fast | `--preset` |
| 画质 | CRF 12（不是实验条件，只为原样保存画面） | `--crf` |
| 帧结构 | `bframes=0:keyint=99999:scenecut=0`，和 `compress.py` 一致 | |
| 帧数 | 600（60 fps 下 10 秒） | `--frames` |
| HDR | BT.2020/PQ，10-bit 4:2:0，limited range | |
| SDR | BT.709/sRGB，8-bit 4:2:0，limited range | |
| 分辨率 | 3840×2160 | |

不写 HDR10 静态元数据：实验程序用 `PsychHDR('HDRMetadata', win, 0, 0, 0)` 统一设置。之前测得 CRF 12 保存会让 HDR 画面比无损版低约 0.4 JOD；参考和测试视频都经过这一步，处理一致。SDR 视频在 Psychtoolbox HDR 窗口里的亮度还需在实验电脑上实测。

## 输出

```text
test_output/
  reference/                      <片段>_{hdr1000,hdr400,sdr}.mp4
  lanczos/
    tonemap_then_compress/
      hdr1000/ hdr400/ sdr/       <片段>_<目标>_<分辨率>p_<码率>M.mp4，各 9 个
    compress_then_tonemap/
      hdr1000/ hdr400/ sdr/       同上，各 9 个
```

`lanczos/` 是放大方式；以后加其他放大方式时并列放在 `test_output/` 下。覆盖已有文件须显式加 `--overwrite`。

验证包括帧数（按 packet 计数）、分辨率、编码格式、像素格式、颜色标签，以及完整解码一遍；全部通过后才改成最终文件名，`.partial.mp4` 为未完成文件。每个视频完成后打印实际码率。
