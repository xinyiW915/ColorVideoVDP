# Simple spline

在项目根目录、cvvdp 环境中运行：

```powershell
# 完整视频
python -B hdr_tone_mapping/simple_spline/run.py --device cuda

# 前 12 帧测试；覆盖已有结果
python -B hdr_tone_mapping/simple_spline/run.py --device cuda --max-frames 12 --overwrite

# 只画曲线
python -B hdr_tone_mapping/simple_spline/run.py --device cuda --plot-only
```

默认输入 `input05_clip_002.mp4`；可用 `--input "视频路径"` 更换。未指定帧数则处理到视频结束。

输出统一保存到本目录 `tone_map_output/`：HDR1000、HDR400、SDR ACES Filmic 三个视频、曲线 PNG 和处理报告。重复覆盖需加 `--overwrite`。

- `run.py`：运行入口、画图、视频转换和报告。
- `tone_mapping.py`：曲线和颜色转换。
- `frame_mapping.py`：相同曲线的 CUDA 逐帧计算。
- `../video_io.py`：共用读写与完整解码验证。

HDR 仅映射亮度 Y：1000/400-nit 曲线分别在 500/200 nits 以下保持 1:1，再压缩高亮。SDR统一对亮度Y应用ACES，再等比例缩放RGB、转BT.709和sRGB；曝光尺度暂为100，白色按100 nits解释。

该版默认自动选择 CUDA，可用 `--device cpu` 回退，编码设置和实验限制见上级 README。SDR与simulation共用相同算子。


