# Calibrated simulation

在项目根目录、cvvdp 环境中运行：

```powershell
# 完整视频，GPU 映射
python -B hdr_tone_mapping/calibrated_simulation/run_calibrated.py --device cuda

# 前 12 帧测试；覆盖已有结果
python -B hdr_tone_mapping/calibrated_simulation/run_calibrated.py --device cuda --max-frames 12 --overwrite

# 只画曲线
python -B hdr_tone_mapping/calibrated_simulation/run_calibrated.py --plot-only
```

默认输入 `input05_clip_002.mp4`；可用 `--input "视频路径"` 更换。未指定帧数则处理到视频结束。默认自动选择CUDA，`--device cpu` 可改用CPU。

输出统一保存到本目录 `tone_map_output/`：HDR1000、HDR400、SDR ACES Filmic 三个视频、两张曲线 PNG、校准参数和处理报告。重复覆盖需加 `--overwrite`。

- `run_calibrated.py`：运行入口、画图、转换和报告。
- `white_response.py`：读取 white CSV，拟合响应、调整尺度和修正暗部。
- `frame_mapping.py`：CUDA 逐帧亮度映射及颜色转换。
- `../video_io.py`：共用读写与完整解码验证。

测量来源：`asushdr_display/resp_asushdr_pq1000_white.csv`。缩放实测响应形成400/1000-nit模拟，保留高亮走势；HDR400在4–40 nits从1:1平滑接回原曲线，HDR1000对应10–100 nits。更低亮度保持1:1。

SDR在亮度Y上应用ACES，再等比例缩放RGB；曝光尺度暂为100，SDR白色按100 nits解释。这里是基于测量形状的模拟，不是显示器逆校准。GPU用于映射，CPU用于编码；详细参数与实验限制见上级README。

