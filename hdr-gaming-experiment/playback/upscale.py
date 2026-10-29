"""Client-side upscaling for the playback videos: Lanczos-5 and Real-ESRGAN, on RGB.

A streaming client decodes the stream, converts it to RGB and scales it to the
display. Here every frame is decoded by ffmpeg to full-range RGB (RGBReader),
upscaled to 3840x2160 on the GPU (Upscaler) and handed to PlaybackWriter, which
converts it back to limited-range YUV 4:2:0 for the playback encode.

  lanczos  Lanczos-5 resampling (x2 from 1080p, x3 from 720p)
  esrgan   1080p: RealESRGAN_x2plus (2160p directly)
           720p:  RealESRGAN_x4plus (2880p), then Lanczos-5 down to 2160p, as
                  Real-ESRGAN's own --outscale resizes with Lanczos

Lanczos runs in torch rather than ffmpeg: swscale stores the step between output
pixels in 16.16 fixed point, so 720p -> 2160p (1/3) and 2880p -> 2160p (4/3) drift by
up to ~0.02 source pixels across the frame; here every position is exact. Kernel:
sinc(x) sinc(x/5) for |x| < 5, centre-aligned, widened by the scale factor when
downscaling, edge pixels replicated, weights normalised to sum 1 (as swscale).

Real-ESRGAN takes frames as full-range RGB code values in [0, 1]: BT.2020/PQ for HDR,
BT.709/sRGB for SDR (it was trained on sRGB images). Whole frames, fp16,
channels_last (Real-ESRGAN's default half precision, no tiles). Weights are
downloaded to playback/models/ on first use and checked against their SHA-256.

RRDBNet, RRDB and ResidualDenseBlock are copied from BasicSR
(basicsr/archs/rrdbnet_arch.py, https://github.com/XPixelGroup/BasicSR), licensed under
the Apache License 2.0, Copyright 2018-2022 BasicSR Authors. Changes: the registry
decorator, weight initialisation and scale-1 path were removed, make_layer is replaced
by nn.Sequential and BasicSR's pixel_unshuffle by torch's (same channel order).
Parameter names are unchanged, so the official weights load with strict=True.
"""

import functools
import math
import sys
from pathlib import Path

import ffmpeg
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
from compression.compress import FORMATS, RESOLUTIONS  # noqa: E402
from hdr_tone_mapping.color_encoding import pq_encode  # noqa: E402
from hdr_tone_mapping.video_io import HDRExperimentWriter, video_reader  # noqa: E402

SIZE = RESOLUTIONS[2160]
LANCZOS_A = 5
UPSCALERS = ("lanczos", "esrgan")
MODEL_DIR = HERE / "models"
RELEASES = "https://github.com/xinntao/Real-ESRGAN/releases/download"
# Name -> (scale, URL, SHA-256, GiB of free VRAM needed: measured whole-frame peak
# + margin).
MODELS = {
    "x4plus": (
        4,
        f"{RELEASES}/v0.1.0/RealESRGAN_x4plus.pth",
        "4fa0d38905f75ac06eb49a7951b426670021be3018265fd191d2125df9d682f1",
        14,
    ),
    "x2plus": (
        2,
        f"{RELEASES}/v0.2.1/RealESRGAN_x2plus.pth",
        "49fafd45f8fd7aa8d31ab2a22d14d91b536c34494a5cfe31eb5d89c2fa266abb",
        6,
    ),
}
# Compressed height -> model.
MODEL_FOR_HEIGHT = {720: "x4plus", 1080: "x2plus"}


@functools.lru_cache(maxsize=8)
def lanczos_matrix(n_in, n_out, device):
    """(n_out, n_in) float32 matrix resampling one axis with Lanczos-5."""
    scale = n_out / n_in
    stretch = min(scale, 1.0)
    support = LANCZOS_A / stretch
    centre = (torch.arange(n_out, dtype=torch.float64) + 0.5) / scale - 0.5
    taps = torch.floor(centre - support).long()[:, None] + torch.arange(
        2 * math.ceil(support) + 2
    )
    x = (taps - centre[:, None]) * stretch
    w = torch.where(
        x.abs() < LANCZOS_A,
        torch.sinc(x) * torch.sinc(x / LANCZOS_A),
        torch.zeros_like(x),
    )
    m = torch.zeros(n_out, n_in, dtype=torch.float64)
    m.scatter_add_(1, taps.clamp(0, n_in - 1), w)
    return (m / m.sum(1, keepdim=True)).float().to(device)


def lanczos_resize(rgb, width, height):
    """(H, W, 3) float tensor -> (height, width, 3), Lanczos-5, not clipped."""
    h, w, _ = rgb.shape
    x = lanczos_matrix(h, height, rgb.device) @ rgb.float().reshape(h, w * 3)
    x = x.reshape(height, w, 3).permute(0, 2, 1).reshape(height * 3, w)
    x = x @ lanczos_matrix(w, width, rgb.device).T
    return x.reshape(height, 3, width).permute(0, 2, 1)


class ResidualDenseBlock(nn.Module):
    def __init__(self, num_feat=64, num_grow_ch=32):
        super().__init__()
        self.conv1 = nn.Conv2d(num_feat, num_grow_ch, 3, 1, 1)
        self.conv2 = nn.Conv2d(num_feat + num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv3 = nn.Conv2d(num_feat + 2 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv4 = nn.Conv2d(num_feat + 3 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv5 = nn.Conv2d(num_feat + 4 * num_grow_ch, num_feat, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x


class RRDB(nn.Module):
    def __init__(self, num_feat, num_grow_ch=32):
        super().__init__()
        self.rdb1 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb2 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb3 = ResidualDenseBlock(num_feat, num_grow_ch)

    def forward(self, x):
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        return out * 0.2 + x


class RRDBNet(nn.Module):
    """ESRGAN generator; scale 2 pixel-unshuffles the input first (x2plus)."""

    def __init__(
        self, num_in_ch, num_out_ch, scale=4, num_feat=64, num_block=23, num_grow_ch=32
    ):
        super().__init__()
        self.scale = scale
        if scale == 2:
            num_in_ch = num_in_ch * 4
        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.body = nn.Sequential(
            *(RRDB(num_feat, num_grow_ch) for _ in range(num_block))
        )
        self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        feat = F.pixel_unshuffle(x, 2) if self.scale == 2 else x
        feat = self.conv_first(feat)
        body_feat = self.conv_body(self.body(feat))
        feat = feat + body_feat
        feat = self.lrelu(self.conv_up1(F.interpolate(feat, scale_factor=2)))
        feat = self.lrelu(self.conv_up2(F.interpolate(feat, scale_factor=2)))
        return self.conv_last(self.lrelu(self.conv_hr(feat)))


class ESRGAN:
    """One Real-ESRGAN model on the GPU: (H, W, 3) RGB in [0, 1] -> (sH, sW, 3) fp32,
    not clipped."""

    def __init__(self, name, device="cuda"):
        scale, url, sha256, vram_gib = MODELS[name]
        free = torch.cuda.mem_get_info(device)[0] / 2**30
        if free < vram_gib:
            # Windows does not raise out-of-memory here: it spills into system RAM and
            # runs many times slower, e.g. while score.py is using the GPU.
            raise RuntimeError(
                f"{name} needs ~{vram_gib} GiB of free GPU memory, {free:.1f} GiB "
                "free; stop other GPU jobs (e.g. score.py) first"
            )
        path = MODEL_DIR / Path(url).name
        if not path.is_file():
            MODEL_DIR.mkdir(exist_ok=True)
            torch.hub.download_url_to_file(url, str(path), hash_prefix=sha256)
        weights = torch.load(path, map_location="cpu", weights_only=True)["params_ema"]
        net = RRDBNet(3, 3, scale=scale)
        net.load_state_dict(weights, strict=True)
        self.net = net.eval().to(
            device, torch.float16, memory_format=torch.channels_last
        )
        self.name, self.scale = name, scale
        torch.backends.cudnn.benchmark = True

    @torch.inference_mode()
    def __call__(self, rgb):
        x = rgb.permute(2, 0, 1)[None]
        x = x.to(torch.float16, memory_format=torch.channels_last)
        return self.net(x)[0].float().permute(1, 2, 0)


class Upscaler:
    """(H, W, 3) RGB in [0, 1] from one compressed height -> (2160, 3840, 3) fp32,
    not clipped."""

    def __init__(self, method, height, device="cuda"):
        self.method, self.height = method, height
        self.esrgan = None
        if method == "esrgan":
            self.esrgan = ESRGAN(MODEL_FOR_HEIGHT[height], device)

    @torch.inference_mode()
    def __call__(self, rgb):
        if self.esrgan is not None:
            rgb = self.esrgan(rgb)
        if rgb.shape[:2] != (SIZE[1], SIZE[0]):
            rgb = lanczos_resize(rgb, *SIZE)
        return rgb


class RGBReader(video_reader):
    """video_reader for limited-range YUV -> full-range RGB48 with an explicit matrix
    (BT.2020 HDR / BT.709 SDR) and accurate_rnd+full_chroma_int: the default swscale
    flags make 8-bit SDR about 0.7 code values too dark."""

    def __init__(self, vidfile, hdr, frames=-1):
        self.matrix = "bt2020" if hdr else "bt709"
        super().__init__(vidfile, frames)

    def _setup_ffmpeg(self, vidfile, resize_fn, resize_height, resize_width, verbose):
        self.dtype, self.bpp = np.uint16, 6
        self.frame_bytes = self.width * self.height * self.bpp
        self.process = (
            ffmpeg.input(vidfile)
            .video.filter(
                "scale",
                in_color_matrix=self.matrix,
                out_color_matrix=self.matrix,
                in_range="tv",
                out_range="pc",
                flags="accurate_rnd+full_chroma_int",
            )
            .output(
                "pipe:", format="rawvideo", pix_fmt="rgb48le", fps_mode="passthrough"
            )
            .global_args("-loglevel", "error")
            .run_async(pipe_stdout=True)
        )


class PlaybackWriter(HDRExperimentWriter):
    """4K RGB tensor in, rounded on the GPU (as HDRExperimentWriter rounds: 16-bit
    HDR, 8-bit SDR) and clipped to the target peak, then limited-range YUV 4:2:0 and
    the given encode options. peak_nits is the HDR target peak; SDR clips to white.

    The peak is the highest 16-bit code at or below PQ(peak_nits): the display being
    simulated cannot exceed it, and on PQ input Real-ESRGAN pushes bright edges such
    as HUD text up to code 1.0 (10000 nits)."""

    def __init__(self, fname, fps, hdr_mode, options, peak_nits=None):
        super().__init__(fname, fps, hdr_mode)
        self.options = options
        if hdr_mode:
            pq = pq_encode(torch.tensor(float(peak_nits), dtype=torch.float64))
            self.peak_code = math.floor(float(pq) * 65535)
        else:
            self.peak_code = 255

    def write_frame_rgb(self, rgb):
        if tuple(rgb.shape[:2]) != (SIZE[1], SIZE[0]):
            raise ValueError(f"Expected a {SIZE[0]}x{SIZE[1]} frame, got {rgb.shape}")
        if self.process is None:
            self.process = self._start_ffmpeg(*rgb.shape[:2])
        scale = 65535 if self.hdr_mode else 255
        frame = torch.round(rgb * scale).clamp(0, self.peak_code)
        frame = frame.to(torch.uint16 if self.hdr_mode else torch.uint8)
        self.process.stdin.write(frame.cpu().numpy().tobytes())

    def _start_ffmpeg(self, height, width):
        fmt = FORMATS["hdr" if self.hdr_mode else "sdr"]
        return (
            ffmpeg.input(
                "pipe:",
                format="rawvideo",
                pix_fmt="rgb48le" if self.hdr_mode else "rgb24",
                s=f"{width}x{height}",
                r=self.fps,
            )
            .filter(
                "scale",
                in_range="pc",
                out_range="tv",
                out_color_matrix="bt2020" if self.hdr_mode else "bt709",
            )
            .filter("format", fmt["pix_fmt"])
            .filter(
                "setparams",
                range="limited",
                color_primaries=fmt["primaries"],
                color_trc=fmt["trc"],
                colorspace=fmt["matrix"],
            )
            .output(self.fname, **self.options)
            .overwrite_output()
            .global_args("-hide_banner", "-loglevel", "error")
            .run_async(pipe_stdin=True)
        )
