"""Video I/O: ColorVideoVDP's video_reader and VideoWriter with their ffmpeg commands replaced.

Frames are still read by video_reader.get_frame()/unpack(). The commands differ in
one respect that matters: VideoWriter never sets the RGB->YUV matrix, so ffmpeg
falls back to BT.601 while the HDR stream is tagged BT.2020 (and SDR is left
untagged). Here the matrix, range and colour tags are explicit on both sides.

The writer also replaces VideoWriter.write_frame_rgb(): VideoWriter truncates
frames to integers, so 8-bit SDR comes out about half a code value too dark;
this writer rounds. Outputs are lossless FFV1 (10-bit HDR, 8-bit SDR) so that
tone mapping adds no compression before the compression experiment.
"""

import subprocess
import sys
from pathlib import Path

import ffmpeg
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from pycvvdp.video_source_file import video_reader  # noqa: E402
from pycvvdp.video_writer import VideoWriter  # noqa: E402


class HDRReader(video_reader):
    """video_reader with an explicit limited-range BT.2020 YUV -> RGB48 ffmpeg command."""

    def _setup_ffmpeg(self, vidfile, resize_fn, resize_height, resize_width, verbose):
        self.dtype, self.bpp = np.uint16, 6
        self.frame_bytes = self.width * self.height * self.bpp
        self.process = (
            ffmpeg.input(vidfile)
            .video.filter(
                "scale",
                in_color_matrix="bt2020",
                out_color_matrix="bt2020",
                in_range="tv",
                out_range="pc",
            )
            .output(
                "pipe:", format="rawvideo", pix_fmt="rgb48le", fps_mode="passthrough"
            )
            .global_args("-loglevel", "error")
            .run_async(pipe_stdout=True)
        )


class HDRExperimentWriter(VideoWriter):
    """VideoWriter with its ffmpeg command and frame packing replaced: rounded RGB in
    (16-bit HDR / 8-bit SDR), explicit RGB -> YUV matrix (BT.2020 HDR / BT.709 SDR)
    and colour tags, lossless FFV1 4:2:0 out (10-bit HDR / 8-bit SDR; write to .mkv)."""

    def __init__(self, fname, fps, hdr_mode):
        super().__init__(str(fname), fps=fps, hdr_mode=hdr_mode)

    def write_frame_rgb(self, rgb):
        if self.process is None:
            self.process = self._start_ffmpeg(*rgb.shape[:2])
        if self.hdr_mode:
            frame = np.round(rgb * 65535).astype(np.uint16)
        else:
            frame = np.round(rgb * 255).astype(np.uint8)
        self.process.stdin.write(frame.tobytes())

    def _start_ffmpeg(self, height, width):
        matrix = "bt2020" if self.hdr_mode else "bt709"
        primaries = "bt2020" if self.hdr_mode else "bt709"
        transfer = "smpte2084" if self.hdr_mode else "iec61966-2-1"
        colorspace = "bt2020nc" if self.hdr_mode else "bt709"
        options = {
            "vcodec": "ffv1",
            "level": 3,
            "slices": 24,
            "pix_fmt": "yuv420p10le" if self.hdr_mode else "yuv420p",
            "color_range": "tv",
            "color_primaries": primaries,
            "color_trc": transfer,
            "colorspace": colorspace,
        }
        return (
            ffmpeg.input(
                "pipe:",
                format="rawvideo",
                pix_fmt="rgb48le" if self.hdr_mode else "rgb24",
                s=f"{width}x{height}",
                r=self.fps,
            )
            .filter("scale", in_range="pc", out_range="tv", out_color_matrix=matrix)
            .filter(
                "setparams",
                range="limited",
                color_primaries=primaries,
                color_trc=transfer,
                colorspace=colorspace,
            )
            .output(self.fname, **options)
            .overwrite_output()
            .global_args("-hide_banner", "-loglevel", "error")
            .run_async(pipe_stdin=True)
        )


def validate_decode(path):
    """Decode every frame of the output, stopping on the first decoder error."""
    result = subprocess.run(
        ["ffmpeg", "-v", "error", "-xerror", "-i", str(path), "-map", "0:v:0", "-f", "null", "-"],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode:
        raise RuntimeError(f"Full decode failed: {path}\n{result.stderr}")
    return {"full_decode_passed": True}
