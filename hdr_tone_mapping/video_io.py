"""Shared experiment video I/O; helper module, not a conversion entry point."""
from pathlib import Path
import sys
import struct
import subprocess
import ffmpeg
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'pycvvdp'))
from pycvvdp.video_source_file import video_reader
from pycvvdp.video_writer import VideoWriter


class HDRReader(video_reader):
    """Use the project reader with explicit limited-range BT.2020 YUV conversion."""
    def _setup_ffmpeg(self, vidfile, resize_fn, resize_height, resize_width, verbose):
        self.dtype, self.bpp = np.uint16, 6
        self.frame_bytes = self.width * self.height * self.bpp
        self.process = (ffmpeg.input(vidfile).video
            .filter("scale", in_color_matrix="bt2020", out_color_matrix="bt2020",
                    in_range="tv", out_range="pc")
            .output("pipe:", format="rawvideo", pix_fmt="rgb48le", fps_mode="passthrough")
            .global_args("-loglevel", "error").run_async(pipe_stdout=True))


class HDRExperimentWriter(VideoWriter):
    """Keep the old HDR/SDR codecs, CRF 12, HDR optimization and bf=0."""
    def __init__(self, fname, fps, mode, preset="fast"):
        super().__init__(str(fname), fps=fps, hdr_mode=not mode.startswith("sdr"))
        self.preset = preset

    def write_frame_rgb(self, rgb):
        rgb = np.asarray(rgb)
        if rgb.ndim != 3 or rgb.shape[-1] != 3 or not np.isfinite(rgb).all():
            raise ValueError("Expected finite HWC RGB frame")
        rgb = np.clip(rgb, 0, 1)
        if self.process is None:
            height, width, _ = rgb.shape
            matrix = "bt2020" if self.hdr_mode else "bt709"
            primaries = "bt2020" if self.hdr_mode else "bt709"
            transfer = "smpte2084" if self.hdr_mode else "iec61966-2-1"
            options = dict(vcodec="libx265" if self.hdr_mode else "libx264",
                           pix_fmt="yuv420p10le" if self.hdr_mode else "yuv420p",
                           crf=12, bf=0, preset=self.preset, movflags="+faststart",
                           color_range="tv", color_primaries=primaries, color_trc=transfer,
                           colorspace="bt2020nc" if self.hdr_mode else "bt709")
            if self.hdr_mode:
                options["tag:v"] = "hvc1"
                options["x265-params"] = (
                    "hdr-opt=1:repeat-headers=1:colorprim=bt2020:transfer=smpte2084:"
                    "colormatrix=bt2020nc:log-level=error:pools=4")
            else:
                options["x264-params"] = "colorprim=bt709:transfer=iec61966-2-1:colormatrix=bt709:threads=4"
            self.process = (ffmpeg.input("pipe:", format="rawvideo",
                    pix_fmt="rgb48le" if self.hdr_mode else "rgb24", s=f"{width}x{height}", r=self.fps)
                .filter("scale", in_range="pc", out_range="tv", out_color_matrix=matrix)
                .filter("setparams", range="limited", color_primaries=primaries,
                        color_trc=transfer, colorspace="bt2020nc" if self.hdr_mode else "bt709")
                .output(self.fname, **options).overwrite_output()
                .global_args("-hide_banner", "-loglevel", "error").run_async(pipe_stdin=True))
        super().write_frame_rgb(rgb)

    def close(self):
        if self.process is not None:
            process, self.process = self.process, None
            try:
                process.stdin.close()
            finally:
                code = process.wait()
            if code:
                raise RuntimeError(f"Video encoder failed ({code}): {self.fname}")


def validate_mp4(path):
    """Check front-loaded MP4 index and strictly decode every frame."""
    atoms = []
    with path.open('rb') as stream:
        size = path.stat().st_size
        while stream.tell() < size:
            start = stream.tell()
            header = stream.read(8)
            if len(header) != 8:
                raise RuntimeError(f'Truncated MP4 atom: {path}')
            length, kind = struct.unpack('>I4s', header)
            header_size = 8
            if length == 1:
                extended = stream.read(8)
                if len(extended) != 8:
                    raise RuntimeError(f'Truncated MP4 atom: {path}')
                length = struct.unpack('>Q', extended)[0]
                header_size = 16
            if length == 0:
                length = size-start
            if length < header_size or start+length > size:
                raise RuntimeError(f'Invalid MP4 atom length: {path}')
            atoms.append(kind)
            stream.seek(start+length)
    if b'moov' not in atoms or b'mdat' not in atoms or atoms.index(b'moov') > atoms.index(b'mdat'):
        raise RuntimeError(f'MP4 faststart index missing: {path}')
    result = subprocess.run(['ffmpeg', '-v', 'error', '-xerror', '-i', str(path),
                             '-map', '0:v:0', '-f', 'null', '-'], capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(f'Full decode failed: {path}\n{result.stderr}')
    return dict(faststart=True, full_decode_passed=True)



