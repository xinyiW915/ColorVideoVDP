"""Make the 4K playback videos that observers see and ColorVideoVDP scores.

For one clip, from the outputs of hdr_tone_mapping/run.py and compression/compress.py:

  reference              tone-mapped lossless MKV                 -> playback encode
  tonemap_then_compress  compressed MP4 -> Lanczos-5 upscale      -> playback encode
  compress_then_tonemap  compressed MP4 -> tone map -> Lanczos-5 upscale -> playback encode

Upscaling is done on the limited-range YUV 4:2:0 code values, as the downscale in
compress.py; 2160p inputs are not scaled. In compress_then_tonemap the tone-mapped
frames are converted to YUV 4:2:0 at the compressed resolution first (as run.py
does at 4K), so both routes upscale the same kind of frame.

Playback encode: x265 CRF 12, preset fast, bframes=0:keyint=99999:scenecut=0 (as
in compress.py), 600 frames. HDR is BT.2020/PQ 10-bit, SDR BT.709/sRGB 8-bit.
Decoded, upscaled and tone-mapped frames are never written to disk.
"""

import argparse
import subprocess
import sys
import time
from contextlib import ExitStack
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
import ffmpeg
from compression.compress import (
    BITRATES_MBPS,
    FORMATS,
    RESOLUTIONS,
    TONE_MAPPED,
    video_stream,
)
from hdr_tone_mapping.video_io import HDRExperimentWriter, HDRReader, validate_decode

TONE_MAP_OUTPUT = ROOT / "hdr_tone_mapping" / "tone_map_output"
COMPRESSION_OUTPUT = ROOT / "compression" / "compression_output"
SIZE = RESOLUTIONS[2160]
UPSCALE = f"scale={SIZE[0]}:{SIZE[1]}:flags=lanczos:param0=5"
X265_PARAMS = "bframes=0:keyint=99999:scenecut=0"
PARTS = ("reference", "tonemap_then_compress", "compress_then_tonemap")


def encode_options(fmt, args):
    return {
        "frames:v": args.frames,
        "vcodec": "libx265",
        "crf": args.crf,
        "preset": args.preset,
        "pix_fmt": fmt["pix_fmt"],
        "x265-params": X265_PARAMS + ":log-level=error",
        "color_range": "tv",
        "color_primaries": fmt["primaries"],
        "color_trc": fmt["trc"],
        "colorspace": fmt["matrix"],
        "an": None,
    }


def transcode(source, target, fmt, args):
    """Decode, Lanczos-5 upscale if below 4K, playback encode; one ffmpeg process."""
    info = video_stream(source)
    stream = ffmpeg.input(str(source)).video
    if (info["width"], info["height"]) != SIZE:
        stream = stream.filter("scale", SIZE[0], SIZE[1], flags="lanczos", param0=5)
    (
        stream.output(str(target), **encode_options(fmt, args))
        .overwrite_output()
        .global_args("-hide_banner", "-loglevel", "error")
        .run()
    )


class UpscaleWriter(HDRExperimentWriter):
    """Rounded RGB in (as run.py), YUV 4:2:0 at the input size, Lanczos-5 upscale, playback encode."""

    def __init__(self, fname, fps, hdr_mode, args):
        super().__init__(fname, fps, hdr_mode)
        self.args = args

    def _start_ffmpeg(self, height, width):
        fmt = FORMATS["hdr" if self.hdr_mode else "sdr"]
        matrix = "bt2020" if self.hdr_mode else "bt709"
        stream = (
            ffmpeg.input(
                "pipe:",
                format="rawvideo",
                pix_fmt="rgb48le" if self.hdr_mode else "rgb24",
                s=f"{width}x{height}",
                r=self.fps,
            )
            .filter("scale", in_range="pc", out_range="tv", out_color_matrix=matrix)
            .filter("format", fmt["pix_fmt"])
        )
        if (width, height) != SIZE:
            stream = stream.filter("scale", SIZE[0], SIZE[1], flags="lanczos", param0=5)
        return (
            stream.filter(
                "setparams",
                range="limited",
                color_primaries=fmt["primaries"],
                color_trc=fmt["trc"],
                colorspace=fmt["matrix"],
            )
            .output(self.fname, **encode_options(fmt, self.args))
            .overwrite_output()
            .global_args("-hide_banner", "-loglevel", "error")
            .run_async(pipe_stdin=True)
        )


def tone_map_and_upscale(source, targets, args):
    """Tone map one compressed HDR clip to every target and write each upscaled playback video."""
    import torch
    from hdr_tone_mapping.color_encoding import pq_decode
    from hdr_tone_mapping.hdr_spline import WHITE_CSV, WhiteResponse
    from hdr_tone_mapping.run import build_mappers

    device = "cuda" if torch.cuda.is_available() else "cpu"
    mappers = build_mappers(WhiteResponse(str(WHITE_CSV)), device, args.exposure_nits)
    fps = video_stream(source)["r_frame_rate"]
    with ExitStack() as stack:
        stack.enter_context(torch.inference_mode())
        reader = stack.enter_context(HDRReader(str(source)))
        writers = {
            variant: stack.enter_context(
                UpscaleWriter(path, fps, not variant.startswith("sdr"), args)
            )
            for variant, path in targets.items()
        }
        for _ in range(args.frames):
            raw = reader.get_frame()
            if raw is None:
                raise RuntimeError(f"{source} has fewer than {args.frames} frames")
            rgb = pq_decode(reader.unpack(raw, device))
            for variant, writer in writers.items():
                writer.write_frame_rgb(mappers[variant](rgb).clamp(0, 1).cpu().numpy())


def validate(path, fmt, args):
    output = video_stream(path, count_packets=None)
    if (
        int(output["nb_read_packets"]) != args.frames
        or (output["width"], output["height"]) != SIZE
        or output.get("codec_name") != "hevc"
        or output.get("pix_fmt") != fmt["pix_fmt"]
        or output.get("color_transfer") != fmt["trc"]
        or output.get("color_primaries") != fmt["primaries"]
    ):
        raise RuntimeError(f"Validation failed: {path}")
    validate_decode(path)


def partial_of(target):
    return target.with_name(target.stem + ".partial.mp4")


def finish(targets, args):
    """Validate .partial.mp4 outputs, then move them to their final names."""
    for variant, target in targets.items():
        validate(
            partial_of(target),
            FORMATS["sdr" if variant.startswith("sdr") else "hdr"],
            args,
        )
    for target in targets.values():
        partial_of(target).replace(target)


def jobs(clip, args):
    """(label, kind, source, {variant: target}) for every playback video of the clip."""
    out = Path(args.output)
    conditions = [(h, m) for h in args.heights for m in args.bitrates]
    found = []
    if "reference" in args.parts:
        for variant, name in TONE_MAPPED.items():
            source = TONE_MAP_OUTPUT / f"{clip}_{variant}.mkv"
            found.append(
                (
                    f"reference {name}",
                    "transcode",
                    source,
                    {variant: out / "reference" / f"{clip}_{name}.mp4"},
                )
            )
    if "tonemap_then_compress" in args.parts:
        for variant, name in TONE_MAPPED.items():
            for h, m in conditions:
                condition = f"{h}p_{m:g}M"
                source = (
                    COMPRESSION_OUTPUT
                    / "tonemap_then_compress"
                    / name
                    / f"{clip}_{variant}_{condition}.mp4"
                )
                target = (
                    out
                    / "lanczos"
                    / "tonemap_then_compress"
                    / name
                    / f"{clip}_{name}_{condition}.mp4"
                )
                found.append(
                    (
                        f"tonemap_then_compress {name} {condition}",
                        "transcode",
                        source,
                        {variant: target},
                    )
                )
    if "compress_then_tonemap" in args.parts:
        for h, m in conditions:
            condition = f"{h}p_{m:g}M"
            source = (
                COMPRESSION_OUTPUT / "compress_then_tonemap" / f"{clip}_{condition}.mp4"
            )
            targets = {
                variant: out
                / "lanczos"
                / "compress_then_tonemap"
                / name
                / f"{clip}_{name}_{condition}.mp4"
                for variant, name in TONE_MAPPED.items()
            }
            found.append(
                (f"compress_then_tonemap {condition}", "tonemap", source, targets)
            )
    return found


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "clip", help="Clip name, e.g. returnal_clip_005 (the original clip's file stem)"
    )
    parser.add_argument("--output", default=str(HERE / "test_output"))
    parser.add_argument("--parts", nargs="+", choices=PARTS, default=list(PARTS))
    parser.add_argument(
        "--heights",
        type=int,
        nargs="+",
        choices=sorted(RESOLUTIONS),
        default=sorted(RESOLUTIONS, reverse=True),
    )
    parser.add_argument(
        "--bitrates",
        type=float,
        nargs="+",
        default=list(BITRATES_MBPS),
        help="Bitrates (Mbps) of the compressed inputs",
    )
    parser.add_argument(
        "--crf", type=int, default=12, help="x265 CRF of the playback encode"
    )
    parser.add_argument(
        "--preset", default="fast", help="x265 preset of the playback encode"
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=600,
        help="Frames per video (default 600: the 10 s clip at 60 fps)",
    )
    parser.add_argument(
        "--exposure-nits",
        type=float,
        default=100,
        help="As hdr_tone_mapping/run.py, for the SDR tone map",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    work = jobs(args.clip, args)
    missing = [str(source) for _, _, source, _ in work if not source.is_file()]
    if missing:
        parser.error("Missing inputs:\n  " + "\n  ".join(missing))
    targets = [t for *_, ts in work for t in ts.values()]
    if not args.overwrite and any(t.exists() for t in targets):
        parser.error("Outputs exist; use --overwrite")

    started = time.perf_counter()
    for i, (label, kind, source, outputs) in enumerate(work, 1):
        job_started = time.perf_counter()
        print(f"[{i}/{len(work)}] {label} ...", flush=True)
        for target in outputs.values():
            target.parent.mkdir(parents=True, exist_ok=True)
        if kind == "transcode":
            ((variant, target),) = outputs.items()
            fmt = FORMATS["sdr" if variant.startswith("sdr") else "hdr"]
            transcode(source, partial_of(target), fmt, args)
        else:
            tone_map_and_upscale(
                source, {v: str(partial_of(t)) for v, t in outputs.items()}, args
            )
        finish(outputs, args)
        mbps = [
            t.stat().st_size * 8 / (args.frames / 60) / 1e6 for t in outputs.values()
        ]
        print(
            f"  {', '.join(f'{x:.0f}' for x in mbps)} Mbps; {time.perf_counter() - job_started:.0f}s",
            flush=True,
        )
    print(
        f"Completed {len(targets)} playback videos in {time.perf_counter() - started:.0f}s",
        flush=True,
    )


if __name__ == "__main__":
    main()
