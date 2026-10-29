"""Make the 4K playback videos that observers see and ColorVideoVDP scores.

For one clip, from the outputs of hdr_tone_mapping/run.py and compression/compress.py,
this simulates the streaming client (upscale.py) and writes the playback encode:

  reference              tone-mapped lossless MKV -> playback encode
  tonemap_then_compress  server-side tone mapping: decode -> RGB -> upscale to 4K
                         -> clip to the target peak -> playback encode
  compress_then_tonemap  client/display tone mapping: decode -> RGB -> upscale to 4K
                         (once) -> tone map to each target at 4K -> playback encode

Upscalers (--upscalers): lanczos (Lanczos-5) and esrgan (Real-ESRGAN), both on RGB in
torch. 2160p is not upscaled, so it is written once to native/: tonemap_then_compress
is transcoded as is, compress_then_tonemap is tone mapped at 4K.

Playback encode: x265 CRF 10, preset fast, bframes=0:keyint=99999:scenecut=0 (as in
compress.py), 600 frames. HDR is BT.2020/PQ 10-bit, SDR BT.709/sRGB 8-bit.
Decoded, upscaled and tone-mapped frames are never written to disk.
"""

import argparse
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
from hdr_tone_mapping.video_io import validate_decode
from upscale import SIZE, UPSCALERS, PlaybackWriter, RGBReader, Upscaler

TONE_MAP_OUTPUT = ROOT / "hdr_tone_mapping" / "tone_map_output"
COMPRESSION_OUTPUT = ROOT / "compression" / "compression_output"
X265_PARAMS = "bframes=0:keyint=99999:scenecut=0"
PARTS = ("reference", "tonemap_then_compress", "compress_then_tonemap")
# Tone-mapped variant -> target peak in nits (None: SDR, clipped to white).
PEAK_NITS = {"hdr1000_calibrated": 1000, "hdr400_calibrated": 400, "sdr100_aces": None}
NATIVE = "native"  # test_output folder of the 2160p videos, shared by both upscalers


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


def is_hdr(variant):
    return PEAK_NITS[variant] is not None


def fmt_of(variant):
    return FORMATS["hdr" if is_hdr(variant) else "sdr"]


def transcode(source, target, variant, args):
    """4K video -> playback encode in one ffmpeg process, without going through RGB."""
    info = video_stream(source)
    if (info["width"], info["height"]) != SIZE:
        raise RuntimeError(f"{source} is not {SIZE[0]}x{SIZE[1]}")
    (
        ffmpeg.input(str(source))
        .video.output(str(target), **encode_options(fmt_of(variant), args))
        .overwrite_output()
        .global_args("-hide_banner", "-loglevel", "error")
        .run()
    )


def upscale(upscaler, source, target, variant, args):
    """tonemap_then_compress: decode -> RGB -> upscale -> clip to the peak -> encode."""
    import torch

    fps = video_stream(source)["r_frame_rate"]
    options = encode_options(fmt_of(variant), args)
    with ExitStack() as stack:
        stack.enter_context(torch.inference_mode())
        reader = stack.enter_context(
            RGBReader(str(source), is_hdr(variant), args.frames)
        )
        writer = stack.enter_context(
            PlaybackWriter(target, fps, is_hdr(variant), options, PEAK_NITS[variant])
        )
        for _ in range(args.frames):
            raw = reader.get_frame()
            if raw is None:
                raise RuntimeError(f"{source} has fewer than {args.frames} frames")
            writer.write_frame_rgb(upscaler(reader.unpack(raw, "cuda")))


def tone_map(upscaler, source, targets, args):
    """compress_then_tonemap: decode the HDR stream -> RGB -> upscale once (unless
    2160p) -> tone map to every target at 4K -> encode each."""
    import torch
    from hdr_tone_mapping.color_encoding import pq_decode
    from hdr_tone_mapping.hdr_spline import WHITE_CSV, WhiteResponse
    from hdr_tone_mapping.run import build_mappers

    mappers = build_mappers(WhiteResponse(str(WHITE_CSV)), "cuda", args.exposure_nits)
    fps = video_stream(source)["r_frame_rate"]
    with ExitStack() as stack:
        stack.enter_context(torch.inference_mode())
        reader = stack.enter_context(RGBReader(str(source), True, args.frames))
        writers = {
            variant: stack.enter_context(
                PlaybackWriter(
                    path,
                    fps,
                    is_hdr(variant),
                    encode_options(fmt_of(variant), args),
                    PEAK_NITS[variant],
                )
            )
            for variant, path in targets.items()
        }
        for _ in range(args.frames):
            raw = reader.get_frame()
            if raw is None:
                raise RuntimeError(f"{source} has fewer than {args.frames} frames")
            rgb = reader.unpack(raw, "cuda")
            if upscaler is not None:
                rgb = upscaler(rgb).clamp(0, 1)
            nits = pq_decode(rgb)
            for variant, writer in writers.items():
                writer.write_frame_rgb(mappers[variant](nits).clamp(0, 1))


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
        validate(partial_of(target), fmt_of(variant), args)
    for target in targets.values():
        partial_of(target).replace(target)


def jobs(clip, args):
    """One dict per playback job, ordered so each Real-ESRGAN model loads once."""
    out = Path(args.output)
    found = []
    if "reference" in args.parts:
        for variant, name in TONE_MAPPED.items():
            found.append(
                {
                    "label": f"reference {name}",
                    "kind": "transcode",
                    "upscaler": None,
                    "height": 2160,
                    "source": TONE_MAP_OUTPUT / f"{clip}_{variant}.mkv",
                    "targets": {variant: out / "reference" / f"{clip}_{name}.mp4"},
                }
            )
    folders = [(NATIVE, 2160)] if 2160 in args.heights else []
    folders += [(u, h) for u in args.upscalers for h in args.heights if h != 2160]
    for folder, height in folders:
        upscaler = None if folder == NATIVE else folder
        for mbps in args.bitrates:
            condition = f"{height}p_{mbps:g}M"
            if "tonemap_then_compress" in args.parts:
                route = "tonemap_then_compress"
                for variant, name in TONE_MAPPED.items():
                    found.append(
                        {
                            "label": f"{folder} {route} {name} {condition}",
                            "kind": "transcode" if upscaler is None else "upscale",
                            "upscaler": upscaler,
                            "height": height,
                            "source": COMPRESSION_OUTPUT
                            / route
                            / name
                            / f"{clip}_{variant}_{condition}.mp4",
                            "targets": {
                                variant: out
                                / folder
                                / route
                                / name
                                / f"{clip}_{name}_{condition}.mp4"
                            },
                        }
                    )
            if "compress_then_tonemap" in args.parts:
                route = "compress_then_tonemap"
                found.append(
                    {
                        "label": f"{folder} {route} {condition}",
                        "kind": "tonemap",
                        "upscaler": upscaler,
                        "height": height,
                        "source": COMPRESSION_OUTPUT
                        / route
                        / f"{clip}_{condition}.mp4",
                        "targets": {
                            variant: out
                            / folder
                            / route
                            / name
                            / f"{clip}_{name}_{condition}.mp4"
                            for variant, name in TONE_MAPPED.items()
                        },
                    }
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
        "--upscalers", nargs="+", choices=UPSCALERS, default=list(UPSCALERS)
    )
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
        "--crf", type=int, default=10, help="x265 CRF of the playback encode"
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
    missing = [str(job["source"]) for job in work if not job["source"].is_file()]
    if missing:
        parser.error("Missing inputs:\n  " + "\n  ".join(missing))
    targets = [t for job in work for t in job["targets"].values()]
    if not args.overwrite and any(t.exists() for t in targets):
        parser.error("Outputs exist; use --overwrite")

    import torch

    started = time.perf_counter()
    upscaler = None
    for i, job in enumerate(work, 1):
        job_started = time.perf_counter()
        print(f"[{i}/{len(work)}] {job['label']} ...", flush=True)
        if job["upscaler"] is not None and (
            upscaler is None
            or (upscaler.method, upscaler.height) != (job["upscaler"], job["height"])
        ):
            upscaler = None
            torch.cuda.empty_cache()
            upscaler = Upscaler(job["upscaler"], job["height"])
        for target in job["targets"].values():
            target.parent.mkdir(parents=True, exist_ok=True)
        partials = {v: partial_of(t) for v, t in job["targets"].items()}
        if job["kind"] == "transcode":
            ((variant, target),) = partials.items()
            transcode(job["source"], target, variant, args)
        elif job["kind"] == "upscale":
            ((variant, target),) = partials.items()
            upscale(upscaler, job["source"], target, variant, args)
        else:
            current = upscaler if job["upscaler"] is not None else None
            tone_map(current, job["source"], partials, args)
        finish(job["targets"], args)
        mbps = [
            t.stat().st_size * 8 / (args.frames / 60) / 1e6
            for t in job["targets"].values()
        ]
        print(
            f"  {', '.join(f'{x:.0f}' for x in mbps)} Mbps; "
            f"{time.perf_counter() - job_started:.0f}s",
            flush=True,
        )
    print(
        f"Completed {len(targets)} playback videos in "
        f"{time.perf_counter() - started:.0f}s",
        flush=True,
    )


if __name__ == "__main__":
    main()
