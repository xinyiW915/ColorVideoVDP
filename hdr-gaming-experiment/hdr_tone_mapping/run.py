"""Write calibrated HDR1000/HDR400 spline and SDR ACES Filmic variants of an HDR clip."""

import argparse
import json
import sys
import time
from contextlib import ExitStack
from fractions import Fraction
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
import ffmpeg
import numpy as np

from hdr_tone_mapping.color_encoding import pq_decode
from hdr_tone_mapping.hdr_spline import (
    TARGET_PEAKS,
    WHITE_CSV,
    HDRSpline,
    WhiteResponse,
)
from hdr_tone_mapping.sdr_aces import SDRAces
from hdr_tone_mapping.video_io import HDRExperimentWriter, HDRReader, validate_decode

COLORS = {400: "tab:blue", 1000: "tab:orange"}


def build_mappers(model, device, exposure_nits):
    """Output variant -> frame mapper (linear BT.2020 RGB nits -> encoded frame)."""
    return {
        "hdr1000_calibrated": HDRSpline(model, 1000, device),
        "hdr400_calibrated": HDRSpline(model, 400, device),
        "sdr100_aces": SDRAces(device, exposure_nits),
    }


def plot(model, folder, exposure_nits):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = np.geomspace(0.005, 10000, 4096)
    fig, ax = plt.subplots(figsize=(8, 6), layout="constrained")
    ax.loglog(
        model.pixel,
        model.measured,
        "o",
        color="black",
        ms=4,
        label="Measured white response",
    )
    ax.loglog(
        x, model.forward(x), "--", color="black", label="Measured response spline"
    )
    ax.loglog(x, x, ":", color="gray", label="1:1")
    for peak in TARGET_PEAKS:
        color = COLORS[peak]
        ax.loglog(
            x,
            model.simulation(x, peak),
            "--",
            color=color,
            alpha=0.6,
            label=f"Calibrated simulation: {peak} nit",
        )
        ax.loglog(
            x,
            model.spline(x, peak),
            color=color,
            label=f"Calibrated: HDR spline {peak} nit",
        )
    ax.loglog(
        x,
        SDRAces(exposure_nits=exposure_nits).curve(x),
        color="tab:green",
        label="SDR100 ACES Filmic",
    )
    ax.set(
        xlabel="Input luminance (nits)",
        ylabel="Output luminance (nits)",
        title="Tone mapping curves",
    )
    ax.grid(which="both", alpha=0.2)
    ax.legend()
    fig.savefig(folder / "tone_curves_with_aces.png", dpi=180)
    plt.close(fig)


def convert(args, source, model, folder, device):
    started = time.perf_counter()
    stream = next(
        s for s in ffmpeg.probe(str(source))["streams"] if s["codec_type"] == "video"
    )
    if tuple(
        stream.get(k)
        for k in ("color_transfer", "color_primaries", "color_space", "color_range")
    ) != ("smpte2084", "bt2020", "bt2020nc", "tv"):
        raise ValueError("Expected limited-range BT.2020/PQ input")
    mappers = build_mappers(model, device, args.exposure_nits)
    targets = {m: folder / f"{source.stem}_{m}.mp4" for m in mappers}
    if not args.overwrite and any(p.exists() for p in targets.values()):
        raise FileExistsError("Outputs exist; use --overwrite")
    partials = {m: p.with_name(p.stem + ".partial.mp4") for m, p in targets.items()}
    import torch

    print(f"Input: {source}", flush=True)
    print(f"Mapping device: {device}; encoding: CPU ({args.preset})", flush=True)
    fps = stream["r_frame_rate"]
    count = 0
    with ExitStack() as stack:
        stack.enter_context(torch.inference_mode())
        reader = stack.enter_context(HDRReader(str(source)))
        if reader.width % 2 or reader.height % 2:
            raise ValueError("4:2:0 output requires even dimensions")
        total = reader.frames if reader.frames > 0 else None
        if args.max_frames is not None:
            total = min(total, args.max_frames) if total else args.max_frames
        last_progress = time.perf_counter()
        writers = {
            m: stack.enter_context(
                HDRExperimentWriter(p, fps, not m.startswith("sdr"), args.preset)
            )
            for m, p in partials.items()
        }
        while args.max_frames is None or count < args.max_frames:
            raw = reader.get_frame()
            if raw is None:
                break
            if raw.size != reader.width * reader.height * 3:
                raise RuntimeError("Incomplete decoded frame")
            # unpack() gives the PQ-encoded RGB in [0, 1]; decode to linear nits.
            rgb = pq_decode(reader.unpack(raw, device))
            for mode, mapper in mappers.items():
                # VideoWriter expects a display-encoded HxWx3 NumPy frame in [0, 1].
                writers[mode].write_frame_rgb(mapper(rgb).clamp(0, 1).cpu().numpy())
            count += 1
            now = time.perf_counter()
            if count == 1 or now - last_progress >= 5 or (total and count == total):
                elapsed = now - started
                rate = count / elapsed
                eta = f"; ETA {(total - count) / rate:.0f}s" if total else ""
                print(
                    f"Frames {count}/{total or '?'}; {rate:.2f} fps; elapsed {elapsed:.1f}s{eta}",
                    flush=True,
                )
                last_progress = now
        print("Finishing encoders and validating outputs...", flush=True)
    if count == 0:
        raise RuntimeError("No frames decoded")
    checks, decode_checks = {}, {}
    for mode, partial in partials.items():
        # Count packets without decoding; validate_decode decodes every frame afterwards.
        output = next(
            s
            for s in ffmpeg.probe(str(partial), count_packets=None)["streams"]
            if s["codec_type"] == "video"
        )
        hdr = not mode.startswith("sdr")
        if (
            int(output["nb_read_packets"]) != count
            or Fraction(output["r_frame_rate"]) != Fraction(fps)
            or output.get("color_transfer") != ("smpte2084" if hdr else "iec61966-2-1")
            or output.get("color_primaries") != ("bt2020" if hdr else "bt709")
            or (output["width"], output["height"]) != (reader.width, reader.height)
        ):
            raise RuntimeError(f"Validation failed: {partial}")
        checks[mode] = output
        print(f"Full decode validation: {mode}", flush=True)
        decode_checks[mode] = validate_decode(partial)
    for mode, partial in partials.items():
        partial.replace(targets[mode])
    report = {
        "input": str(source),
        "parameters": vars(args),
        "calibration": model.summary(),
        "frames": count,
        "outputs": checks,
        "mappings": {m: mapper.summary() for m, mapper in mappers.items()},
        "mapping_device": device,
        "elapsed_seconds": time.perf_counter() - started,
        "playback_validation": decode_checks,
        "reader": "pycvvdp video_reader; ffmpeg command with explicit limited-range BT.2020 YUV->RGB",
        "writer_settings": (
            "pycvvdp VideoWriter; ffmpeg command with explicit RGB->YUV matrix and tags "
            f"(BT.2020/PQ HDR, BT.709/sRGB SDR); x265/x264 CRF 12, preset {args.preset}, bf=0, "
            "hdr-opt=1 (HDR), MP4 faststart"
        ),
        "playback": "PQ output contains simulated absolute nits; SDR requires managed white at 100 nits",
        "timing": "Nominal CFR; no audio; original timestamps not preserved",
    }
    (folder / f"{source.stem}_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    elapsed = time.perf_counter() - started
    print(
        f"Completed {count} frames for each of three variants in {elapsed:.0f}s "
        f"({count / elapsed:.2f} fps including encoder flush and validation)",
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "video", nargs="?", help="Path to the HDR video file to process"
    )
    parser.add_argument("--white-csv", default=str(WHITE_CSV))
    parser.add_argument("--output", default=str(HERE / "tone_map_output"))
    parser.add_argument(
        "--exposure-nits",
        type=float,
        default=100,
        help="Input luminance mapped to ACES x=1 for the SDR variant",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        help="Maximum frames to process; omit to process the entire video",
    )
    parser.add_argument("--preset", default="fast", help="x265/x264 preset")
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Mapping device; auto selects CUDA when available; encoding stays on CPU",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--plot-only", action="store_true")
    args = parser.parse_args()
    if args.max_frames is not None and args.max_frames <= 0:
        parser.error("--max-frames must be positive")
    if args.exposure_nits <= 0:
        parser.error("--exposure-nits must be positive")
    source = None
    if not args.plot_only:
        if args.video is None:
            parser.error("give the path of the video to process")
        source = Path(args.video)
        if not source.is_file():
            parser.error(f"Video file not found: {source}")
    folder = Path(args.output)
    folder.mkdir(parents=True, exist_ok=True)
    model = WhiteResponse(args.white_csv)
    plot(model, folder, args.exposure_nits)
    summary = model.summary()
    (folder / "calibration_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)
    if args.plot_only:
        return
    import torch

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    elif device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable; use --device cpu")
    convert(args, source, model, folder, device)


if __name__ == "__main__":
    main()
