"""Read HDR clips with ColorVideoVDP and write HDR1000/HDR400/SDR variants."""
import argparse
from contextlib import ExitStack
from fractions import Fraction
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
# This checkout's video_source_file uses absolute sibling imports.
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "pycvvdp"))
import ffmpeg
import numpy as np
from hdr_tone_mapping.video_io import HDRReader, HDRExperimentWriter, validate_mp4
from hdr_tone_mapping.sdr_mapping import encode_numpy

from hdr_tone_mapping.simple_spline.tone_mapping import (hdr_spline, map_hdr_luminance, pq_decode, pq_encode, sdr_aces_fit,
                          srgb_encode, RGB2020_TO_709)


def plot_curves(folder, args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    x = np.geomspace(.005, 10000, 4096)
    fig, ax = plt.subplots(figsize=(8, 6), layout="constrained")
    ax.loglog(x, x, "--", color="gray", label="1:1")
    for peak in (1000, 400):
        y = hdr_spline(x, peak, args.source_peak, args.knee_ratio)
        ax.loglog(x, y, label=f"HDR spline {peak} nits")
    y = 100 * sdr_aces_fit(x, args.exposure_nits)
    ax.loglog(x, y, label="SDR ACES Filmic (100-nit display)")
    ax.set(xlabel="Input luminance (nits)", ylabel="Output luminance (nits)",
           title="Tone mapping curves")
    ax.grid(alpha=.25)
    ax.legend()
    fig.savefig(folder / "tone_mapping_curves.png", dpi=180)
    plt.close(fig)


def convert(path, folder, args):
    info = ffmpeg.probe(str(path))
    stream = next(s for s in info["streams"] if s["codec_type"] == "video")
    if (stream.get("color_transfer"), stream.get("color_primaries"),
        stream.get("color_space"), stream.get("color_range")) != (
            "smpte2084", "bt2020", "bt2020nc", "tv"):
        raise ValueError(f"Expected limited-range BT.2020/PQ input: {path}")
    fps = stream["r_frame_rate"]
    if stream["width"] % 2 or stream["height"] % 2:
        raise ValueError("4:2:0 output requires even dimensions")
    codes = np.arange(65536, dtype=np.float64) / 65535
    nits = pq_decode(codes)
    import torch
    device = args.device
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    mapper = None
    if device == 'cuda':
        from hdr_tone_mapping.simple_spline.frame_mapping import ToneMapper
        mapper = ToneMapper(nits.astype(np.float32), RGB2020_TO_709,
                            args.source_peak, args.knee_ratio, args.exposure_nits)
    modes = ["hdr1000_spline", "hdr400_spline", "sdr_aces_fit"]
    targets = {m: folder / f"{path.stem}_{m}.mp4" for m in modes}
    if any(p.exists() for p in targets.values()) and not args.overwrite:
        raise FileExistsError(f"Outputs exist for {path.name}; use --overwrite explicitly")
    partials = {m: p.with_name(p.stem + ".partial.mp4") for m, p in targets.items()}
    count, start = 0, time.monotonic()
    with ExitStack() as stack:
        reader = stack.enter_context(HDRReader(str(path)))
        total = reader.frames if reader.frames > 0 else None
        if args.max_frames is not None:
            total = min(total, args.max_frames) if total else args.max_frames
        last_progress = time.monotonic()
        print(f'Mapping device: {device}; encoding: CPU ({args.preset})', flush=True)
        writers = {m: stack.enter_context(HDRExperimentWriter(p, fps, m, args.preset)) for m, p in partials.items()}
        while args.max_frames is None or count < args.max_frames:
            raw = reader.get_frame()
            if raw is None:
                break
            if raw.size != reader.width * reader.height * 3:
                raise RuntimeError("Incomplete decoded frame")
            codes_frame = raw.reshape(reader.height, reader.width, 3)
            if mapper is not None:
                rgb_nits, y = mapper.prepare(codes_frame)
                for mode in modes:
                    writers[mode].write_frame_rgb(mapper.encode(rgb_nits, y, mode))
            else:
                rgb_nits = nits[codes_frame].astype(np.float32)
                for peak, mode in zip((1000, 400), modes):
                    mapped = map_hdr_luminance(rgb_nits, peak, args.source_peak, args.knee_ratio)
                    writers[mode].write_frame_rgb(pq_encode(mapped))
                encoded_sdr, _ = encode_numpy(rgb_nits, RGB2020_TO_709, args.exposure_nits)
                writers["sdr_aces_fit"].write_frame_rgb(encoded_sdr)
            count += 1
            now = time.monotonic()
            if count == 1 or now-last_progress >= 5 or (total and count == total):
                elapsed = now-start
                rate = count/elapsed
                eta = f'; ETA {(total-count)/rate:.0f}s' if total else ''
                print(f'Frames {count}/{total or "?"}; {rate:.2f} fps; elapsed {elapsed:.1f}s{eta}', flush=True)
                last_progress = now
        print('Finishing encoders and validating outputs...', flush=True)
    if not count:
        raise RuntimeError(f"No frames decoded: {path}")
    # Count encoded frames before considering the files complete.
    validation = {}
    playback_validation = {}
    for mode, partial in partials.items():
        probe = ffmpeg.probe(str(partial), count_frames=None)
        output = next(s for s in probe["streams"] if s["codec_type"] == "video")
        expected_transfer = "iec61966-2-1" if mode == "sdr_aces_fit" else "smpte2084"
        if (int(output["nb_read_frames"]) != count or
            Fraction(output["r_frame_rate"]) != Fraction(fps) or
            output.get("color_transfer") != expected_transfer or
            (output["width"], output["height"]) != (reader.width, reader.height)):
            raise RuntimeError(f"Output validation failed: {partial}")
        playback_validation[mode] = validate_mp4(partial)
        validation[mode] = output
    for mode, partial in partials.items():
        partial.replace(targets[mode])
    report = dict(input=str(path), input_stream=stream, frames=count,
                  sdr_mapping="Shared Y-only ACES Filmic; exposure_nits; SDR white=100 nits",
                  mapping_device=device,
                  hdr_mapping="luminance spline; BT.2020 Y; RGB scaled by Y_out/Y_in",
                  parameters=vars(args), outputs=validation, playback_validation=playback_validation, elapsed_seconds=time.monotonic()-start,
                  timing="Nominal CFR; input timestamps are not preserved; no audio")
    (folder / f"{path.stem}_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Completed {path.name}: {count} frames for each of three variants", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=r"C:\Users\xw506\Videos\Elgato Studio\HDR\test\clips\input05\input05_clip_002.mp4",
                        help="Single HDR video file")
    parser.add_argument("--output", default=str(Path(__file__).parent / "tone_map_output"))
    parser.add_argument("--source-peak", type=float, default=10000)
    parser.add_argument("--knee-ratio", type=float, default=.5)
    parser.add_argument("--exposure-nits", type=float, default=100)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--preset", default="fast")
    parser.add_argument('--device', choices=['auto','cpu','cuda'], default='auto',
                        help='Mapping device; auto selects CUDA when available; encoding stays on CPU')
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--plot-only", action="store_true")
    args = parser.parse_args()
    if args.max_frames is not None and args.max_frames <= 0:
        parser.error("--max-frames must be positive")
    source = Path(args.input)
    if not args.plot_only and not source.is_file():
        parser.error(f"Input must be a single existing video file: {source}")
    folder = Path(args.output)
    folder.mkdir(parents=True, exist_ok=True)
    plot_curves(folder, args)
    if not args.plot_only:
        convert(source, folder, args)


if __name__ == "__main__":
    main()

