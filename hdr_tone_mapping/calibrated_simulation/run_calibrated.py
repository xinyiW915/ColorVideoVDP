"""Simulate measured white response with a spline and ACES Filmic."""
import argparse
from contextlib import ExitStack
from fractions import Fraction
import json
from pathlib import Path
import sys
import time



HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from hdr_tone_mapping.video_io import HDRReader, HDRExperimentWriter, validate_mp4
from hdr_tone_mapping.sdr_mapping import encode_numpy

from hdr_tone_mapping.simple_spline.tone_mapping import pq_decode, pq_encode, srgb_encode, BT2020_LUMA, RGB2020_TO_709
from hdr_tone_mapping.calibrated_simulation.white_response import WhiteResponse, scale_luminance
import ffmpeg
import numpy as np


def plot(model, folder):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    x = np.geomspace(.005, model.source_peak, 4096)
    fig, ax = plt.subplots(figsize=(8, 6), layout="constrained")
    ax.loglog(model.pixel, model.measured, 'o', color='black', ms=4, label='Measured white response')
    ax.loglog(x, model.forward(x), '--', color='black', label='Measured response spline')
    ax.loglog(x, x, ':', color='gray', label='1:1')
    for peak, label in [(400, 'Calibrated: HDR spline 400 nit'), (1000, 'Calibrated: HDR spline 1000 nit')]:
        color = 'tab:blue' if peak == 400 else 'tab:orange'
        ax.loglog(x, model.baseline(x, peak), '--', color=color, alpha=.6, label=f'Calibrated simulation: {peak} nit')
        ax.loglog(x, model.spline(x, peak), color=color, label=label)
    # Save the reference-style HDR figure first; add ACES in a separate comparison.
    ax.set(xlabel='Input luminance (nits)', ylabel='Output luminance (nits)',
           title='Tone mapping curves')
    ax.grid(which='both', alpha=.2)
    ax.legend()
    fig.savefig(folder / "calibrated_curves.png", dpi=180)
    ax.loglog(x, model.aces(x), color='tab:green', label='SDR100 ACES Filmic')
    ax.legend()
    fig.savefig(folder / 'tone_curves_with_aces.png', dpi=180)
    plt.close(fig)


def convert(args, model, folder):
    started = time.perf_counter()
    source = Path(args.input)
    stream = next(s for s in ffmpeg.probe(str(source))["streams"] if s["codec_type"] == "video")
    if tuple(stream.get(k) for k in ("color_transfer", "color_primaries", "color_space", "color_range")) != (
            "smpte2084", "bt2020", "bt2020nc", "tv"):
        raise ValueError("Expected limited-range BT.2020/PQ input")
    modes = ["hdr1000_spline", "hdr400_spline", "sdr100_aces_filmic"]
    targets = {m: folder / f"{source.stem}_{m}.mp4" for m in modes}
    if not args.overwrite and any(p.exists() for p in targets.values()):
        raise FileExistsError("Outputs exist; use --overwrite")
    partials = {m: p.with_name(p.stem + ".partial.mp4") for m, p in targets.items()}
    lut = pq_decode(np.arange(65536)/65535).astype(np.float32)
    import torch
    device = args.device
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    gpu = None
    if device == 'cuda':
        from hdr_tone_mapping.calibrated_simulation.frame_mapping import ToneMapper
        gpu = ToneMapper(model, lut, RGB2020_TO_709)
    print(f'Mapping device: {device}; encoding: CPU ({args.preset})', flush=True)
    fps = stream["r_frame_rate"]
    count, gamut_clipped = 0, {m: 0 for m in modes if m.startswith("sdr")}
    with ExitStack() as stack:
        reader = stack.enter_context(HDRReader(str(source)))
        if reader.width % 2 or reader.height % 2:
            raise ValueError("Even dimensions required")
        total = reader.frames if reader.frames > 0 else None
        if args.max_frames is not None:
            total = min(total, args.max_frames) if total else args.max_frames
        last_progress = time.perf_counter()
        writers = {m: stack.enter_context(HDRExperimentWriter(p, fps, m, args.preset)) for m, p in partials.items()}
        while args.max_frames is None or count < args.max_frames:
            raw = reader.get_frame()
            if raw is None:
                break
            if raw.size != reader.width*reader.height*3:
                raise RuntimeError("Incomplete decoded frame")
            codes = raw.reshape(reader.height, reader.width, 3)
            if gpu is not None:
                rgb, y = gpu.prepare(codes)
            else:
                rgb = lut[codes]
                y = rgb @ BT2020_LUMA
            for mode, peak in zip(modes, (1000, 400, 100)):
                if gpu is not None:
                    encoded, clipped = gpu.encode(rgb, y, mode)
                    if mode.startswith('sdr'):
                        gamut_clipped[mode] += clipped
                    writers[mode].write_frame_rgb(encoded)
                    continue
                if mode.startswith('sdr'):
                    encoded, clipped = encode_numpy(rgb, RGB2020_TO_709, args.exposure_nits)
                    gamut_clipped[mode] += clipped
                    writers[mode].write_frame_rgb(encoded)
                    continue
                target = model.spline(y, peak)
                mapped = scale_luminance(rgb, target, BT2020_LUMA)
                if np.max(mapped) > 10000+1e-6:
                    raise ValueError('Mapped RGB exceeds PQ range')
                encoded = pq_encode(mapped)
                writers[mode].write_frame_rgb(encoded)
            count += 1
            now = time.perf_counter()
            if count == 1 or now-last_progress >= 5 or (total and count == total):
                elapsed = now-started
                rate = count/elapsed
                eta = f'; ETA {(total-count)/rate:.0f}s' if total else ''
                print(f'Frames {count}/{total or "?"}; {rate:.2f} fps; elapsed {elapsed:.1f}s{eta}', flush=True)
                last_progress = now
        print('Finishing encoders and validating outputs...', flush=True)
    if count == 0:
        raise RuntimeError("No frames decoded")
    checks = {}
    decode_checks = {}
    for mode, partial in partials.items():
        output = next(s for s in ffmpeg.probe(str(partial), count_frames=None)["streams"] if s["codec_type"] == "video")
        transfer = "iec61966-2-1" if mode.startswith("sdr") else "smpte2084"
        primaries = "bt709" if mode.startswith("sdr") else "bt2020"
        if (int(output["nb_read_frames"]) != count or Fraction(output["r_frame_rate"]) != Fraction(fps)
            or output.get("color_transfer") != transfer or output.get("color_primaries") != primaries
            or (output["width"], output["height"]) != (reader.width, reader.height)):
            raise RuntimeError(f"Validation failed: {partial}")
        checks[mode] = output
        print(f'Full decode validation: {mode}', flush=True)
        decode_checks[mode] = validate_mp4(partial)
    for mode, partial in partials.items():
        partial.replace(targets[mode])
    report = dict(parameters=vars(args), calibration=model.summary(), frames=count, outputs=checks,
                  sdr_mapping="Shared Y-only ACES Filmic; exposure_nits; SDR white=100 nits",
                  mapping_device=device, elapsed_seconds=time.perf_counter()-started,
                  playback_validation=decode_checks,
                  writer_settings='legacy CPU x265/x264, CRF 12, bf=0, hdr-opt=1 (HDR), MP4 faststart',
                  sdr_white_nits=100, sdr_gamut_clipped_pixels=gamut_clipped,
                  mapping="Scaled measured response with shadow-only correction; Y-only",
                  drive_assumption="Scaled measured pixel/Y response; nominal input luminance assumption",
                  playback="PQ output contains simulated absolute nits; SDR requires managed white at sdr_white_nits",
                  timing="Nominal CFR; no audio; original timestamps not preserved")
    (folder / f"{source.stem}_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Completed {count} frames for each of three variants", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=r"C:\Users\xw506\Videos\Elgato Studio\HDR\test\clips\input05\input05_clip_002.mp4")
    parser.add_argument("--white-csv", default=str(ROOT / "asushdr_display" / "resp_asushdr_pq1000_white.csv"))
    parser.add_argument("--output", default=str(HERE / "tone_map_output"))
    parser.add_argument('--source-peak', type=float, default=10000)
    parser.add_argument('--exposure-nits', type=float, default=100)
    parser.add_argument('--shadow-start', type=float, default=.01)
    parser.add_argument('--shadow-end', type=float, default=.1)
    parser.add_argument("--max-frames", type=int,
                        help="Maximum frames to process; omit to process the entire video")
    parser.add_argument("--preset", default="fast")
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto',
                        help='Mapping device; auto selects CUDA when available; encoding stays on CPU')
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--plot-only", action="store_true")
    args = parser.parse_args()
    if args.max_frames is not None and args.max_frames <= 0:
        parser.error("max-frames must be positive")
    folder = Path(args.output)
    folder.mkdir(parents=True, exist_ok=True)
    model = WhiteResponse(args.white_csv, args.source_peak, args.exposure_nits, args.shadow_start, args.shadow_end)
    plot(model, folder)
    summary = model.summary()
    (folder / "calibration_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    if not args.plot_only:
        convert(args, model, folder)


if __name__ == "__main__":
    main()






