"""Preview what ColorVideoVDP sees after the display model, and check its luminance.

For each target, one playback test video and its reference are read with the same
video source and display models as score.py, and the display-model output is
written by ColorVideoVDP's dm-preview "metric" as OpenEXR frames: absolute linear
BT.709 RGB in cd/m^2, test on top, reference below.

  scoring/dm_preview/<clip>/<test video>-<frame>-test.exr  (ignored by git)

The luminance of those frames is printed next to the range the display model can
produce, which catches videos decoded with the wrong transfer function, range or
display model:

  floor    black level + ambient reflection (PQ also clips at 0.005 nits first)
  ceiling  target peak (HDR1000 1000 / HDR400 400 / SDR 100) + the same offsets

Run from the ColorVideoVDP root:

  python -B hdr-gaming-experiment/scoring/preview.py returnal_clip_005
"""

import argparse
import math

import numpy as np
from score import BITRATES, DISPLAY_FILE, HEIGHTS, HERE, ROUTES, TARGETS, TEST_OUTPUT

LUMA = np.array([0.2126, 0.7152, 0.0722])  # Linear BT.709 RGB -> luminance
TOLERANCE = (
    0.02  # Relative slack on the floor and ceiling (codec overshoot, float rounding)
)
OVERSHOOT = 0.1  # Pixels this far above the ceiling are out of range, not codec noise


def expected_range(photometry, target_peak):
    """Darkest and brightest luminance the display model can output for this target."""
    black, refl = photometry.get_black_level()
    if photometry.EOTF == "PQ":
        return 0.005 + black + refl, min(
            target_peak, photometry.get_peak_luminance()
        ) + black + refl
    return black + refl, photometry.get_peak_luminance() + refl


def stats(y):
    return {
        "min": y.min(),
        "median": np.median(y),
        "p99.9": np.percentile(y, 99.9),
        "max": y.max(),
    }


def preview(clip, target, out, args):
    import pyexr
    import torch

    import pycvvdp
    from pycvvdp.dm_preview_metric import dm_preview_exr_sbs

    display, peak = TARGETS[target]
    test = (
        TEST_OUTPUT
        / args.upscaler
        / args.route
        / target
        / f"{clip}_{target}_{args.height}p_{args.mbps:g}M.mp4"
    )
    reference = TEST_OUTPUT / "reference" / f"{clip}_{target}.mp4"
    missing = [str(p) for p in (test, reference) if not p.is_file()]
    if missing:
        raise FileNotFoundError("Missing playback videos:\n  " + "\n  ".join(missing))

    config = [str(DISPLAY_FILE)]
    photometry = pycvvdp.vvdp_display_photometry.load(display, config)
    display_width, display_height = pycvvdp.vvdp_display_geometry.load(
        display, config
    ).resolution
    source = pycvvdp.video_source_file(
        str(test),
        str(reference),
        display_photometry=display,
        config_paths=config,
        frames=args.frames,
    )
    height, width, frames = source.get_video_size()

    for old in out.glob(f"{test.stem}-*test.exr"):
        old.unlink()
    metric = dm_preview_exr_sbs(display_photometry=photometry)
    metric.set_base_fname(str(out / test.stem))
    with torch.no_grad():
        metric.predict_video_source(source)

    test_y, ref_y = [], []
    for path in sorted(out.glob(f"{test.stem}-*test.exr")):
        y = pyexr.read(str(path))[..., :3] @ LUMA
        test_y.append(y[: y.shape[0] // 2])
        ref_y.append(y[y.shape[0] // 2 :])
    images = {"test": np.stack(test_y), "ref": np.stack(ref_y)}
    rows = {name: stats(y) for name, y in images.items()}
    floor, ceiling = expected_range(photometry, peak)

    print(f"\n{target}  {display}  {frames} frames  {test.name}")
    print(f"  video {width}x{height}, display {display_width}x{display_height}")
    print(f"  expected  floor {floor:.4f}  ceiling {ceiling:.1f}  (cd/m^2)")
    print(f"  {'':6}" + "".join(f"{k:>11}" for k in rows["test"]))
    for name, row in rows.items():
        print(f"  {name:6}" + "".join(f"{v:11.4f}" for v in row.values()))

    warnings, notes = [], []
    if (width, height) != (display_width, display_height):
        warnings.append(
            f"video is {width}x{height} but the display is {display_width}x{display_height}; "
            "ColorVideoVDP shows it unscaled in the centre of the screen"
        )
    for name, row in rows.items():
        if row["min"] < floor * (1 - TOLERANCE):
            warnings.append(
                f"{name} is darker than the floor: is the display model applied?"
            )
        # A wrong transfer function or range scales the whole image well past the ceiling.
        # Codec noise only nudges pixels around it: areas clipped at the target peak come
        # back a few percent either side, and 4:2:0 overshoots at highlight edges.
        far = np.mean(images[name] > ceiling * (1 + OVERSHOOT)) * 100
        if far > 0.1:
            warnings.append(
                f"{name}: {far:.2f}% of pixels are over {OVERSHOOT:.0%} brighter than the ceiling: "
                "wrong transfer function, range or display model?"
            )
        elif row["max"] > ceiling * (1 + TOLERANCE):
            share = np.mean(images[name] > ceiling * (1 + TOLERANCE)) * 100
            notes.append(
                f"{name}: {share:.4f}% of pixels are above the ceiling (max {row['max']:.1f}), "
                "codec noise around areas clipped at the peak / 4:2:0 overshoot at highlight edges"
            )
    if abs(math.log(rows["test"]["median"] / rows["ref"]["median"])) > math.log(1.1):
        warnings.append(
            "test and reference medians differ by more than 10%: different content or decoding?"
        )
    if rows["ref"]["max"] < 0.5 * ceiling:
        notes.append(
            f"the reference peaks at {rows['ref']['max']:.1f}, well below the ceiling "
            "(fine if these frames have no bright highlights)"
        )
    for w in warnings:
        print(f"  WARNING: {w}")
    for n in notes:
        print(f"  note: {n}")
    if not warnings:
        print("  OK")
    return len(warnings)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("clip", help="Clip name, e.g. returnal_clip_005")
    parser.add_argument(
        "--targets", nargs="+", choices=list(TARGETS), default=list(TARGETS)
    )
    parser.add_argument("--route", choices=ROUTES, default=ROUTES[0])
    parser.add_argument("--height", type=int, choices=HEIGHTS, default=2160)
    parser.add_argument("--mbps", type=float, choices=BITRATES, default=25)
    parser.add_argument("--upscaler", default="lanczos", help="test_output subfolder")
    parser.add_argument(
        "--frames", type=int, default=3, help="Frames to write and check"
    )
    args = parser.parse_args()

    out = HERE / "dm_preview" / args.clip
    out.mkdir(parents=True, exist_ok=True)
    warnings = sum(preview(args.clip, target, out, args) for target in args.targets)
    print(f"\n{warnings} warning(s). EXR frames in {out}")


if __name__ == "__main__":
    main()
