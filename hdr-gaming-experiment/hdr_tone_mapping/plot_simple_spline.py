"""Plot the earlier simple HDR spline against the calibrated spline, for reference only.

The simple spline is no longer used to render videos.
"""

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
import numpy as np
from scipy.interpolate import CubicHermiteSpline

from hdr_tone_mapping.hdr_spline import TARGET_PEAKS, WHITE_CSV, WhiteResponse

# Same colours as the run.py curve plots.
COLORS = {400: "tab:blue", 1000: "tab:orange"}


def simple_spline(nits, peak, source_peak=10000, knee_ratio=0.5):
    """Identity below knee; C1 monotone cubic spline in log(input), log(output).

    The two shoulder pieces join at input=peak. The final endpoint is
    (source_peak, peak), with zero slope; a designed content transform,
    not derived from the measured display.
    """
    if not (0 < knee_ratio <= 0.5 and 0 < peak < source_peak <= 10000):
        raise ValueError("Require 0 < knee_ratio <= .5 and 0 < peak < source_peak <= 10000")
    knee = peak * knee_ratio
    a, b, c = np.log([knee, peak, source_peak])
    delta = b - a
    middle_slope = min(0.2, 1.2 * delta / (c - b))
    spline = CubicHermiteSpline(
        [a, b, c], [a, a + 0.6 * delta, b], [1, middle_slope, 0], extrapolate=False
    )
    values = np.clip(np.asarray(nits, dtype=np.float64), 0, source_peak)
    shoulder = np.exp(spline(np.log(np.clip(values, knee, source_peak))))
    return np.where(values <= knee, values, shoulder)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--white-csv", default=str(WHITE_CSV))
    parser.add_argument("--output", default=str(HERE / "tone_map_output"))
    parser.add_argument("--knee-ratio", type=float, default=0.5)
    args = parser.parse_args()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    model = WhiteResponse(args.white_csv)
    x = np.geomspace(0.005, 10000, 4096)
    fig, ax = plt.subplots(figsize=(8, 6), layout="constrained")
    ax.loglog(x, x, ":", color="gray", label="1:1")
    for peak in TARGET_PEAKS:
        color = COLORS[peak]
        ax.loglog(x, simple_spline(x, peak, knee_ratio=args.knee_ratio), "--", color=color,
                  label=f"Simple spline {peak} nit (knee {args.knee_ratio * peak:.0f})")
        ax.loglog(x, model.spline(x, peak), color=color,
                  label=f"Calibrated spline {peak} nit (knee {model.knee * peak:.0f})")
    ax.set(xlabel="Input luminance (nits)", ylabel="Output luminance (nits)",
           title="Simple vs calibrated HDR spline")
    ax.grid(which="both", alpha=0.2)
    ax.legend()
    folder = Path(args.output)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "simple_vs_calibrated_spline.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
