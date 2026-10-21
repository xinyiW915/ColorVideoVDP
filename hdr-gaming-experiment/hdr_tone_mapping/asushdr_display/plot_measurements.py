"""Plot the unmodified ASUS HDR display ramp measurements."""
# CSV: pixel = input pixel value; Y = measured luminance (cd/m², or nits); x, y = CIE chromaticity coordinates; 
# L, M, S = long-, medium-, and short-wavelength cone responses (the exact transform and scale depend on the measurement software).

from pathlib import Path
import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

folder = Path(__file__).resolve().parent
plots = [plt.subplots(figsize=(8, 5.5), layout="constrained") for _ in range(2)]
axes = [ax for _, ax in plots]
colors = {"white": "#333333", "red": "#d62728", "green": "#228833", "blue": "#3366cc"}

for name, color in colors.items():
    with (folder / f"resp_asushdr_pq1000_{name}.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream, skipinitialspace=True))
    x = np.array([float(row["pixel"]) for row in rows])
    y = np.array([float(row["Y"]) for row in rows])
    order = np.argsort(x)
    for ax in axes:
        ax.plot(x[order], y[order], "o-", color=color, markersize=4,
                linewidth=1.6, label=f"{name.title()} (max Y = {y.max():.2f})")
    print(f"{name}: {len(rows)} points; maximum measured Y = {y.max():.4g} cd/m^2")

axes[0].set(xscale="log", yscale="log", xlim=(0.005, 10000),
            title="Full measured range (log scales)")
axes[1].set(xlim=(0, 120), ylim=(0, 330), title="Low input range (linear scales)")
for ax in axes:
    ax.set_xlabel("Pixel value")
    ax.set_ylabel("Measured output luminance Y (cd/m²)")
    ax.grid(True, which="major", alpha=0.25)
    ax.legend(fontsize=8, loc="upper left")

for (fig, _), suffix in zip(plots, ("log", "linear")):
    fig.suptitle("ASUS HDR display: measured RGB and white responses", fontsize=13)
    target = folder / f"asus_hdr_response_{suffix}.png"
    fig.savefig(target, dpi=180)
    plt.close(fig)
    print(target)
