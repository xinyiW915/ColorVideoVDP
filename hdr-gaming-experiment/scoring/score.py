"""Score the playback videos with ColorVideoVDP and plot the results.

Each test video in playback/test_output/<upscaler>/<route>/<target>/ is compared
with the reference of the same target, playback/test_output/reference/. All
targets use display models of the same ASUS PA32UCX in the same room
(display_models_asus.json), so scores are comparable across targets:

  hdr1000, hdr400  asus_pa32ucx_pq1000  (BT.2020/PQ, 1000-nit peak)
  sdr              asus_pa32ucx_sdr100  (BT.709/sRGB, 100-nit white)

Results go to scoring_output/<clip>_cvvdp.csv, one row per test video, written as
each score finishes; a rerun skips rows already scored with the same frame count.
The plots are drawn from the CSV:

  <clip>_jod_vs_bitrate.png     JOD vs bitrate, both routes
  <clip>_route_difference.png   JOD(tonemap_then_compress) - JOD(compress_then_tonemap)
  <clip>_route_scatter.png      the same comparison as a scatter plot
  <clip>_peak_luminance.png     JOD vs peak luminance (SDR 100 / HDR400 / HDR1000)
"""

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT.parent))

TEST_OUTPUT = ROOT / "playback" / "test_output"
DISPLAY_FILE = HERE / "display_models_asus.json"
# Target -> (ColorVideoVDP display model, peak luminance in nits).
TARGETS = {
    "hdr1000": ("asus_pa32ucx_pq1000", 1000),
    "hdr400": ("asus_pa32ucx_pq1000", 400),
    "sdr": ("asus_pa32ucx_sdr100", 100),
}
ROUTES = ("tonemap_then_compress", "compress_then_tonemap")
HEIGHTS = (720, 1080, 2160)
BITRATES = (5, 12, 25)
FIELDS = [
    "clip", "upscaler", "route", "target", "height", "mbps", "frames", "jod",
    "display", "test", "reference",
]

# Categorical slots 1-3 of the dataviz reference palette (validated all-pairs).
RES_COLORS = {720: "#2a78d6", 1080: "#eb6834", 2160: "#1baf7a"}
# Route -> (line style, filled marker, label).
ROUTE_STYLE = {
    "tonemap_then_compress": ("-", True, "TM first"),
    "compress_then_tonemap": ("--", False, "Compress first"),
}
RC = {
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
    "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10, "figure.titlesize": 13,
    "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "axes.grid.axis": "y", "grid.color": "#dddddd", "grid.linewidth": 0.8,
    "axes.axisbelow": True, "lines.linewidth": 1.8, "lines.markersize": 6,
}
BITRATE_MARKERS = {5: "o", 12: "s", 25: "^"}
TARGET_LABELS = {"hdr1000": "HDR1000", "hdr400": "HDR400", "sdr": "SDR"}
SAME_BAND = 0.2  # +/- JOD drawn as "too small to notice" in the route plots; a guide only


def key(row):
    return (row["upscaler"], row["route"], row["target"], int(row["height"]), float(row["mbps"]))


def jobs(clip, args):
    for upscaler in args.upscalers:
        for route in ROUTES:
            for target in args.targets:
                for height in args.heights:
                    for mbps in args.bitrates:
                        test = (TEST_OUTPUT / upscaler / route / target
                                / f"{clip}_{target}_{height}p_{mbps:g}M.mp4")
                        yield {
                            "clip": clip, "upscaler": upscaler, "route": route,
                            "target": target, "height": height, "mbps": mbps,
                            "test": test,
                            "reference": TEST_OUTPUT / "reference" / f"{clip}_{target}.mp4",
                        }


def read_rows(path):
    if not path.is_file():
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def score(clip, csv_path, args):
    import pycvvdp
    import torch

    todo = list(jobs(clip, args))
    missing = [str(p) for job in todo for p in (job["test"], job["reference"]) if not p.is_file()]
    if missing:
        raise FileNotFoundError("Missing playback videos:\n  " + "\n  ".join(sorted(set(missing))))
    frames = str(args.frames)
    rows = read_rows(csv_path)
    done = set() if args.overwrite else {key(r) for r in rows if r["frames"] == frames}
    todo = [j for j in todo if key(j) not in done]
    # Rows about to be rescored (overwrite, or a different frame count) are dropped.
    rescored = {key(j) for j in todo}
    rows = [r for r in rows if key(r) not in rescored]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"{len(done)} already scored; {len(todo)} to score", flush=True)

    config = [str(DISPLAY_FILE)]
    metrics = {}
    started = time.perf_counter()
    for i, job in enumerate(todo, 1):
        display = TARGETS[job["target"]][0]
        if display not in metrics:
            metrics[display] = pycvvdp.cvvdp(display_name=display, config_paths=config)
            print(metrics[display].get_info_string(), flush=True)
        source = pycvvdp.video_source_file(
            str(job["test"]), str(job["reference"]),
            display_photometry=display, config_paths=config, frames=args.frames,
        )
        job_started = time.perf_counter()
        with torch.no_grad():
            jod = float(metrics[display].predict_video_source(source)[0])
        row = {**job, "frames": frames, "jod": f"{jod:.4f}", "display": display,
               "test": job["test"].relative_to(ROOT), "reference": job["reference"].relative_to(ROOT)}
        with open(csv_path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=FIELDS).writerow(row)
        print(f"[{i}/{len(todo)}] {job['route']} {job['target']} {job['height']}p {job['mbps']:g}M: "
              f"{jod:.3f} JOD ({time.perf_counter() - job_started:.0f}s)", flush=True)
    print(f"Scored {len(todo)} videos in {time.perf_counter() - started:.0f}s", flush=True)


def table(rows):
    """{(upscaler, route, target, height, mbps): jod}"""
    return {key(r): float(r["jod"]) for r in rows}


def draw(ax, x, y, color, linestyle="-", filled=True):
    ax.plot(x, y, color=color, linestyle=linestyle, marker="o", markeredgecolor=color,
            markerfacecolor=color if filled else "white", markeredgewidth=1.5)


def categories(ax, labels, xlabel):
    """Experimental conditions as evenly spaced categories."""
    ax.set_xticks(range(len(labels)), labels)
    ax.set_xlim(-0.3, len(labels) - 0.7)
    ax.set_xlabel(xlabel)


def plot(clip, rows, args):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    jod = table(rows)
    upscaler = args.upscalers[0]
    get = lambda r, t, h, b: jod.get((upscaler, r, t, h, float(b)), np.nan)  # noqa: E731
    targets = [t for t in TARGETS if t in args.targets]
    out = HERE / "scoring_output"
    x = range(len(BITRATES))
    bitrate_labels = [f"{b}" for b in BITRATES]
    resolutions = [Line2D([], [], color=c, marker="o", label=f"{h}p") for h, c in RES_COLORS.items()]
    routes = [Line2D([], [], color="black", linestyle=ls, marker="o", markeredgewidth=1.5,
                     markerfacecolor="black" if filled else "white", label=label)
              for ls, filled, label in ROUTE_STYLE.values()]

    def figure(nrows, ncols, title, width=3.6, height=3.0, **kwargs):
        fig, axes = plt.subplots(nrows, ncols, figsize=(width * ncols, height * nrows + 0.6),
                                 squeeze=False, layout="constrained", **kwargs)
        fig.suptitle(title)
        return fig, axes

    def save(fig, handles, name):
        fig.legend(handles=handles, loc="outside lower center", ncol=len(handles), frameon=False)
        fig.savefig(out / f"{clip}_{name}.png", dpi=200)
        plt.close(fig)

    with plt.rc_context(RC):
        # 1. JOD vs bitrate.
        fig, axes = figure(1, len(targets), "JOD vs bitrate", sharey=True)
        for ax, t in zip(axes[0], targets):
            for route, (ls, filled, _) in ROUTE_STYLE.items():
                for h, c in RES_COLORS.items():
                    draw(ax, x, [get(route, t, h, b) for b in BITRATES], c, ls, filled)
            ax.set_title(TARGET_LABELS[t])
            categories(ax, bitrate_labels, "Bitrate (Mbps)")
        axes[0][0].set_ylabel("JOD")
        save(fig, resolutions + routes, "jod_vs_bitrate")

        # 2. Route difference.
        diff = {(t, h, b): get(ROUTES[0], t, h, b) - get(ROUTES[1], t, h, b)
                for t in targets for h in HEIGHTS for b in BITRATES}
        span = max(SAME_BAND * 1.5,
                   max((abs(v) for v in diff.values() if np.isfinite(v)), default=0) * 1.15)
        fig, axes = figure(1, len(targets), "TM first − compress first", sharey=True)
        for ax, t in zip(axes[0], targets):
            ax.axhspan(-SAME_BAND, SAME_BAND, color="#eeeeee", linewidth=0)
            ax.axhline(0, color="black", linewidth=0.8)
            for h, c in RES_COLORS.items():
                draw(ax, x, [diff[t, h, b] for b in BITRATES], c)
            ax.set_title(TARGET_LABELS[t])
            ax.set_ylim(-span, span)
            categories(ax, bitrate_labels, "Bitrate (Mbps)")
        axes[0][0].set_ylabel("ΔJOD")
        save(fig, resolutions + [Patch(color="#eeeeee", label=f"±{SAME_BAND} JOD")],
             "route_difference")

        # 2b. Route scatter.
        values = [v for v in jod.values() if np.isfinite(v)]
        lo = (min(values) if values else 8) - 0.1
        hi = min(10, (max(values) if values else 10) + 0.1)
        fig, axes = figure(1, len(targets), "TM first vs compress first", height=4.4,
                           sharex=True, sharey=True)
        for ax, t in zip(axes[0], targets):
            ax.grid(True, axis="both")
            ax.plot([lo, hi], [lo, hi], color="black", linewidth=0.8)
            for h, c in RES_COLORS.items():
                for b, m in BITRATE_MARKERS.items():
                    ax.plot(get(ROUTES[1], t, h, b), get(ROUTES[0], t, h, b), linestyle="",
                            marker=m, markersize=7, color=c)
            ax.set(title=TARGET_LABELS[t], xlabel="JOD, compress first", xlim=(lo, hi),
                   ylim=(lo, hi), aspect="equal")
        axes[0][0].set_ylabel("JOD, TM first")
        markers = [Line2D([], [], color="black", linestyle="", marker=m, label=f"{b} Mbps")
                   for b, m in BITRATE_MARKERS.items()]
        save(fig, [Line2D([], [], color=c, linestyle="", marker="o", label=f"{h}p")
                   for h, c in RES_COLORS.items()] + markers, "route_scatter")

        # 3. Peak luminance.
        by_peak = sorted(targets, key=lambda t: TARGETS[t][1])
        fig, axes = figure(len(ROUTES), len(BITRATES), "JOD vs peak luminance",
                           sharex=True, sharey=True)
        for row, route in enumerate(ROUTES):
            for col, b in enumerate(BITRATES):
                ax = axes[row][col]
                for h, c in RES_COLORS.items():
                    draw(ax, range(len(by_peak)), [get(route, t, h, b) for t in by_peak], c)
                categories(ax, [f"{TARGET_LABELS[t]}\n{TARGETS[t][1]} nits" for t in by_peak],
                           "Peak luminance" if row == len(ROUTES) - 1 else "")
                if row == 0:
                    ax.set_title(f"{b} Mbps")
            axes[row][0].set_ylabel(f"{ROUTE_STYLE[route][2]}\nJOD")
        save(fig, resolutions, "peak_luminance")
    print(f"Plots saved in {out}", flush=True)


def summary(rows):
    jod = table(rows)
    for upscaler in sorted({k[0] for k in jod}):
        for t in TARGETS:
            print(f"\n{upscaler} {t}: JOD  TM first / compress first (difference)")
            print("         " + "".join(f"{b:>22} Mbps" for b in BITRATES))
            for h in HEIGHTS:
                cells = []
                for b in BITRATES:
                    a = jod.get((upscaler, ROUTES[0], t, h, float(b)), np.nan)
                    c = jod.get((upscaler, ROUTES[1], t, h, float(b)), np.nan)
                    cells.append(f"{a:6.2f} / {c:5.2f} ({a - c:+.2f})")
                print(f"  {h:>5}p " + "".join(f"{s:>27}" for s in cells))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("clip", help="Clip name, e.g. returnal_clip_005")
    parser.add_argument("--targets", nargs="+", choices=list(TARGETS), default=list(TARGETS))
    parser.add_argument("--heights", type=int, nargs="+", choices=HEIGHTS, default=list(HEIGHTS))
    parser.add_argument("--bitrates", type=float, nargs="+", default=list(BITRATES))
    parser.add_argument("--upscalers", nargs="+", default=["lanczos"], help="test_output subfolders to score")
    parser.add_argument("--frames", type=int, default=-1, help="Frames to score; -1 for all (600)")
    parser.add_argument("--plot-only", action="store_true", help="Only redraw the plots from the CSV")
    parser.add_argument("--overwrite", action="store_true", help="Rescore rows already in the CSV")
    args = parser.parse_args()

    out = HERE / "scoring_output"
    out.mkdir(exist_ok=True)
    csv_path = out / f"{args.clip}_cvvdp.csv"
    if not args.plot_only:
        score(args.clip, csv_path, args)
    rows = [r for r in read_rows(csv_path) if r["frames"] == str(args.frames)]
    if not rows:
        raise SystemExit(f"No scores with --frames {args.frames} in {csv_path}")
    summary(rows)
    plot(args.clip, rows, args)


if __name__ == "__main__":
    main()
