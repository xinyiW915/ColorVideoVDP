"""Score the playback videos with ColorVideoVDP and plot the results.

Each test video in playback/test_output/<upscaler>/<route>/<target>/ is compared
with the reference of the same target, playback/test_output/reference/. 2160p is
not upscaled: it lives in test_output/native/, is scored once (upscaler "native" in
the CSV) and is used for both upscalers. All targets use display models of the
same ASUS PA32UCX in the same room (display_models_asus.json), so scores are
comparable across targets:

  hdr1000, hdr400  asus_pa32ucx_pq1000  (BT.2020/PQ, 1000-nit peak)
  sdr              asus_pa32ucx_sdr100  (BT.709/sRGB, 100-nit white)

Results go to scoring_output/<clip>_cvvdp.csv, one row per test video, written as
each score finishes; a rerun skips rows already scored with the same frame count.
The plots are drawn from the CSV, with both upscalers in each plot (Lanczos-5
solid and filled, Real-ESRGAN dashed and hollow, 2160p dotted):

  <clip>_jod_vs_bitrate.png    JOD vs bitrate; a row per route, a column per target
  <clip>_peak_luminance.png    JOD vs peak luminance (SDR 100 / HDR400 / HDR1000)

With several clips, all_clips_jod_vs_bitrate.png and all_clips_peak_luminance.png
put them side by side: a column per clip, a row per route and target (bitrate),
one JOD axis. all_clips_route_scatter.png puts their route scatters side by side
(TM first vs compress first; a column per clip, a row per target), with the same
JOD range on both axes of every subplot. Route scatters are only drawn there.
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
RESULTS = HERE / "scoring_output"
UPSCALERS = ("lanczos", "esrgan")
NATIVE = "native"  # test_output folder of the 2160p videos, shared by both upscalers
UPSCALER_LABELS = {"lanczos": "Lanczos-5", "esrgan": "Real-ESRGAN"}
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
    "clip",
    "upscaler",
    "route",
    "target",
    "height",
    "mbps",
    "frames",
    "jod",
    "display",
    "test",
    "reference",
]

# Categorical slots 1-3 of the dataviz reference palette (validated all-pairs).
RES_COLORS = {720: "#2a78d6", 1080: "#eb6834", 2160: "#1baf7a"}
ROUTE_LABELS = {
    "tonemap_then_compress": "TM first",
    "compress_then_tonemap": "Compress first",
}
# Upscaler -> (line style, filled marker).
UPSCALER_STYLE = {"lanczos": ("-", True), "esrgan": ("--", False)}
RC = {
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "figure.titlesize": 13,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.grid.axis": "y",
    "grid.color": "#dddddd",
    "grid.linewidth": 0.8,
    "axes.axisbelow": True,
    "lines.linewidth": 1.8,
    "lines.markersize": 6,
}
BITRATE_MARKERS = {5: "o", 12: "s", 25: "^"}
TARGET_LABELS = {"hdr1000": "HDR1000", "hdr400": "HDR400", "sdr": "SDR"}
UPSCALED = tuple(h for h in HEIGHTS if h != 2160)
JOD_RANGE = (5.0, 10.0)  # JOD axis of every plot (and both axes of the route scatter)
ALL_CLIPS_STEMS = ("jod_vs_bitrate", "peak_luminance", "route_scatter")


def key(row):
    return (
        row["upscaler"],
        row["route"],
        row["target"],
        int(row["height"]),
        float(row["mbps"]),
    )


def folder_of(upscaler, height):
    """test_output folder of a condition: 2160p is not upscaled, so it is native."""
    return NATIVE if height == 2160 else upscaler


def test_path(test_output, clip, upscaler, route, target, height, mbps):
    return (
        Path(test_output)
        / folder_of(upscaler, height)
        / route
        / target
        / f"{clip}_{target}_{height}p_{mbps:g}M.mp4"
    )


def rel(path):
    return path.relative_to(ROOT) if path.is_relative_to(ROOT) else path


def jobs(clip, args):
    """One job per test video; 2160p appears once whatever the upscalers."""
    seen = set()
    for upscaler in args.upscalers:
        for route in ROUTES:
            for target in args.targets:
                for height in args.heights:
                    for mbps in args.bitrates:
                        job = {
                            "clip": clip,
                            "upscaler": folder_of(upscaler, height),
                            "route": route,
                            "target": target,
                            "height": height,
                            "mbps": mbps,
                            "test": test_path(
                                args.test_output,
                                clip,
                                upscaler,
                                route,
                                target,
                                height,
                                mbps,
                            ),
                            "reference": Path(args.test_output)
                            / "reference"
                            / f"{clip}_{target}.mp4",
                        }
                        if key(job) not in seen:
                            seen.add(key(job))
                            yield job


def read_rows(path):
    if not path.is_file():
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def score(clip, csv_path, args):
    import torch

    import pycvvdp

    todo = list(jobs(clip, args))
    missing = [
        str(p)
        for job in todo
        for p in (job["test"], job["reference"])
        if not p.is_file()
    ]
    if missing:
        raise FileNotFoundError(
            "Missing playback videos:\n  " + "\n  ".join(sorted(set(missing)))
        )
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
            str(job["test"]),
            str(job["reference"]),
            display_photometry=display,
            config_paths=config,
            frames=args.frames,
        )
        job_started = time.perf_counter()
        with torch.no_grad():
            jod = float(metrics[display].predict_video_source(source)[0])
        row = {
            **job,
            "frames": frames,
            "jod": f"{jod:.4f}",
            "display": display,
            "test": rel(job["test"]),
            "reference": rel(job["reference"]),
        }
        with open(csv_path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=FIELDS).writerow(row)
        print(
            f"[{i}/{len(todo)}] {job['route']} {job['target']} {job['height']}p {job['mbps']:g}M: "
            f"{jod:.3f} JOD ({time.perf_counter() - job_started:.0f}s)",
            flush=True,
        )
    print(
        f"Scored {len(todo)} videos in {time.perf_counter() - started:.0f}s", flush=True
    )


def table(rows):
    """{(upscaler, route, target, height, mbps): jod}"""
    return {key(r): float(r["jod"]) for r in rows}


def draw(ax, x, y, color, linestyle="-", filled=True):
    ax.plot(
        x,
        y,
        color=color,
        linestyle=linestyle,
        marker="o",
        markeredgecolor=color,
        markerfacecolor=color if filled else "white",
        markeredgewidth=1.5,
    )


def axis_range(values, fixed=JOD_RANGE, step=0.5, label="JOD"):
    """The axis range of every plot of a metric: fixed, so plots of different runs
    and clips compare directly. Widened in steps (with a note) only if some value
    falls outside it."""
    values = [v for v in values if np.isfinite(v)]
    lo, hi = fixed
    if values and (min(values) < lo or max(values) > hi):
        lo = min(lo, float(np.floor(min(values) / step) * step))
        hi = max(hi, float(np.ceil(max(values) / step) * step))
        print(
            f"note: {label} {min(values):.2f}..{max(values):.2f} is outside {fixed}; "
            f"plotting {lo:g}..{hi:g}, which does not match the other plots",
            flush=True,
        )
    return lo, hi


def categories(ax, labels, xlabel):
    """Experimental conditions as evenly spaced categories."""
    ax.set_xticks(range(len(labels)), labels)
    ax.set_xlim(-0.3, len(labels) - 0.7)
    ax.set_xlabel(xlabel)


def upscaler_lines(ax, xs, values, upscalers):
    """2160p (not upscaled) as one dotted line, then each upscaler at 720p/1080p.
    values(upscaler, height) gives the y values."""
    draw(ax, xs, values(NATIVE, 2160), RES_COLORS[2160], ":")
    for u in upscalers:
        ls, filled = UPSCALER_STYLE[u]
        for h in UPSCALED:
            draw(ax, xs, values(u, h), RES_COLORS[h], ls, filled)


def upscaler_handle(u, **kwargs):
    from matplotlib.lines import Line2D

    ls, filled = UPSCALER_STYLE[u]
    return Line2D(
        [],
        [],
        color="black",
        marker="o",
        markeredgewidth=1.5,
        markerfacecolor="black" if filled else "white",
        label=UPSCALER_LABELS[u],
        **{"linestyle": ls, **kwargs},
    )


def line_legend(upscalers):
    """Legend of upscaler_lines."""
    from matplotlib.lines import Line2D

    return (
        [
            Line2D([], [], color=RES_COLORS[h], marker="o", label=f"{h}p")
            for h in UPSCALED
        ]
        + [
            Line2D(
                [],
                [],
                color=RES_COLORS[2160],
                linestyle=":",
                marker="o",
                label="2160p (not upscaled)",
            )
        ]
        + [upscaler_handle(u) for u in upscalers]
    )


def route_scatter(ax, jod, target, upscalers, lo, hi):
    """Score of TM first (y) vs compress first (x) for one target: a colour per
    height, a marker per bitrate, hollow for Real-ESRGAN."""
    ax.grid(True, axis="both")
    ax.plot([lo, hi], [lo, hi], color="black", linewidth=0.8)
    for u, h in [(NATIVE, 2160)] + [(u, h) for u in upscalers for h in UPSCALED]:
        c = RES_COLORS[h]
        for b, m in BITRATE_MARKERS.items():
            ax.plot(
                jod.get((u, ROUTES[1], target, h, float(b)), np.nan),
                jod.get((u, ROUTES[0], target, h, float(b)), np.nan),
                linestyle="",
                marker=m,
                markersize=7,
                color=c,
                markeredgewidth=1.5,
                markerfacecolor="white" if u == "esrgan" else c,
            )
    ax.set(xlim=(lo, hi), ylim=(lo, hi), aspect="equal")


def scatter_legend(upscalers):
    """Legend of route_scatter."""
    from matplotlib.lines import Line2D

    return (
        [
            Line2D([], [], color=c, linestyle="", marker="o", label=f"{h}p")
            for h, c in RES_COLORS.items()
        ]
        + [
            Line2D([], [], color="black", linestyle="", marker=m, label=f"{b} Mbps")
            for b, m in BITRATE_MARKERS.items()
        ]
        + [upscaler_handle(u, linestyle="") for u in upscalers]
    )


def plot(name, jod, args, ylim):
    """Draw the plots of one clip's JOD table, {(upscaler, route, target, height, mbps): jod},
    with the JOD axis range ylim."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lookup = lambda u, r, t, h, b: jod.get((folder_of(u, h), r, t, h, float(b)), np.nan)  # noqa: E731
    targets = [t for t in TARGETS if t in args.targets]
    upscalers = [u for u in UPSCALERS if u in args.upscalers]
    out = Path(args.results)
    x = range(len(BITRATES))
    bitrate_labels = [f"{b}" for b in BITRATES]
    last = len(ROUTES) - 1
    lines = lambda ax, xs, values: upscaler_lines(ax, xs, values, upscalers)  # noqa: E731
    line_handles = line_legend(upscalers)

    def figure(nrows, ncols, title):
        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(3.6 * ncols, 3.0 * nrows + 0.6),
            squeeze=False,
            sharex=True,
            sharey=True,
            layout="constrained",
        )
        axes[0][0].set_ylim(ylim)  # shared by every subplot
        fig.suptitle(f"{title} ({name})")
        return fig, axes

    def save(fig, handles, stem):
        fig.legend(
            handles=handles,
            loc="outside lower center",
            ncol=len(handles),
            frameon=False,
        )
        fig.savefig(out / f"{name}_{stem}.png", dpi=200)
        plt.close(fig)

    with plt.rc_context(RC):
        # JOD vs bitrate.
        fig, axes = figure(len(ROUTES), len(targets), "JOD vs bitrate")
        for row, route in enumerate(ROUTES):
            for col, t in enumerate(targets):
                ax = axes[row][col]
                lines(
                    ax, x, lambda u, h: [lookup(u, route, t, h, b) for b in BITRATES]
                )
                if row == 0:
                    ax.set_title(TARGET_LABELS[t])
                categories(
                    ax, bitrate_labels, "Bitrate (Mbps)" if row == last else ""
                )
            axes[row][0].set_ylabel(f"{ROUTE_LABELS[route]}\nJOD")
        save(fig, line_handles, "jod_vs_bitrate")

        # JOD vs peak luminance.
        by_peak = sorted(targets, key=lambda t: TARGETS[t][1])
        fig, axes = figure(len(ROUTES), len(BITRATES), "JOD vs peak luminance")
        for row, route in enumerate(ROUTES):
            for col, b in enumerate(BITRATES):
                ax = axes[row][col]
                lines(
                    ax,
                    range(len(by_peak)),
                    lambda u, h: [lookup(u, route, t, h, b) for t in by_peak],
                )
                categories(
                    ax,
                    [f"{TARGET_LABELS[t]}\n{TARGETS[t][1]} nits" for t in by_peak],
                    "Peak luminance" if row == last else "",
                )
                if row == 0:
                    ax.set_title(f"{b} Mbps")
            axes[row][0].set_ylabel(f"{ROUTE_LABELS[route]}\nJOD")
        save(fig, line_handles, "peak_luminance")
    print(f"{name}: plots saved in {out}", flush=True)


def plot_clips(tables, args, ylim, label="JOD", stems=ALL_CLIPS_STEMS):
    """Several clips side by side, with the axis range ylim: a column per clip, and
    a row per route and target (vs bitrate) or per route and bitrate (vs peak
    luminance), plus the route scatters. tables is {clip: {key: score}} of the metric
    called label; the plots go to all_clips_<stem>.png, with the stems of the
    bitrate, peak luminance and route scatter plots."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    targets = [t for t in TARGETS if t in args.targets]
    by_peak = sorted(targets, key=lambda t: TARGETS[t][1])
    upscalers = [u for u in UPSCALERS if u in args.upscalers]

    def grid(stem, title, rows, row_label, xlabels, xlabel, conditions):
        """conditions(row) gives the (route, target, mbps) of each x position."""
        handles = line_legend(upscalers)
        fig, axes = plt.subplots(
            len(rows),
            len(tables),
            figsize=(3.2 * len(tables) + 0.6, 2.4 * len(rows) + 0.8),
            squeeze=False,
            sharex=True,
            sharey=True,
            layout="constrained",
        )
        axes[0][0].set_ylim(ylim)  # shared by every subplot
        for col, (clip, jod) in enumerate(tables.items()):
            for row, key_ in enumerate(rows):
                ax = axes[row][col]
                upscaler_lines(
                    ax,
                    range(len(xlabels)),
                    lambda u, h: [
                        jod.get((folder_of(u, h), r, t, h, float(b)), np.nan)
                        for r, t, b in conditions(key_)
                    ],
                    upscalers,
                )
                categories(ax, xlabels, xlabel if row == len(rows) - 1 else "")
                if row == 0:
                    ax.set_title(clip)
                if col == 0:
                    ax.set_ylabel(f"{row_label(key_)}\n{label}")
        fig.suptitle(f"{title}, all clips")
        fig.legend(
            handles=handles, loc="outside lower center", ncol=len(handles), frameon=False
        )
        fig.savefig(Path(args.results) / f"all_clips_{stem}.png", dpi=200)
        plt.close(fig)

    with plt.rc_context(RC):
        grid(
            stems[0],
            f"{label} vs bitrate",
            [(r, t) for r in ROUTES for t in targets],
            lambda rt: f"{ROUTE_LABELS[rt[0]]}\n{TARGET_LABELS[rt[1]]}",
            [str(b) for b in BITRATES],
            "Bitrate (Mbps)",
            lambda rt: [(rt[0], rt[1], b) for b in BITRATES],
        )
        grid(
            stems[1],
            f"{label} vs peak luminance",
            [(r, b) for r in ROUTES for b in BITRATES],
            lambda rb: f"{ROUTE_LABELS[rb[0]]}\n{rb[1]} Mbps",
            [f"{TARGET_LABELS[t]}\n{TARGETS[t][1]} nits" for t in by_peak],
            "Peak luminance",
            lambda rb: [(rb[0], t, rb[1]) for t in by_peak],
        )

        # TM first vs compress first: a column per clip, a row per target, and the
        # same range on both axes of every subplot.
        lo, hi = ylim
        fig, axes = plt.subplots(
            len(targets),
            len(tables),
            figsize=(3.4 * len(tables) + 0.6, 3.4 * len(targets) + 0.8),
            squeeze=False,
            sharex=True,
            sharey=True,
            layout="constrained",
        )
        for col, (clip, jod) in enumerate(tables.items()):
            for row, t in enumerate(targets):
                ax = axes[row][col]
                route_scatter(ax, jod, t, upscalers, lo, hi)
                if row == 0:
                    ax.set_title(clip)
                if row == len(targets) - 1:
                    ax.set_xlabel(f"{label}, compress first")
                if col == 0:
                    ax.set_ylabel(f"{TARGET_LABELS[t]}\n{label}, TM first")
        fig.suptitle("TM first vs compress first, all clips")
        handles = scatter_legend(upscalers)
        fig.legend(
            handles=handles, loc="outside lower center", ncol=len(handles), frameon=False
        )
        fig.savefig(Path(args.results) / f"all_clips_{stems[2]}.png", dpi=200)
        plt.close(fig)
    print(f"all clips: plots saved in {args.results}", flush=True)


def summary(name, jod, args):
    for upscaler in args.upscalers:
        for t in TARGETS:
            print(
                f"\n{name} {upscaler} {t}: JOD  TM first / compress first (difference)"
            )
            print("         " + "".join(f"{b:>22} Mbps" for b in BITRATES))
            for h in HEIGHTS:
                cells = []
                for b in BITRATES:
                    a = jod.get(
                        (folder_of(upscaler, h), ROUTES[0], t, h, float(b)), np.nan
                    )
                    c = jod.get(
                        (folder_of(upscaler, h), ROUTES[1], t, h, float(b)), np.nan
                    )
                    cells.append(f"{a:6.2f} / {c:5.2f} ({a - c:+.2f})")
                print(f"  {h:>5}p " + "".join(f"{s:>27}" for s in cells))


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "clips", nargs="+", help="Clip names, e.g. returnal_clip_005; scored in turn"
    )
    parser.add_argument(
        "--targets", nargs="+", choices=list(TARGETS), default=list(TARGETS)
    )
    parser.add_argument(
        "--heights", type=int, nargs="+", choices=HEIGHTS, default=list(HEIGHTS)
    )
    parser.add_argument("--bitrates", type=float, nargs="+", default=list(BITRATES))
    parser.add_argument(
        "--upscalers",
        nargs="+",
        choices=UPSCALERS,
        default=list(UPSCALERS),
        help="test_output subfolders to score (2160p always comes from native/)",
    )
    parser.add_argument(
        "--test-output", default=str(TEST_OUTPUT), help="Playback videos to score"
    )
    parser.add_argument(
        "--results", default=str(RESULTS), help="Folder of the CSV and plots"
    )
    parser.add_argument(
        "--frames", type=int, default=-1, help="Frames to score; -1 for all (600)"
    )
    parser.add_argument(
        "--plot-only", action="store_true", help="Only redraw the plots from the CSV"
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Rescore rows already in the CSV"
    )
    args = parser.parse_args()

    out = Path(args.results)
    out.mkdir(exist_ok=True)
    tables = {}
    for clip in args.clips:
        csv_path = out / f"{clip}_cvvdp.csv"
        if not args.plot_only:
            score(clip, csv_path, args)
        rows = [r for r in read_rows(csv_path) if r["frames"] == str(args.frames)]
        if not rows:
            print(f"{clip}: no scores with --frames {args.frames} in {csv_path}")
            continue
        tables[clip] = table(rows)
        summary(clip, tables[clip], args)
    if not tables:
        return
    # The same JOD range in every plot (JOD_RANGE), so all plots compare directly.
    ylim = axis_range([v for jod in tables.values() for v in jod.values()])
    for clip, jod in tables.items():
        plot(clip, jod, args, ylim)
    if len(tables) > 1:
        plot_clips(tables, args, ylim)


if __name__ == "__main__":
    main()
