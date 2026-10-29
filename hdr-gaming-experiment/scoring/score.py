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
The plots are drawn from the CSV, one set per upscaler:

  <clip>_<upscaler>_jod_vs_bitrate.png     JOD vs bitrate, both routes
  <clip>_<upscaler>_route_difference.png   JOD(tonemap_then_compress) - JOD(compress_then_tonemap)
  <clip>_<upscaler>_route_scatter.png      the same comparison as a scatter plot
  <clip>_<upscaler>_peak_luminance.png     JOD vs peak luminance (SDR 100 / HDR400 / HDR1000)

With both upscalers, two more plots put them side by side (a row per route):

  <clip>_upscaler_comparison.png   JOD vs bitrate, Lanczos-5 and Real-ESRGAN together
  <clip>_upscaler_difference.png   JOD(esrgan) - JOD(lanczos) at 720p and 1080p
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
# Route -> (line style, filled marker, label).
ROUTE_STYLE = {
    "tonemap_then_compress": ("-", True, "TM first"),
    "compress_then_tonemap": ("--", False, "Compress first"),
}
# Upscaler -> (line style, filled marker), for the plots that compare upscalers.
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
SAME_BAND = (
    0.2  # +/- JOD drawn as "too small to notice" in the route plots; a guide only
)


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
    lookup = lambda u, r, t, h, b: jod.get((folder_of(u, h), r, t, h, float(b)), np.nan)  # noqa: E731
    get = lambda r, t, h, b: lookup(upscaler, r, t, h, b)  # noqa: E731
    targets = [t for t in TARGETS if t in args.targets]
    out = Path(args.results)
    x = range(len(BITRATES))
    bitrate_labels = [f"{b}" for b in BITRATES]
    resolutions = [
        Line2D([], [], color=c, marker="o", label=f"{h}p")
        for h, c in RES_COLORS.items()
    ]
    routes = [
        Line2D(
            [],
            [],
            color="black",
            linestyle=ls,
            marker="o",
            markeredgewidth=1.5,
            markerfacecolor="black" if filled else "white",
            label=label,
        )
        for ls, filled, label in ROUTE_STYLE.values()
    ]

    def figure(nrows, ncols, title, width=3.6, height=3.0, suffix=None, **kwargs):
        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(width * ncols, height * nrows + 0.6),
            squeeze=False,
            layout="constrained",
            **kwargs,
        )
        fig.suptitle(f"{title} ({suffix or UPSCALER_LABELS.get(upscaler, upscaler)})")
        return fig, axes

    def save(fig, handles, name, per_upscaler=True):
        fig.legend(
            handles=handles,
            loc="outside lower center",
            ncol=len(handles),
            frameon=False,
        )
        stem = f"{clip}_{upscaler}_{name}" if per_upscaler else f"{clip}_{name}"
        fig.savefig(out / f"{stem}.png", dpi=200)
        plt.close(fig)

    with plt.rc_context(RC):
        for upscaler in args.upscalers:
            # 1. JOD vs bitrate.
            fig, axes = figure(1, len(targets), "JOD vs bitrate", sharey=True)
            for ax, t in zip(axes[0], targets):
                for route, (ls, filled, _) in ROUTE_STYLE.items():
                    for h, c in RES_COLORS.items():
                        draw(
                            ax,
                            x,
                            [get(route, t, h, b) for b in BITRATES],
                            c,
                            ls,
                            filled,
                        )
                ax.set_title(TARGET_LABELS[t])
                categories(ax, bitrate_labels, "Bitrate (Mbps)")
            axes[0][0].set_ylabel("JOD")
            save(fig, resolutions + routes, "jod_vs_bitrate")

            # 2. Route difference.
            diff = {
                (t, h, b): get(ROUTES[0], t, h, b) - get(ROUTES[1], t, h, b)
                for t in targets
                for h in HEIGHTS
                for b in BITRATES
            }
            span = max(
                SAME_BAND * 1.5,
                max((abs(v) for v in diff.values() if np.isfinite(v)), default=0)
                * 1.15,
            )
            fig, axes = figure(
                1, len(targets), "TM first − compress first", sharey=True
            )
            for ax, t in zip(axes[0], targets):
                ax.axhspan(-SAME_BAND, SAME_BAND, color="#eeeeee", linewidth=0)
                ax.axhline(0, color="black", linewidth=0.8)
                for h, c in RES_COLORS.items():
                    draw(ax, x, [diff[t, h, b] for b in BITRATES], c)
                ax.set_title(TARGET_LABELS[t])
                ax.set_ylim(-span, span)
                categories(ax, bitrate_labels, "Bitrate (Mbps)")
            axes[0][0].set_ylabel("ΔJOD")
            save(
                fig,
                resolutions + [Patch(color="#eeeeee", label=f"±{SAME_BAND} JOD")],
                "route_difference",
            )

            # 2b. Route scatter.
            values = [v for v in jod.values() if np.isfinite(v)]
            lo = (min(values) if values else 8) - 0.1
            hi = min(10, (max(values) if values else 10) + 0.1)
            fig, axes = figure(
                1,
                len(targets),
                "TM first vs compress first",
                height=4.4,
                sharex=True,
                sharey=True,
            )
            for ax, t in zip(axes[0], targets):
                ax.grid(True, axis="both")
                ax.plot([lo, hi], [lo, hi], color="black", linewidth=0.8)
                for h, c in RES_COLORS.items():
                    for b, m in BITRATE_MARKERS.items():
                        ax.plot(
                            get(ROUTES[1], t, h, b),
                            get(ROUTES[0], t, h, b),
                            linestyle="",
                            marker=m,
                            markersize=7,
                            color=c,
                        )
                ax.set(
                    title=TARGET_LABELS[t],
                    xlabel="JOD, compress first",
                    xlim=(lo, hi),
                    ylim=(lo, hi),
                    aspect="equal",
                )
            axes[0][0].set_ylabel("JOD, TM first")
            markers = [
                Line2D([], [], color="black", linestyle="", marker=m, label=f"{b} Mbps")
                for b, m in BITRATE_MARKERS.items()
            ]
            save(
                fig,
                [
                    Line2D([], [], color=c, linestyle="", marker="o", label=f"{h}p")
                    for h, c in RES_COLORS.items()
                ]
                + markers,
                "route_scatter",
            )

            # 3. Peak luminance.
            by_peak = sorted(targets, key=lambda t: TARGETS[t][1])
            fig, axes = figure(
                len(ROUTES),
                len(BITRATES),
                "JOD vs peak luminance",
                sharex=True,
                sharey=True,
            )
            for row, route in enumerate(ROUTES):
                for col, b in enumerate(BITRATES):
                    ax = axes[row][col]
                    for h, c in RES_COLORS.items():
                        draw(
                            ax,
                            range(len(by_peak)),
                            [get(route, t, h, b) for t in by_peak],
                            c,
                        )
                    categories(
                        ax,
                        [f"{TARGET_LABELS[t]}\n{TARGETS[t][1]} nits" for t in by_peak],
                        "Peak luminance" if row == len(ROUTES) - 1 else "",
                    )
                    if row == 0:
                        ax.set_title(f"{b} Mbps")
                axes[row][0].set_ylabel(f"{ROUTE_STYLE[route][2]}\nJOD")
            save(fig, resolutions, "peak_luminance")

        if all(u in args.upscalers for u in UPSCALERS):
            pair = " vs ".join(UPSCALER_LABELS[u] for u in UPSCALERS)
            upscaled = [h for h in HEIGHTS if h != 2160]
            last = len(ROUTES) - 1

            # 4. Upscaler comparison: 2160p is not upscaled, so it is one shared line.
            fig, axes = figure(
                len(ROUTES),
                len(targets),
                "JOD vs bitrate",
                suffix=pair,
                sharex=True,
                sharey=True,
            )
            for row, route in enumerate(ROUTES):
                for col, t in enumerate(targets):
                    ax = axes[row][col]
                    draw(
                        ax,
                        x,
                        [lookup(NATIVE, route, t, 2160, b) for b in BITRATES],
                        RES_COLORS[2160],
                        ":",
                    )
                    for u, (ls, filled) in UPSCALER_STYLE.items():
                        for h in upscaled:
                            draw(
                                ax,
                                x,
                                [lookup(u, route, t, h, b) for b in BITRATES],
                                RES_COLORS[h],
                                ls,
                                filled,
                            )
                    if row == 0:
                        ax.set_title(TARGET_LABELS[t])
                    categories(
                        ax, bitrate_labels, "Bitrate (Mbps)" if row == last else ""
                    )
                axes[row][0].set_ylabel(f"{ROUTE_STYLE[route][2]}\nJOD")
            handles = [
                Line2D([], [], color=RES_COLORS[h], marker="o", label=f"{h}p")
                for h in upscaled
            ]
            handles.append(
                Line2D(
                    [],
                    [],
                    color=RES_COLORS[2160],
                    linestyle=":",
                    marker="o",
                    label="2160p (not upscaled)",
                )
            )
            handles += [
                Line2D(
                    [],
                    [],
                    color="black",
                    linestyle=ls,
                    marker="o",
                    markeredgewidth=1.5,
                    markerfacecolor="black" if filled else "white",
                    label=UPSCALER_LABELS[u],
                )
                for u, (ls, filled) in UPSCALER_STYLE.items()
            ]
            save(fig, handles, "upscaler_comparison", per_upscaler=False)

            # 5. Upscaler difference.
            base, other = UPSCALERS
            diff = {
                (r, t, h, b): lookup(other, r, t, h, b) - lookup(base, r, t, h, b)
                for r in ROUTES
                for t in targets
                for h in upscaled
                for b in BITRATES
            }
            span = max(
                SAME_BAND * 1.5,
                max((abs(v) for v in diff.values() if np.isfinite(v)), default=0)
                * 1.15,
            )
            fig, axes = figure(
                len(ROUTES),
                len(targets),
                "Upscaler difference",
                suffix=f"{UPSCALER_LABELS[other]} − {UPSCALER_LABELS[base]}",
                sharex=True,
                sharey=True,
            )
            for row, route in enumerate(ROUTES):
                for col, t in enumerate(targets):
                    ax = axes[row][col]
                    ax.axhspan(-SAME_BAND, SAME_BAND, color="#eeeeee", linewidth=0)
                    ax.axhline(0, color="black", linewidth=0.8)
                    for h in upscaled:
                        draw(
                            ax,
                            x,
                            [diff[route, t, h, b] for b in BITRATES],
                            RES_COLORS[h],
                        )
                    if row == 0:
                        ax.set_title(TARGET_LABELS[t])
                    ax.set_ylim(-span, span)
                    categories(
                        ax, bitrate_labels, "Bitrate (Mbps)" if row == last else ""
                    )
                axes[row][0].set_ylabel(f"{ROUTE_STYLE[route][2]}\nΔJOD")
            save(
                fig,
                handles[: len(upscaled)]
                + [Patch(color="#eeeeee", label=f"±{SAME_BAND} JOD")],
                "upscaler_difference",
                per_upscaler=False,
            )
    print(f"Plots saved in {out}", flush=True)


def summary(rows, args):
    jod = table(rows)
    for upscaler in args.upscalers:
        for t in TARGETS:
            print(f"\n{upscaler} {t}: JOD  TM first / compress first (difference)")
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
    parser.add_argument("clip", help="Clip name, e.g. returnal_clip_005")
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
    csv_path = out / f"{args.clip}_cvvdp.csv"
    if not args.plot_only:
        score(args.clip, csv_path, args)
    rows = [r for r in read_rows(csv_path) if r["frames"] == str(args.frames)]
    if not rows:
        raise SystemExit(f"No scores with --frames {args.frames} in {csv_path}")
    summary(rows, args)
    plot(args.clip, rows, args)


if __name__ == "__main__":
    main()
