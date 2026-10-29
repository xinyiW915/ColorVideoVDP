"""Score the playback videos with VMAF and compare the scores with ColorVideoVDP.

Uses the same test/reference pairs as score_cvvdp.py (2160p from native/, scored once).
VMAF runs in ffmpeg's libvmaf, in one of two ways:

  VMAF (default)  the decoded code values of the videos (10-bit PQ for HDR1000/
                  HDR400, 8-bit sRGB for SDR) with two 4K models together:
                    vmaf      vmaf_4k_v0.6.1     the standard model
                    vmaf_neg  vmaf_4k_v0.6.1neg  no enhancement gain (sharpening
                                                 is not rewarded)
                  HDR is outside VMAF's (SDR) training domain.
  PU21-VMAF       (--pu21) as the group's metric_calibration project computes VMAF
                  (cpp_metrics/vmaf_metric.py): the frames are read like score_cvvdp.py
                  does, through the same display models, as pycvvdp's
                  'display_encoded_01' RGB, i.e. PU21 of the display's linear RGB
                  divided by PU21(10000 cd/m^2) for PQ, and the sRGB values
                  unchanged for SDR. That goes to libvmaf as BT.709 YCbCr 4:4:4,
                  10-bit limited range (the project's YUVWriter), scored with
                  vmaf_b_v0.6.3. Its "vmaf" score is the first model of that
                  bootstrap collection, which is what libvmaf reports as "vmaf"
                  for it; ffmpeg cannot load collections, so that model is loaded
                  on its own from vmaf_b_v0.6.3.json (Netflix/vmaf model/).

Either way, VMAF only reads luma and scores are not comparable across targets
(SDR is not PU21-encoded). Scores are the mean over frames.

Results go to scoring_output/<clip>_vmaf.csv (<clip>_pu21_vmaf.csv with --pu21),
one row per test video, written as each score finishes; a rerun skips rows
already scored with the same frame count. The plots are score_cvvdp.py's
all-clips plots with VMAF instead of JOD (same layout, a fixed VMAF axis), to put
side by side with them (pu21_vmaf instead of vmaf in the names with --pu21):

  all_clips_vmaf_vs_bitrate.png      next to all_clips_jod_vs_bitrate.png
  all_clips_vmaf_peak_luminance.png  next to all_clips_peak_luminance.png
  all_clips_vmaf_route_scatter.png   next to all_clips_route_scatter.png

With <clip>_cvvdp.csv of the same frame count, the terminal also shows how VMAF
and JOD agree: their correlation, and their mean differences between the routes
and between the upscalers.
"""

import argparse
import json
import os
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np

from score_cvvdp import (
    BITRATES,
    DISPLAY_FILE,
    HEIGHTS,
    HERE,
    RESULTS,
    ROUTES,
    TARGETS,
    TEST_OUTPUT,
    UPSCALERS,
    axis_range,
    jobs,
    key,
    plot_clips,
    read_rows,
    rel,
)

BOOTSTRAP_MODEL = HERE / "vmaf_b_v0.6.3.json"
# Method -> (file prefix, label, {score name: libvmaf model}). Model files are
# relative to the working directory of ffmpeg.
METHODS = {
    "pq": (
        "vmaf",
        "VMAF",
        {"vmaf": "version=vmaf_4k_v0.6.1", "vmaf_neg": "version=vmaf_4k_v0.6.1neg"},
    ),
    "pu21": ("pu21_vmaf", "PU21-VMAF", {"vmaf": "path=vmaf_b_v0.6.3_0.json"}),
}
# BT.709 RGB -> YCbCr, as metric_calibration's utils/yuv_utils.py.
RGB2YCBCR_709 = [
    [0.2126, 0.7152, 0.0722],
    [-0.114572, -0.385428, 0.5],
    [0.5, -0.454153, -0.045847],
]
FIELDS = [
    "clip",
    "upscaler",
    "route",
    "target",
    "height",
    "mbps",
    "frames",
    "vmaf",
    "vmaf_neg",
    "test",
    "reference",
]
VMAF_RANGE = (20.0, 100.0)  # VMAF axis of every plot, like score_cvvdp's JOD_RANGE


def libvmaf(method):
    """libvmaf filter on [test][ref]. libvmaf takes the distorted video first; the
    log goes to the working directory to avoid escaping a Windows path in the graph."""
    models = "|".join(f"{m}\\:name={n}" for n, m in METHODS[method][2].items())
    return (
        f"[test][ref]libvmaf=model='{models}':n_threads={os.cpu_count()}"
        ":log_fmt=json:log_path=vmaf.json"
    )


def pooled(tmp, method):
    """{score name: mean over frames} from the libvmaf log in tmp, and the frame count."""
    log = json.loads((Path(tmp) / "vmaf.json").read_text())
    means = {n: log["pooled_metrics"][n]["mean"] for n in METHODS[method][2]}
    return means, len(log["frames"])


def pq_vmaf(job, frames):
    """VMAF on the decoded code values of the test video and its reference."""
    # Trim both inputs: -frames:v limits the output and lets libvmaf read one more frame.
    trim = f"trim=end_frame={frames}," if frames > 0 else ""
    graph = (
        f"[0:v]{trim}setpts=PTS-STARTPTS[test];[1:v]{trim}setpts=PTS-STARTPTS[ref];"
        + libvmaf("pq")
    )
    with tempfile.TemporaryDirectory() as tmp:
        cmd = ["ffmpeg", "-hide_banner", "-v", "error", "-nostdin"]
        cmd += ["-i", str(Path(job["test"]).resolve())]
        cmd += ["-i", str(Path(job["reference"]).resolve())]
        subprocess.run(
            cmd + ["-lavfi", graph, "-f", "null", "-"], cwd=tmp, check=True
        )
        return pooled(tmp, "pq")


def pu21_vmaf(job, frames):
    """VMAF as metric_calibration's vmaf_metric.py, on score_cvvdp.py's display models."""
    import torch

    import pycvvdp

    source = pycvvdp.video_source_file(
        str(job["test"]),
        str(job["reference"]),
        display_photometry=TARGETS[job["target"]][0],
        config_paths=[str(DISPLAY_FILE)],
        frames=frames,
    )
    height, width, n = source.get_video_size()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rgb2ycbcr = torch.tensor(RGB2YCBCR_709, device=device)
    # float2fixed in metric_calibration's utils/yuv_utils.py: 10-bit limited range.
    weight = torch.tensor([876.0, 896.0, 896.0], device=device).view(3, 1, 1)
    offset = torch.tensor([64.0, 512.0, 512.0], device=device).view(3, 1, 1)

    def yuv(rgb):
        ycbcr = torch.einsum("ij,jhw->ihw", rgb2ycbcr, rgb[0, :, 0])
        return (weight * ycbcr + offset).round().clamp(0, 1023)

    # Test and reference go side by side through one pipe (Windows has no named
    # pipes for a second input) as planar yuv444p10le, which libvmaf reads without
    # a conversion, and are cropped apart again.
    graph = (
        f"[0:v]split[a][b];[a]crop={width}:{height}:0:0[test];"
        f"[b]crop={width}:{height}:{width}:0[ref];" + libvmaf("pu21")
    )
    with tempfile.TemporaryDirectory() as tmp:
        # ffmpeg cannot load the bootstrap collection; its first model is the one
        # libvmaf reports as "vmaf".
        model = json.loads(BOOTSTRAP_MODEL.read_text())["0"]
        (Path(tmp) / "vmaf_b_v0.6.3_0.json").write_text(json.dumps(model))
        cmd = ["ffmpeg", "-hide_banner", "-v", "error", "-f", "rawvideo"]
        cmd += ["-pix_fmt", "yuv444p10le", "-s", f"{2 * width}x{height}", "-i", "pipe:"]
        process = subprocess.Popen(
            cmd + ["-lavfi", graph, "-f", "null", "-"],
            cwd=tmp,
            stdin=subprocess.PIPE,
        )
        try:
            with torch.no_grad():
                for f in range(n):
                    test = source.get_test_frame(f, device, colorspace="display_encoded_01")
                    ref = source.get_reference_frame(f, device, colorspace="display_encoded_01")
                    planes = torch.cat([yuv(test), yuv(ref)], dim=2)
                    process.stdin.write(planes.to(torch.int16).cpu().numpy().tobytes())
        finally:
            process.stdin.close()
            process.wait()
        if process.returncode:
            raise subprocess.CalledProcessError(process.returncode, cmd)
        return pooled(tmp, "pu21")


def score(clip, csv_path, method, args):
    import csv

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
    print(f"{clip}: {len(done)} already scored; {len(todo)} to score", flush=True)

    run = pu21_vmaf if method == "pu21" else pq_vmaf
    started = time.perf_counter()
    for i, job in enumerate(todo, 1):
        job_started = time.perf_counter()
        means, n = run(job, args.frames)
        row = {
            **job,
            "frames": frames,
            **{name: f"{v:.4f}" for name, v in means.items()},
            "test": rel(job["test"]),
            "reference": rel(job["reference"]),
        }
        with open(csv_path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=FIELDS).writerow(row)
        print(
            f"[{i}/{len(todo)}] {job['upscaler']} {job['route']} {job['target']} "
            f"{job['height']}p {job['mbps']:g}M: "
            + ", ".join(f"{name} {v:.2f}" for name, v in means.items())
            + f" ({n} frames, {time.perf_counter() - job_started:.0f}s)",
            flush=True,
        )
    print(
        f"Scored {len(todo)} videos in {time.perf_counter() - started:.0f}s", flush=True
    )


def load(clip, results, frames, prefix):
    """Test videos with both a JOD and a VMAF score, as dicts."""
    jod = {
        key(r): float(r["jod"])
        for r in read_rows(results / f"{clip}_cvvdp.csv")
        if r["frames"] == frames
    }
    points = []
    for r in read_rows(results / f"{clip}_{prefix}.csv"):
        k = key(r)
        if r["frames"] == frames and k in jod:
            points.append(
                {
                    "clip": clip,
                    **dict(zip(("upscaler", "route", "target", "height", "mbps"), k)),
                    "jod": jod[k],
                    "vmaf": float(r["vmaf"]),
                    # PU21-VMAF has no NEG model.
                    "vmaf_neg": float(r["vmaf_neg"] or "nan"),
                }
            )
    return points


def differences(points):
    """Paired differences: TM first - compress first, and ESRGAN - Lanczos."""
    by = {(p["clip"],) + key(p): p for p in points}
    route, upscaler = [], []

    def delta(a, b):
        return {
            "target": a["target"],
            "height": a["height"],
            **{m: a[m] - b[m] for m in ("jod", "vmaf", "vmaf_neg")},
        }

    for (c, u, r, t, h, b), p in by.items():
        if r == ROUTES[0] and (other := by.get((c, u, ROUTES[1], t, h, b))):
            route.append(delta(p, other))
        if u == UPSCALERS[1] and (other := by.get((c, UPSCALERS[0], r, t, h, b))):
            upscaler.append(delta(p, other))
    return route, upscaler


def correlation(x, y):
    """Pearson r and Spearman rho (ties are rare in these scores)."""
    x, y = np.asarray(x), np.asarray(y)
    if len(x) < 3:
        return np.nan, np.nan
    ranks = lambda a: np.argsort(np.argsort(a))  # noqa: E731
    return np.corrcoef(x, y)[0, 1], np.corrcoef(ranks(x), ranks(y))[0, 1]


def agree(ds, metric="vmaf"):
    """Pairs where VMAF and JOD prefer the same side."""
    return sum(np.sign(d[metric]) == np.sign(d["jod"]) for d in ds)


def summary(name, points, targets, label):
    route, upscaler = differences(points)
    neg = any(np.isfinite(p["vmaf_neg"]) for p in points)
    print(f"\n{name}: {label} vs JOD")
    print(
        "  target    n  Pearson  Spearman |  route dJOD  dVMAF"
        + ("  dNEG" if neg else "")
        + " | ESRGAN dJOD  dVMAF"
        + ("  dNEG  (same sign VMAF/NEG)" if neg else "  (same sign)")
    )
    for t in targets:
        ps = [p for p in points if p["target"] == t]
        r, rho = correlation([p["vmaf"] for p in ps], [p["jod"] for p in ps])
        rd = [d for d in route if d["target"] == t]
        ud = [d for d in upscaler if d["target"] == t]
        mean = lambda ds, m: np.mean([d[m] for d in ds]) if ds else np.nan  # noqa: E731
        print(
            f"  {t:8} {len(ps):3}  {r:7.2f}  {rho:8.2f} |"
            f"  {mean(rd, 'jod'):+10.2f} {mean(rd, 'vmaf'):+6.2f}"
            + (f" {mean(rd, 'vmaf_neg'):+5.2f}" if neg else "")
            + f" | {mean(ud, 'jod'):+11.2f} {mean(ud, 'vmaf'):+6.2f}"
            + (f" {mean(ud, 'vmaf_neg'):+5.2f}" if neg else "")
            + f"  ({agree(ud)}/{len(ud)}"
            + (f", {agree(ud, 'vmaf_neg')}/{len(ud)})" if neg else ")")
        )


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("clips", nargs="+", help="Clip names, e.g. returnal_clip_005")
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
        "--results", default=str(RESULTS), help="Folder of the CSVs and plots"
    )
    parser.add_argument(
        "--frames", type=int, default=-1, help="Frames to score; -1 for all (600)"
    )
    parser.add_argument(
        "--plot-only", action="store_true", help="Only redraw the plots from the CSVs"
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Rescore rows already in the CSV"
    )
    parser.add_argument(
        "--pu21",
        action="store_true",
        help="PU21-VMAF on display-model luminance instead of VMAF on code values",
    )
    args = parser.parse_args()

    out = Path(args.results)
    out.mkdir(exist_ok=True)
    method = "pu21" if args.pu21 else "pq"
    prefix, label, _ = METHODS[method]
    targets = [t for t in TARGETS if t in args.targets]
    tables, everything = {}, []
    for clip in args.clips:
        csv_path = out / f"{clip}_{prefix}.csv"
        if not args.plot_only:
            score(clip, csv_path, method, args)
        rows = [r for r in read_rows(csv_path) if r["frames"] == str(args.frames)]
        if not rows:
            print(f"{clip}: no {label} scores with --frames {args.frames} in {csv_path}")
            continue
        tables[clip] = {key(r): float(r["vmaf"]) for r in rows}
        points = load(clip, out, str(args.frames), prefix)
        if points:
            summary(clip, points, targets, label)
            everything += points
    if len({p["clip"] for p in everything}) > 1:
        summary("all clips", everything, targets, label)
    if not tables:
        return
    ylim = axis_range(
        [v for t in tables.values() for v in t.values()], VMAF_RANGE, 5, label
    )
    plot_clips(
        tables,
        args,
        ylim,
        label=label,
        stems=tuple(
            f"{prefix}_{stem}" for stem in ("vs_bitrate", "peak_luminance", "route_scatter")
        ),
    )


if __name__ == "__main__":
    main()
