"""Encode clips with x265 (H.265) at each resolution x bitrate of the experiment ladder.

Inputs are either original HDR clips (compress_then_tonemap) or tone-mapped
outputs of hdr_tone_mapping/run.py (tonemap_then_compress). The output folder
follows from the input file name:

  compression_output/compress_then_tonemap/                 original clips
  compression_output/tonemap_then_compress/{hdr1000,hdr400,sdr}/  tone-mapped clips

Each output runs two ffmpeg processes connected by a pipe, so no raw YUV file is
written to disk:

  decode -> Lanczos-5 downscale -> raw YUV 4:2:0 -> x265

The encode command follows the supervisors' SVT-AV1 recipe:

  ffmpeg -y -f rawvideo -pixel_format yuv420p -video_size WxH -framerate 60 \\
    -i input.yuv -c:v libsvtav1 -preset 12 -pix_fmt yuv420p -b:v 20M \\
    -svtav1-params 'pred-struct=1:rc=1:keyint=99999' -loglevel error output.mp4

pred-struct=1 (low delay) -> bframes=0; rc=1 (VBR) -> single-pass -b:v (x265
average bitrate); keyint=99999 is kept, with scenecut=0 so x265 inserts no extra
keyframes. x265 has no counterpart to SVT-AV1 preset 12; fast approximates the
hardware encoders used in game streaming. HDR inputs (BT.2020/PQ) are encoded
in 10-bit, SDR inputs (BT.709/sRGB) in 8-bit as in the recipe; both carry their
colour tags. Scaling is done on the limited-range YUV code values.
"""

import argparse
import subprocess
import sys
import time
from fractions import Fraction
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
import ffmpeg
from hdr_tone_mapping.video_io import validate_decode

RESOLUTIONS = {2160: (3840, 2160), 1080: (1920, 1080), 720: (1280, 720)}
BITRATES_MBPS = (5, 12, 25)
FORMATS = {
    "hdr": {
        "pix_fmt": "yuv420p10le",
        "primaries": "bt2020",
        "trc": "smpte2084",
        "matrix": "bt2020nc",
    },
    "sdr": {
        "pix_fmt": "yuv420p",
        "primaries": "bt709",
        "trc": "iec61966-2-1",
        "matrix": "bt709",
    },
}
# hdr_tone_mapping/run.py output name suffix -> tonemap_then_compress subfolder.
TONE_MAPPED = {
    "hdr1000_calibrated": "hdr1000",
    "hdr400_calibrated": "hdr400",
    "sdr100_aces": "sdr",
}


def video_stream(path, **kwargs):
    return next(
        s
        for s in ffmpeg.probe(str(path), **kwargs)["streams"]
        if s["codec_type"] == "video"
    )


def input_format(stream):
    """'hdr' or 'sdr' from the colour tags; only limited-range BT.2020/PQ or BT.709/sRGB."""
    tags = tuple(
        stream.get(k) for k in ("color_primaries", "color_transfer", "color_space")
    )
    for name, fmt in FORMATS.items():
        if stream.get("color_range") == "tv" and tags == (
            fmt["primaries"],
            fmt["trc"],
            fmt["matrix"],
        ):
            return name
    raise ValueError(
        f"Expected limited-range BT.2020/PQ or BT.709/sRGB input, got {tags}"
    )


def output_folder(root, source, fmt):
    for suffix, name in TONE_MAPPED.items():
        if source.stem.endswith("_" + suffix):
            if (name == "sdr") != (fmt == "sdr"):
                raise ValueError(
                    f"{source.name} is named {name} but tagged {fmt.upper()}"
                )
            return root / "tonemap_then_compress" / name
    return root / "compress_then_tonemap"


def decode_command(source, size, source_size, max_frames, pix_fmt):
    command = ["ffmpeg", "-v", "error", "-i", str(source), "-map", "0:v:0"]
    if max_frames is not None:
        command += ["-frames:v", str(max_frames)]
    if size != source_size:
        command += ["-vf", f"scale={size[0]}:{size[1]}:flags=lanczos:param0=5"]
    return command + [
        "-fps_mode",
        "passthrough",
        "-f",
        "rawvideo",
        "-pix_fmt",
        pix_fmt,
        "pipe:",
    ]


def encode_command(target, size, fps, mbps, preset, fmt):
    return [
        "ffmpeg",
        "-y",
        "-f",
        "rawvideo",
        "-pixel_format",
        fmt["pix_fmt"],
        "-video_size",
        f"{size[0]}x{size[1]}",
        "-framerate",
        fps,
        # Set on the raw input so the frames, and thus the HEVC bitstream, carry the tags.
        "-color_range",
        "tv",
        "-color_primaries",
        fmt["primaries"],
        "-color_trc",
        fmt["trc"],
        "-colorspace",
        fmt["matrix"],
        "-i",
        "pipe:",
        "-c:v",
        "libx265",
        "-preset",
        preset,
        "-pix_fmt",
        fmt["pix_fmt"],
        "-b:v",
        f"{mbps:g}M",
        "-x265-params",
        "bframes=0:keyint=99999:scenecut=0",
        "-loglevel",
        "error",
        str(target),
    ]


def encode(decode, encode):
    decoder = subprocess.Popen(decode, stdout=subprocess.PIPE)
    # x265 logs its settings to stderr regardless of -loglevel; show it only on failure.
    encoder = subprocess.Popen(
        encode, stdin=decoder.stdout, stderr=subprocess.PIPE, text=True
    )
    # Let the decoder see a broken pipe if the encoder exits early.
    decoder.stdout.close()
    _, encoder_log = encoder.communicate()
    if decoder.wait() or encoder.returncode:
        raise RuntimeError(f"ffmpeg failed\n{encoder_log}")


def compress(source, args):
    started = time.perf_counter()
    stream = video_stream(source)
    name = input_format(stream)
    fmt = FORMATS[name]
    source_size = (stream["width"], stream["height"])
    fps = stream["r_frame_rate"]
    # Count decoded frames (not the container's nb_frames, which can differ).
    frames = int(video_stream(source, count_frames=None)["nb_read_frames"])
    if args.max_frames is not None:
        frames = min(frames, args.max_frames)

    folder = output_folder(Path(args.output), source, name)
    folder.mkdir(parents=True, exist_ok=True)
    jobs = [
        (height, mbps, folder / f"{source.stem}_{height}p_{mbps:g}M.mp4")
        for height in args.heights
        for mbps in args.bitrates
    ]
    if not args.overwrite and any(target.exists() for *_, target in jobs):
        raise FileExistsError(f"Outputs exist in {folder}; use --overwrite")

    print(
        f"Input: {source} ({source_size[0]}x{source_size[1]}, {fps} fps, {frames} frames, {name.upper()})\n"
        f"Output: {folder}",
        flush=True,
    )
    for height, mbps, target in jobs:
        size = RESOLUTIONS[height]
        partial = target.with_name(target.stem + ".partial.mp4")
        job_started = time.perf_counter()
        print(f"Encoding {target.name} ...", flush=True)
        encode(
            decode_command(source, size, source_size, args.max_frames, fmt["pix_fmt"]),
            encode_command(partial, size, fps, mbps, args.preset, fmt),
        )
        output = video_stream(partial, count_packets=None)
        if (
            int(output["nb_read_packets"]) != frames
            or (output["width"], output["height"]) != size
            or Fraction(output["r_frame_rate"]) != Fraction(fps)
            or output.get("codec_name") != "hevc"
            or output.get("pix_fmt") != fmt["pix_fmt"]
            or output.get("color_transfer") != fmt["trc"]
            or output.get("color_primaries") != fmt["primaries"]
        ):
            raise RuntimeError(f"Validation failed: {partial}")
        validate_decode(partial)
        partial.replace(target)
        actual_mbps = target.stat().st_size * 8 / float(frames / Fraction(fps)) / 1e6
        print(
            f"  {actual_mbps:.2f} Mbps (target {mbps:g}); {time.perf_counter() - job_started:.0f}s",
            flush=True,
        )
    print(
        f"Completed {len(jobs)} encodes in {time.perf_counter() - started:.0f}s",
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "videos",
        nargs="+",
        help="Paths of the video files to encode (original clips or tone-mapped MKVs)",
    )
    parser.add_argument(
        "--output",
        default=str(HERE / "compression_output"),
        help="Root folder; subfolders follow from the input names",
    )
    parser.add_argument(
        "--heights",
        type=int,
        nargs="+",
        choices=sorted(RESOLUTIONS),
        default=sorted(RESOLUTIONS, reverse=True),
    )
    parser.add_argument(
        "--bitrates",
        type=float,
        nargs="+",
        default=list(BITRATES_MBPS),
        help="Target bitrates in Mbps",
    )
    parser.add_argument("--preset", default="fast", help="x265 preset")
    parser.add_argument(
        "--max-frames",
        type=int,
        default=600,
        help="Maximum frames to encode (default 600: the 10 s clip at 60 fps)",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.max_frames <= 0:
        parser.error("--max-frames must be positive")
    sources = [Path(v) for v in args.videos]
    for source in sources:
        if not source.is_file():
            parser.error(f"Video file not found: {source}")

    for source in sources:
        compress(source, args)


if __name__ == "__main__":
    main()
