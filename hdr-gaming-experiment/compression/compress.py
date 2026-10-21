"""Encode an HDR clip with x265 (H.265) at each resolution x bitrate of the experiment ladder.

Each output runs two ffmpeg processes connected by a pipe, so no raw YUV file is
written to disk:

  decode -> Lanczos-5 downscale -> raw yuv420p10le -> x265

The encode command follows:

  ffmpeg -y -f rawvideo -pixel_format yuv420p10le -video_size WxH -framerate 60 \\
    -i pipe: -c:v libx265 -preset fast -pix_fmt yuv420p10le -b:v 20M \\
    -x265-params bframes=0:keyint=99999:scenecut=0 -loglevel error output.mp4

pred-struct=1 (low delay) -> bframes=0; keyint=99999 is kept, with scenecut=0 so
x265 inserts no extra keyframes. x265 has no counterpart to SVT-AV1 preset 12;
fast approximates the hardware encoders used in game streaming. The only
additions are 10-bit instead of 8-bit and BT.2020/PQ colour tags on the raw input, which HDR needs.
Scaling is done on the limited-range BT.2020/PQ YUV code values.
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
PIX_FMT = "yuv420p10le"
# Set on the raw input so the frames, and thus the HEVC bitstream, carry the tags.
COLOR_TAGS = [
    "-color_range", "tv",
    "-color_primaries", "bt2020",
    "-color_trc", "smpte2084",
    "-colorspace", "bt2020nc",
]


def video_stream(path, **kwargs):
    return next(
        s for s in ffmpeg.probe(str(path), **kwargs)["streams"] if s["codec_type"] == "video"
    )


def decode_command(source, size, source_size, max_frames):
    command = ["ffmpeg", "-v", "error", "-i", str(source), "-map", "0:v:0"]
    if max_frames is not None:
        command += ["-frames:v", str(max_frames)]
    if size != source_size:
        command += ["-vf", f"scale={size[0]}:{size[1]}:flags=lanczos:param0=5"]
    return command + [
        "-fps_mode", "passthrough", "-f", "rawvideo", "-pix_fmt", PIX_FMT, "pipe:",
    ]


def encode_command(target, size, fps, mbps, preset):
    return [
        "ffmpeg", "-y",
        "-f", "rawvideo", "-pixel_format", PIX_FMT,
        "-video_size", f"{size[0]}x{size[1]}", "-framerate", fps,
        *COLOR_TAGS,
        "-i", "pipe:",
        "-c:v", "libx265", "-preset", preset, "-pix_fmt", PIX_FMT,
        "-b:v", f"{mbps:g}M",
        "-x265-params", "bframes=0:keyint=99999:scenecut=0",
        "-loglevel", "error", str(target),
    ]


def encode(decode, encode):
    decoder = subprocess.Popen(decode, stdout=subprocess.PIPE)
    # x265 logs its settings to stderr regardless of -loglevel; show it only on failure.
    encoder = subprocess.Popen(encode, stdin=decoder.stdout, stderr=subprocess.PIPE, text=True)
    # Let the decoder see a broken pipe if the encoder exits early.
    decoder.stdout.close()
    _, encoder_log = encoder.communicate()
    if decoder.wait() or encoder.returncode:
        raise RuntimeError(f"ffmpeg failed\n{encoder_log}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("video", help="Path to the HDR video file to encode")
    parser.add_argument("--output", default=str(HERE / "compression_output" / "original"))
    parser.add_argument("--heights", type=int, nargs="+", choices=sorted(RESOLUTIONS), default=sorted(RESOLUTIONS, reverse=True))
    parser.add_argument("--bitrates", type=float, nargs="+", default=list(BITRATES_MBPS), help="Target bitrates in Mbps")
    parser.add_argument("--preset", default="fast", help="x265 preset")
    parser.add_argument("--max-frames", type=int, default=600, help="Maximum frames to encode (default 600: the 10 s clip at 60 fps)")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.max_frames is not None and args.max_frames <= 0:
        parser.error("--max-frames must be positive")
    source = Path(args.video)
    if not source.is_file():
        parser.error(f"Video file not found: {source}")

    started = time.perf_counter()
    stream = video_stream(source)
    if tuple(
        stream.get(k) for k in ("color_transfer", "color_primaries", "color_space", "color_range")
    ) != ("smpte2084", "bt2020", "bt2020nc", "tv"):
        raise ValueError("Expected limited-range BT.2020/PQ input")
    source_size = (stream["width"], stream["height"])
    fps = stream["r_frame_rate"]
    # Count decoded frames (not the container's nb_frames, which can differ).
    frames = int(video_stream(source, count_frames=None)["nb_read_frames"])
    if args.max_frames is not None:
        frames = min(frames, args.max_frames)

    folder = Path(args.output)
    folder.mkdir(parents=True, exist_ok=True)
    jobs = [
        (height, mbps, folder / f"{source.stem}_{height}p_{mbps:g}M.mp4")
        for height in args.heights
        for mbps in args.bitrates
    ]
    if not args.overwrite and any(target.exists() for *_, target in jobs):
        raise FileExistsError("Outputs exist; use --overwrite")

    print(f"Input: {source} ({source_size[0]}x{source_size[1]}, {fps} fps, {frames} frames)", flush=True)
    for height, mbps, target in jobs:
        size = RESOLUTIONS[height]
        partial = target.with_name(target.stem + ".partial.mp4")
        job_started = time.perf_counter()
        print(f"Encoding {target.name} ...", flush=True)
        encode(
            decode_command(source, size, source_size, args.max_frames),
            encode_command(partial, size, fps, mbps, args.preset),
        )
        output = video_stream(partial, count_packets=None)
        if (
            int(output["nb_read_packets"]) != frames
            or (output["width"], output["height"]) != size
            or Fraction(output["r_frame_rate"]) != Fraction(fps)
            or output.get("codec_name") != "hevc"
            or output.get("pix_fmt") != PIX_FMT
            or output.get("color_transfer") != "smpte2084"
            or output.get("color_primaries") != "bt2020"
        ):
            raise RuntimeError(f"Validation failed: {partial}")
        validate_decode(partial)
        partial.replace(target)
        actual_mbps = target.stat().st_size * 8 / float(frames / Fraction(fps)) / 1e6
        print(
            f"  {actual_mbps:.2f} Mbps (target {mbps:g}); {time.perf_counter() - job_started:.0f}s",
            flush=True,
        )
    print(f"Completed {len(jobs)} encodes in {time.perf_counter() - started:.0f}s", flush=True)


if __name__ == "__main__":
    main()
