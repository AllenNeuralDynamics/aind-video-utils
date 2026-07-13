"""CLI for generating video QC plots."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> None:
    """Entry point for aind-video-qc CLI."""
    parser = argparse.ArgumentParser(
        prog="aind-video-qc",
        description="Generate video quality-control plots.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # -- linear-to-bt709 subcommand --
    p_compare = subparsers.add_parser(
        "linear-to-bt709",
        help="Compare linear-light input against BT.709-encoded output.",
    )
    p_compare.add_argument("input_video", type=Path, help="Path to the input video.")
    p_compare.add_argument("output_video", type=Path, help="Path to the output video.")
    p_compare.add_argument("--frame-time", type=float, default=0, help="Time in seconds (default: 0).")
    p_compare.add_argument(
        "--no-coerce",
        dest="coerce",
        action="store_false",
        help="Trust the input's color-space tags instead of assuming linear light.",
    )
    p_compare.add_argument(
        "--range-override",
        choices=["pc", "tv"],
        default=None,
        help="Authoritative input range ('pc'=full, 'tv'=limited); overrides source tags.",
    )
    p_compare.add_argument("--rig-group", default=None, help="Rig group label shown in the summary table.")
    p_compare.add_argument(
        "--noise-frames",
        type=int,
        default=8,
        help="Number of frames sampled for the noise-transfer gamma test (default: 8).",
    )
    p_compare.add_argument("--output", "-o", type=Path, default=None, help="Output PNG path.")
    p_compare.add_argument("--dpi", type=int, default=180, help="Output DPI (default: 180).")

    # -- opencv subcommand --
    p_opencv = subparsers.add_parser(
        "opencv",
        help="Compare ffmpeg luma extraction with OpenCV decode.",
    )
    p_opencv.add_argument("input_video", type=Path, help="Path to the input video.")
    p_opencv.add_argument("--frame-time", type=float, default=0, help="Time in seconds (default: 0).")
    p_opencv.add_argument("--output", "-o", type=Path, default=None, help="Output PNG path.")
    p_opencv.add_argument("--dpi", type=int, default=180, help="Output DPI (default: 180).")

    # -- color-range subcommand --
    p_color = subparsers.add_parser(
        "color-range",
        help="Check if pixel data matches color range metadata.",
    )
    p_color.add_argument("input_video", type=Path, help="Path to the input video.")
    p_color.add_argument("--frame-time", type=float, default=0, help="Time in seconds (default: 0).")
    p_color.add_argument("--output", "-o", type=Path, default=None, help="Output PNG path.")
    p_color.add_argument("--dpi", type=int, default=180, help="Output DPI (default: 180).")

    args = parser.parse_args()

    try:
        import matplotlib

        matplotlib.use("Agg")  # save-to-PNG CLI: never require a display (runs headless on EC2)
        from aind_video_utils.video_qc import (
            check_color_range,
            compare_luma_opencv_frames,
            transcode_qc_figure,
        )
    except ImportError:
        print("Error: plotting dependencies not installed. Run: pip install aind-video-utils[plotting]", file=sys.stderr)
        sys.exit(1)

    if args.command == "linear-to-bt709":
        output_path = args.output or Path(f"{args.input_video.stem}_qc.png")
        fig = transcode_qc_figure(
            args.input_video,
            args.output_video,
            args.frame_time,
            range_override=args.range_override,
            rig_group=args.rig_group,
            noise_frames=args.noise_frames,
            coerce_input_color_space=args.coerce,
        )
    elif args.command == "opencv":
        output_path = args.output or Path(f"{args.input_video.stem}_opencv_qc.png")
        fig = compare_luma_opencv_frames(
            args.input_video,
            frame_time=args.frame_time,
        )
    else:  # color-range
        output_path = args.output or Path(f"{args.input_video.stem}_color_range_qc.png")
        fig = check_color_range(
            args.input_video,
            frame_time=args.frame_time,
        )

    fig.savefig(output_path, dpi=args.dpi, bbox_inches="tight")
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
