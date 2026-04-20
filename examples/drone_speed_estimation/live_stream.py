"""
Minimal launcher for the drone vehicle speed estimation pipeline.

No jsonargparse required. Run directly:

    python live_stream.py                          # prompts for RTMP and WS URLs
    python live_stream.py --rtmp <URL> --ws <URL>  # pass URLs directly
    python live_stream.py --device mps --weights yolo26s.pt
    python live_stream.py --output output.mp4 --no-preview
    python live_stream.py --no-calib    # force GSD-only mode

RTMP stream URL and WebSocket URL are prompted interactively if not supplied
via --rtmp / --ws. You can also set them as environment variables:
    DRONE_RTMP  and  DRONE_WS

When calibration_auto.json is present in the same directory it is loaded
automatically; pass --no-calib to disable it and fall back to GSD mode.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Ensure sibling modules (drone_speed, speed_utils, dji_telemetry, …) are importable
# regardless of where the script is invoked from.
sys.path.insert(0, str(Path(__file__).parent))

from drone_speed import run  # noqa: E402

DEFAULT_WEIGHTS = "yolo26s.pt"


def _prompt_url(label: str, env_var: str, flag: str) -> str:
    """Read a URL from env var, then fall back to an interactive prompt.

    Args:
        label: Human-readable name shown in the prompt (e.g. ``"RTMP stream"``).
        env_var: Environment variable name to check first.
        flag: CLI flag name shown in the hint (e.g. ``"--rtmp"``).

    Returns:
        The URL string entered by the user (stripped of whitespace).
    """
    val = os.environ.get(env_var, "").strip()
    if val:
        print(f"[live_stream] {label} URL from env {env_var}: {val}")
        return val
    print(f"  (tip: set ${env_var} or pass {flag} <URL> to skip this prompt)")
    while True:
        val = input(f"Enter {label} URL: ").strip()
        if val:
            return val
        print("  URL cannot be empty, please try again.")

_CALIB_FILE = Path(__file__).parent / "calibration_auto.json"


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Drone vehicle speed estimation — live RTMP stream",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Stream endpoints ───────────────────────────────────────────────────
    p.add_argument(
        "--rtmp",
        default=None,
        metavar="URL",
        help="RTMP stream URL (prompted interactively if omitted; or set $DRONE_RTMP)",
    )
    p.add_argument(
        "--ws",
        default=None,
        metavar="URL",
        help="DJI Cloud API WebSocket URL (prompted if omitted; or set $DRONE_WS)",
    )

    # ── Model ──────────────────────────────────────────────────────────────
    p.add_argument(
        "--weights",
        default=DEFAULT_WEIGHTS,
        metavar="FILE",
        help="YOLO weights file (e.g. yolov8n.pt, yolo11x.pt, yolo26s.pt)",
    )
    p.add_argument(
        "--device",
        default=None,
        metavar="DEV",
        help="Inference device: cpu | cuda:0 | mps | None (auto)",
    )
    p.add_argument(
        "--imgsz",
        type=int,
        default=1536,
        metavar="N",
        help="YOLO inference image size in pixels",
    )
    p.add_argument(
        "--confidence",
        type=float,
        default=0.10,
        metavar="F",
        help="Detection confidence threshold (0–1)",
    )
    p.add_argument(
        "--iou",
        type=float,
        default=0.70,
        metavar="F",
        help="NMS IoU threshold (0–1)",
    )

    # ── Processing ─────────────────────────────────────────────────────────
    p.add_argument(
        "--stride",
        type=int,
        default=1,
        metavar="N",
        help="Run YOLO every N frames (display always at full stream rate)",
    )
    p.add_argument(
        "--speed-window",
        type=float,
        default=1.0,
        metavar="SEC",
        help="Sliding window in seconds for speed smoothing",
    )
    p.add_argument(
        "--hfov",
        type=float,
        default=84.0,
        metavar="DEG",
        help="Camera base horizontal FOV in degrees (DJI Mavic 3 wide ≈ 84°)",
    )
    p.add_argument(
        "--zoom",
        type=float,
        default=1.0,
        metavar="X",
        help="Manual zoom override (1.0 = use telemetry value)",
    )

    # ── Calibration ────────────────────────────────────────────────────────
    p.add_argument(
        "--calib",
        default=None,
        metavar="FILE",
        help=(
            "Path to calibration JSON (overrides auto-detected calibration_auto.json). "
            "Omit to use calibration_auto.json if present."
        ),
    )
    p.add_argument(
        "--no-calib",
        action="store_true",
        help="Disable calibration entirely; use GSD formula only",
    )
    p.add_argument(
        "--auto-calib",
        action="store_true",
        help="Run automatic lane calibration from the first ~90 frames then start",
    )
    p.add_argument(
        "--lane-width",
        type=float,
        default=3.5,
        metavar="M",
        help="Lane width in metres used when --auto-calib is set",
    )
    p.add_argument(
        "--stripe-width",
        type=float,
        default=0.40,
        metavar="M",
        help=(
            "Crosswalk stripe width in metres for intersection auto-calibration "
            "(China standard = 0.40 m, Europe = 0.50 m)"
        ),
    )

    # ── Output ─────────────────────────────────────────────────────────────
    p.add_argument(
        "--output",
        default=None,
        metavar="FILE",
        help="Optional path to write annotated video (e.g. output.mp4)",
    )
    p.add_argument(
        "--no-preview",
        action="store_true",
        help="Suppress the OpenCV display window (useful for headless recording)",
    )

    # ── Reconnection ───────────────────────────────────────────────────────
    p.add_argument(
        "--max-reconnects",
        type=int,
        default=10,
        metavar="N",
        help="Maximum RTMP reconnection attempts before giving up",
    )
    p.add_argument(
        "--reconnect-delay",
        type=float,
        default=2.0,
        metavar="SEC",
        help="Seconds between RTMP reconnection attempts",
    )

    # ── Altitude drift ─────────────────────────────────────────────────────
    p.add_argument(
        "--altitude-tolerance",
        type=float,
        default=8.0,
        metavar="M",
        help=(
            "Auto-disable lane calibration when altitude deviates more than "
            "this many metres from calibration altitude (default: 8 m). "
            "Set to 0 to disable drift detection."
        ),
    )

    # ── Tracking ───────────────────────────────────────────────────────────
    p.add_argument(
        "--lost-track-buffer",
        type=int,
        default=90,
        metavar="N",
        help=(
            "ByteTrack: keep a lost track alive for N frames before dropping it. "
            "Higher values reduce ID switching through brief occlusions "
            "(default: 90 ≈ ~3 s at 30 fps inference, or longer at lower fps)."
        ),
    )
    p.add_argument(
        "--min-track-hits",
        type=int,
        default=2,
        metavar="N",
        help=(
            "ByteTrack: minimum consecutive detection hits before confirming a "
            "new track (default: 2, reduces one-frame false positives)."
        ),
    )

    return p


def main() -> None:
    """Entry point for the live drone speed estimation pipeline.

    Example::

        # Quickest start with default settings
        python live_stream.py

        # Record to file headlessly
        python live_stream.py --output out.mp4 --no-preview --device mps

        # Force GSD-only mode (no calibration file)
        python live_stream.py --no-calib
    """
    args = _build_parser().parse_args()

    # ── Resolve stream URLs (prompt if not supplied) ───────────────────────
    rtmp_url = args.rtmp or _prompt_url("RTMP stream", "DRONE_RTMP", "--rtmp")
    ws_url = args.ws or _prompt_url("WebSocket", "DRONE_WS", "--ws")

    # Resolve calibration path:
    # 1. --no-calib flag → None
    # 2. explicit --calib path → use it
    # 3. calibration_auto.json present in script directory → use it
    # 4. nothing found → None (GSD fallback)
    if args.no_calib:
        calib_path = None
    elif args.calib is not None:
        calib_path = args.calib
    elif _CALIB_FILE.exists():
        calib_path = str(_CALIB_FILE)
        print(f"[live_stream] using calibration: {_CALIB_FILE.name}")
    else:
        calib_path = None
        print("[live_stream] no calibration file found — running in GSD mode")

    run(
        rtmp_url=rtmp_url,
        ws_url=ws_url,
        weights=args.weights,
        camera_hfov=args.hfov,
        zoom_factor=args.zoom,
        calibration=calib_path,
        auto_calibrate_lane=args.auto_calib,
        lane_width_m=args.lane_width,
        confidence=args.confidence,
        iou_threshold=args.iou,
        frame_stride=args.stride,
        target_video=args.output,
        device=args.device,
        max_reconnects=args.max_reconnects,
        reconnect_delay=args.reconnect_delay,
        speed_window=args.speed_window,
        show_preview=not args.no_preview,
        imgsz=args.imgsz,
        altitude_tolerance=args.altitude_tolerance,
        crosswalk_stripe_width_m=args.stripe_width,
        lost_track_buffer=args.lost_track_buffer,
        minimum_consecutive_frames=args.min_track_hits,
    )


if __name__ == "__main__":
    main()
