"""
Interactive lane-marking calibration tool.

Captures one frame from the RTMP stream, lets the user click **4 corners**
of a road section whose real-world dimensions are known, and saves a
``calibration.json`` that ``drone_speed.py`` loads for accurate speed
estimation.

Usage
-----
Step 1 — capture a frame and calibrate::

    python calibrate_lane.py \\
        --rtmp_url rtmp://aplay.wucepro.com/live/1581F8HGX25CC00A17UC-99-0-0 \\
        --real_width 7.0 \\
        --real_length 20.0 \\
        --output calibration.json

    # --real_width  : real-world width  of the selected rectangle in metres
    #                 e.g. 3.5 = one lane, 7.0 = two lanes, 10.5 = three lanes
    # --real_length : real-world length of the selected rectangle in metres
    #                 (pick a road section spanning ~10–30 m for best accuracy)

Step 2 — run speed estimation with calibration::

    python drone_speed.py \\
        --rtmp_url rtmp://... \\
        --ws_url   ws://... \\
        --weights  yolo26s.pt \\
        --calibration calibration.json

How to click the 4 corners
--------------------------
Select the corners of a **flat road section** in this exact order:

    top-left (far-left)  →  top-right (far-right)
                                    ↓
    bottom-left (near-left) ← bottom-right (near-right)

    ┌───────────────────────────────┐
    │  1 (TL) ──────────── 2 (TR)  │  ← far edge of your chosen section
    │   │                   │       │
    │  4 (BL) ──────────── 3 (BR)  │  ← near edge of your chosen section
    └───────────────────────────────┘

Tips
----
* Use **lane-dividing lines** as your reference — their width is precisely
  3.5 m (urban/national road) or 3.75 m (highway) in China.
* Zoom the display with the ``--display_scale`` flag if the stream is large.
* Press ``r`` to reset and start over; ``q`` or Esc to quit without saving.
* The road section does NOT need to be a rectangle in the image — perspective
  distortion is handled automatically by the homography.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np


def _open_stream(url: str, max_retries: int = 10, retry_delay: float = 2.0) -> cv2.VideoCapture:
    for attempt in range(1, max_retries + 1):
        cap = cv2.VideoCapture(url)
        if cap.isOpened():
            return cap
        print(f"[calibrate] attempt {attempt}/{max_retries} failed, retrying…")
        cap.release()
        time.sleep(retry_delay)
    raise RuntimeError(f"Cannot connect to stream: {url}")


def _grab_frame(url: str) -> np.ndarray:
    """Read one frame from the stream (or a local image/video file)."""
    # Allow a local image path as shortcut
    p = Path(url)
    if p.exists() and p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp"):
        img = cv2.imread(str(p))
        if img is not None:
            return img

    cap = _open_stream(url)
    for _ in range(5):          # skip a few frames to avoid black/compressed artefacts
        ret, frame = cap.read()
    cap.release()
    if not ret or frame is None:
        raise RuntimeError("Could not read a frame from the stream.")
    return frame


def run_calibration(
    rtmp_url: str,
    real_width: float,
    real_length: float,
    output: str = "calibration.json",
    display_scale: float = 1.0,
) -> None:
    """Interactive 4-point calibration.

    Args:
        rtmp_url: RTMP stream URL or path to a local image/video file.
        real_width: Real-world width of the selected rectangle in metres.
        real_length: Real-world length of the selected rectangle in metres.
        output: Path to write ``calibration.json``.
        display_scale: Resize the displayed image by this factor (< 1 to shrink).
    """
    print("[calibrate] grabbing frame…")
    frame = _grab_frame(rtmp_url)
    h, w = frame.shape[:2]

    disp_w = int(w * display_scale)
    disp_h = int(h * display_scale)

    clicks: list[list[int]] = []   # pixel coords in original resolution

    COLORS = [
        (0,   255,  0  ),   # TL  green
        (0,   200,  255),   # TR  cyan
        (0,   0,    255),   # BR  red
        (255, 128,  0  ),   # BL  orange
    ]
    LABELS = ["1-TL (far-left)", "2-TR (far-right)", "3-BR (near-right)", "4-BL (near-left)"]

    def draw(img: np.ndarray) -> np.ndarray:
        vis = img.copy()
        for i, pt in enumerate(clicks):
            px = int(pt[0] * display_scale)
            py = int(pt[1] * display_scale)
            cv2.circle(vis, (px, py), 8, COLORS[i], -1)
            cv2.putText(vis, LABELS[i], (px + 10, py - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLORS[i], 2, cv2.LINE_AA)
        if len(clicks) > 1:
            pts = [(int(p[0]*display_scale), int(p[1]*display_scale)) for p in clicks]
            for i in range(len(pts) - 1):
                cv2.line(vis, pts[i], pts[i+1], (255, 255, 0), 2)
            if len(clicks) == 4:
                cv2.line(vis, pts[3], pts[0], (255, 255, 0), 2)
        remaining = 4 - len(clicks)
        if remaining > 0:
            msg = f"Click {LABELS[len(clicks)]}   ({remaining} remaining)"
        else:
            msg = f"Press ENTER to save  |  r = reset  |  q = quit"
        cv2.putText(vis, msg, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(vis, f"Real: {real_width} m wide  x  {real_length} m long",
                    (12, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 200), 2, cv2.LINE_AA)
        return vis

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(clicks) < 4:
            orig_x = int(x / display_scale)
            orig_y = int(y / display_scale)
            clicks.append([orig_x, orig_y])
            cv2.imshow("Lane Calibration", draw(cv2.resize(frame, (disp_w, disp_h))))

    display = cv2.resize(frame, (disp_w, disp_h))
    cv2.imshow("Lane Calibration", draw(display))
    cv2.setMouseCallback("Lane Calibration", on_mouse)

    print("[calibrate] click 4 corners in order: TL → TR → BR → BL")
    print("[calibrate] press ENTER to save, r to reset, q/Esc to quit")

    while True:
        key = cv2.waitKey(20) & 0xFF
        cv2.imshow("Lane Calibration", draw(cv2.resize(frame, (disp_w, disp_h))))

        if key in (ord("q"), 27):
            print("[calibrate] cancelled.")
            cv2.destroyAllWindows()
            return

        if key == ord("r"):
            clicks.clear()
            print("[calibrate] reset — click again from TL")

        if key == 13 and len(clicks) == 4:   # Enter
            break

    cv2.destroyAllWindows()

    if len(clicks) != 4:
        print("[calibrate] need exactly 4 clicks — aborted.")
        return

    # Build target rectangle in real-world metres
    # TL=0, TR=1, BR=2, BL=3  (matching click order)
    target = [
        [0.0,          0.0         ],   # TL
        [real_width,   0.0         ],   # TR
        [real_width,   real_length ],   # BR
        [0.0,          real_length ],   # BL
    ]

    calib = {
        "source_pixels":    clicks,
        "target_meters":    target,
        "real_width_m":     real_width,
        "real_length_m":    real_length,
        "frame_size":       [w, h],
        "description": (
            f"{real_width} m wide × {real_length} m long road section, "
            f"calibrated from {rtmp_url}"
        ),
    }

    Path(output).write_text(json.dumps(calib, indent=2))
    print(f"[calibrate] saved → {output}")
    print(f"  source_pixels: {clicks}")
    print(f"  target_meters: {target}")
    print(f"\nRun speed estimation with:  --calibration {output}")


def main(
    rtmp_url: str,
    real_width: float = 7.0,
    real_length: float = 20.0,
    output: str = "calibration.json",
    display_scale: float = 1.0,
) -> None:
    """Interactive lane calibration tool.

    Args:
        rtmp_url: RTMP stream URL (or path to a saved frame image).
        real_width: Real width of the road section in metres.
            Common values: 3.5 (one urban lane), 7.0 (two lanes),
            3.75 (one highway lane), 10.5 (three lanes).
        real_length: Real length of the road section in metres.
            Pick a distance you can estimate visually, e.g. 15–30 m.
        output: Path to write the calibration JSON file.
        display_scale: Scale factor for the calibration window (try 0.5 for
            large 1080p/4K streams on a small screen).
    """
    run_calibration(
        rtmp_url=rtmp_url,
        real_width=real_width,
        real_length=real_length,
        output=output,
        display_scale=display_scale,
    )


if __name__ == "__main__":
    from jsonargparse import auto_cli, set_parsing_settings
    set_parsing_settings(parse_optionals_as_positionals=True)
    auto_cli(main, as_positional=False)
