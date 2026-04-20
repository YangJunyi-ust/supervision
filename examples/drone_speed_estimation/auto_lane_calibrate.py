"""
Automatic lane-line calibration for drone speed estimation.

Uses OpenCV colour filtering + Hough-line detection to find lane markings
in multiple frames, clusters the resulting lines by angle and position,
measures the inter-lane pixel spacing and derives either:

  * A full perspective homography (pixel → real-world metres), **or**
  * A plain GSD value (metres / pixel) as a fallback.

The output ``calibration.json`` is identical in format to the one produced by
``calibrate_lane.py`` (manual tool), so ``drone_speed.py`` accepts it
transparently.

Standalone usage
----------------
::

    # Grab ~3 s of stream and auto-calibrate
    python auto_lane_calibrate.py \\
        --rtmp_url rtmp://aplay.wucepro.com/live/1581F8HGX25CC00A17UC-99-0-0 \\
        --output   calibration.json \\
        --lane_width_m 3.5 \\
        --num_frames   90 \\
        --debug        True

Programmatic usage (from drone_speed.py)
-----------------------------------------
::

    from auto_lane_calibrate import auto_calibrate
    ok = auto_calibrate(
        rtmp_url="rtmp://…",
        output="calibration.json",
        lane_width_m=3.5,
        num_frames=90,
    )
    if ok:
        # load calibration.json as ViewTransformer …
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

_MIN_LANE_PX_FRAC  = 0.003  # lane must be at least 0.3% of image width (~4 px on 1280)
_MAX_LANE_PX_FRAC  = 0.15   # at most 15% — one lane is never > 15% of image width
                             # (15% of 1280 px ≈ 192 px ↔ altitude ≤ ~14 m at 84° FOV)
_CLUSTER_GAP_FRAC  = 0.008  # 1-D x-cluster gap threshold (fraction of width)
_ANGLE_TOL_DEG     = 22.0   # lines within ±22° of dominant angle are kept


def _white_yellow_mask(frame: np.ndarray) -> np.ndarray:
    """Return a binary mask covering white and yellow lane-marking pixels."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    white  = cv2.inRange(hsv, (0,   0,  180), (180,  45, 255))
    yellow = cv2.inRange(hsv, (15, 55,   55), ( 45, 255, 255))
    return cv2.bitwise_or(white, yellow)


def _canny(frame: np.ndarray) -> np.ndarray:
    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur  = cv2.GaussianBlur(gray, (5, 5), 0)
    return cv2.Canny(blur, 40, 130)


def _detect_hough_lines(frame: np.ndarray) -> list[tuple[int, int, int, int]]:
    """Run colour + Canny + HoughLinesP and return raw (x1,y1,x2,y2) segments."""
    h, w = frame.shape[:2]
    color_mask = _white_yellow_mask(frame)
    edges      = _canny(frame)
    combined   = cv2.bitwise_and(edges, color_mask)

    lines = cv2.HoughLinesP(
        combined,
        rho=1,
        theta=np.pi / 360,
        threshold=40,
        minLineLength=int(h * 0.06),
        maxLineGap=int(h * 0.06),
    )
    if lines is None:
        return []
    return [tuple(l[0]) for l in lines]


def _angle(x1: int, y1: int, x2: int, y2: int) -> float:
    """Line angle in degrees, normalised to [0°, 180°)."""
    return math.degrees(math.atan2(y2 - y1, x2 - x1)) % 180.0


def _x_at_y(x1: int, y1: int, x2: int, y2: int, y_target: float) -> float | None:
    """X-intercept of the line at *y_target*; None if line is horizontal."""
    if y1 == y2:
        return None
    t = (y_target - y1) / (y2 - y1)
    return x1 + t * (x2 - x1)


# ---------------------------------------------------------------------------
# Core detection
# ---------------------------------------------------------------------------

def _dominant_angle(lines: list[tuple[int, int, int, int]]) -> float:
    """Return the dominant angle (degrees) of a list of line segments."""
    angles = [_angle(*l) for l in lines]
    hist, bins = np.histogram(angles, bins=72, range=(0.0, 180.0))
    best_bin = int(np.argmax(hist))
    return (bins[best_bin] + bins[best_bin + 1]) / 2.0


def _filter_by_angle(
    lines: list[tuple[int, int, int, int]],
    dominant: float,
    tol: float = _ANGLE_TOL_DEG,
) -> list[tuple[int, int, int, int]]:
    kept = []
    for l in lines:
        a = _angle(*l)
        diff = min(abs(a - dominant), 180.0 - abs(a - dominant))
        if diff < tol:
            kept.append(l)
    return kept


def _lane_clusters(
    lines: list[tuple[int, int, int, int]],
    y_sample: float,
    frame_w: int,
) -> list[list[float]]:
    """
    Project each line to its x-intercept at *y_sample*; cluster by proximity.
    Returns a list of clusters (each cluster is a list of x-values).
    """
    xs = []
    for l in lines:
        xi = _x_at_y(*l, y_sample)
        if xi is not None and -frame_w * 0.2 < xi < frame_w * 1.2:
            xs.append(xi)
    if not xs:
        return []

    xs_sorted = sorted(xs)
    gap_px = frame_w * _CLUSTER_GAP_FRAC
    clusters: list[list[float]] = [[xs_sorted[0]]]
    for xi in xs_sorted[1:]:
        if xi - clusters[-1][-1] < gap_px:
            clusters[-1].append(xi)
        else:
            clusters.append([xi])
    return clusters


def _expected_lane_px(
    frame_w: int,
    lane_width_m: float,
    altitude_m: float | None,
    camera_hfov_deg: float,
) -> tuple[float, float]:
    """
    Compute the expected pixel width range for one lane given altitude.

    Returns ``(min_px, max_px)`` — a generous ±70% band around the expected
    single-lane width, extended up to 4× for multiple-lane spans.
    If *altitude_m* is None the module-level fraction constants are used.
    """
    if altitude_m is not None and altitude_m > 5.0:
        gsd_approx = (
            2.0 * altitude_m * math.tan(math.radians(camera_hfov_deg / 2.0)) / frame_w
        )
        one_lane_px = lane_width_m / gsd_approx
        # Accept anything from 0.3× to 4× expected single-lane width
        return max(2.0, one_lane_px * 0.3), one_lane_px * 4.0
    return frame_w * _MIN_LANE_PX_FRAC, frame_w * _MAX_LANE_PX_FRAC


def _best_adjacent_pair(
    clusters: list[list[float]],
    frame_w: int,
    lane_width_m: float = 3.5,
    altitude_m: float | None = None,
    camera_hfov_deg: float = 84.0,
) -> tuple[float, float] | None:
    """
    Find the pair of adjacent cluster centres whose pixel gap most plausibly
    represents one lane width.  Returns (x_left, x_right) or None.
    """
    if len(clusters) < 2:
        return None

    centres = [float(np.median(c)) for c in clusters]
    min_px, max_px = _expected_lane_px(frame_w, lane_width_m, altitude_m, camera_hfov_deg)

    # Score = number of supporting lines; prefer pairs whose gap is closest
    # to the expected single-lane width (penalise very large gaps)
    best_pair  = None
    best_score = -1.0

    if altitude_m is not None and altitude_m > 5.0:
        gsd_approx = (
            2.0 * altitude_m * math.tan(math.radians(camera_hfov_deg / 2.0)) / frame_w
        )
        ideal_px = lane_width_m / gsd_approx
    else:
        ideal_px = (min_px + max_px) / 2.0

    for i in range(len(centres) - 1):
        gap = centres[i + 1] - centres[i]
        if min_px <= gap <= max_px:
            support = len(clusters[i]) + len(clusters[i + 1])
            # Prefer gap closest to ideal single lane; degrade for multi-lane spans
            proximity = 1.0 / (1.0 + abs(gap - ideal_px) / max(ideal_px, 1.0))
            score = support * proximity
            if score > best_score:
                best_score = score
                best_pair  = (centres[i], centres[i + 1])

    return best_pair


# ---------------------------------------------------------------------------
# Multi-frame aggregation → lane width in pixels
# ---------------------------------------------------------------------------

def estimate_lane_pixel_widths(
    frames: list[np.ndarray],
    y_fracs: tuple[float, float] = (0.45, 0.70),
    lane_width_m: float = 3.5,
    altitude_m: float | None = None,
    camera_hfov_deg: float = 84.0,
) -> dict[str, float | None]:
    """
    Process *frames* and return the estimated lane pixel widths at two
    vertical fractions of the image height.

    Args:
        frames: List of BGR frames to analyse.
        y_fracs: Vertical sampling positions (fraction of image height).
        lane_width_m: Real-world width of one lane in metres.
        altitude_m: Drone altitude (used to compute expected lane pixel width).
            Pass ``None`` to use hard-coded fraction limits.
        camera_hfov_deg: Base horizontal FOV in degrees.

    Returns:
        ``{"y_near": w_px_near, "y_far": w_px_far}`` — either may be ``None``
        if detection failed at that level.
    """
    if not frames:
        return {"y_near": None, "y_far": None}

    h, w = frames[0].shape[:2]
    results: dict[str, float | None] = {}

    for key, y_frac in zip(("y_near", "y_far"), y_fracs):
        y_sample = h * y_frac
        all_widths: list[float] = []

        for frame in frames:
            raw = _detect_hough_lines(frame)
            if not raw:
                continue
            dom   = _dominant_angle(raw)
            filt  = _filter_by_angle(raw, dom)
            clust = _lane_clusters(filt, y_sample, w)
            pair  = _best_adjacent_pair(
                clust, w,
                lane_width_m=lane_width_m,
                altitude_m=altitude_m,
                camera_hfov_deg=camera_hfov_deg,
            )
            if pair is not None:
                all_widths.append(pair[1] - pair[0])

        if len(all_widths) >= 3:
            results[key] = float(np.median(all_widths))
        else:
            results[key] = None

    return results


# ---------------------------------------------------------------------------
# Homography builder
# ---------------------------------------------------------------------------

def _representative_line(
    lines: list[tuple[int, int, int, int]],
    x_center: float,
    frame_w: int,
) -> tuple[int, int, int, int] | None:
    """Return the line whose midpoint x is closest to *x_center*."""
    if not lines:
        return None
    best   = None
    best_d = float("inf")
    for l in lines:
        mx = (l[0] + l[2]) / 2.0
        d  = abs(mx - x_center)
        if d < best_d:
            best_d = d
            best   = l
    return best


def build_homography_from_frames(
    frames: list[np.ndarray],
    lane_width_m: float = 3.5,
    altitude_m: float | None = None,
    camera_hfov_deg: float = 84.0,
    y_fracs: tuple[float, float] = (0.45, 0.70),
    debug_frame: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray] | None:
    """
    Attempt to build a 4-point homography source/target from auto-detected
    lane lines.

    Returns:
        ``(source_px, target_m)`` as float32 arrays of shape (4, 2), or
        ``None`` if detection failed.
    """
    if not frames:
        return None

    h, w = frames[0].shape[:2]

    # ── Step 1: collect all filtered lines across frames ────────────────────
    all_filtered: list[tuple[int, int, int, int]] = []
    for frame in frames:
        raw  = _detect_hough_lines(frame)
        if not raw:
            continue
        dom  = _dominant_angle(raw)
        all_filtered.extend(_filter_by_angle(raw, dom))

    if not all_filtered:
        return None

    # ── Step 2: find lane pair at image-centre level ─────────────────────────
    y_mid = h * 0.6
    clust_mid = _lane_clusters(all_filtered, y_mid, w)
    pair_mid  = _best_adjacent_pair(
        clust_mid, w,
        lane_width_m=lane_width_m,
        altitude_m=altitude_m,
        camera_hfov_deg=camera_hfov_deg,
    )
    if pair_mid is None:
        return None

    x_left_mid, x_right_mid = pair_mid

    # ── Step 3: find representative lines for left / right boundary ──────────
    line_left  = _representative_line(all_filtered, x_left_mid,  w)
    line_right = _representative_line(all_filtered, x_right_mid, w)
    if line_left is None or line_right is None:
        return None

    # ── Step 4: sample both lines at two y-levels ───────────────────────────
    y_near = h * y_fracs[1]   # lower in image = nearer vehicle
    y_far  = h * y_fracs[0]   # higher in image = farther away

    src_pts_dict: dict[str, float | None] = {}
    for name, line, y in [
        ("tl", line_left,  y_far),
        ("tr", line_right, y_far),
        ("br", line_right, y_near),
        ("bl", line_left,  y_near),
    ]:
        xi = _x_at_y(*line, y)
        src_pts_dict[name] = xi

    if any(v is None for v in src_pts_dict.values()):
        return None

    source = np.array(
        [
            [src_pts_dict["tl"], y_far ],
            [src_pts_dict["tr"], y_far ],
            [src_pts_dict["br"], y_near],
            [src_pts_dict["bl"], y_near],
        ],
        dtype=np.float32,
    )

    # ── Step 5: measure lane widths at both y-levels ─────────────────────────
    # Use projective scaling to estimate real-world depth between the two rows.
    # depth_m ≈ lane_width_m × pixel_row_gap / avg(lane_px_near, lane_px_far)
    lane_px_near = abs(src_pts_dict["br"] - src_pts_dict["bl"])
    lane_px_far  = abs(src_pts_dict["tr"] - src_pts_dict["tl"])
    if lane_px_near < 1 or lane_px_far < 1:
        return None

    avg_lane_px = (lane_px_near + lane_px_far) / 2.0
    depth_m     = lane_width_m * abs(y_near - y_far) / avg_lane_px

    target = np.array(
        [
            [0.0,          0.0     ],
            [lane_width_m, 0.0     ],
            [lane_width_m, depth_m ],
            [0.0,          depth_m ],
        ],
        dtype=np.float32,
    )

    # ── Optional debug overlay ────────────────────────────────────────────────
    if debug_frame is not None:
        colors = [(0, 255, 0), (0, 200, 255), (0, 0, 255), (255, 128, 0)]
        labels = ["TL", "TR", "BR", "BL"]
        for pt, color, lbl in zip(source, colors, labels):
            px, py = int(pt[0]), int(pt[1])
            cv2.circle(debug_frame, (px, py), 10, color, -1)
            cv2.putText(debug_frame, lbl, (px + 12, py - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA)
        # draw the quadrilateral
        pts_i = source.astype(np.int32)
        cv2.polylines(debug_frame, [pts_i], isClosed=True,
                      color=(255, 255, 0), thickness=2)
        cv2.putText(
            debug_frame,
            f"Auto-calib: {lane_width_m:.1f}m x {depth_m:.1f}m",
            (12, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2,
        )

    return source, target


# ---------------------------------------------------------------------------
# Stream utilities
# ---------------------------------------------------------------------------

def _open_stream(url: str, retries: int = 8, delay: float = 2.0) -> cv2.VideoCapture:
    for i in range(1, retries + 1):
        cap = cv2.VideoCapture(url)
        if cap.isOpened():
            return cap
        cap.release()
        print(f"[auto_calib] connect attempt {i}/{retries} failed, retrying…")
        time.sleep(delay)
    raise RuntimeError(f"Cannot connect: {url}")


def _grab_frames(url: str, num_frames: int = 90) -> list[np.ndarray]:
    """Read up to *num_frames* frames from *url* (skip first 5 for warmup)."""
    p = Path(url)
    if p.exists() and p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp"):
        img = cv2.imread(str(p))
        return [img] * num_frames if img is not None else []

    cap = _open_stream(url)
    frames: list[np.ndarray] = []
    warmup = 5
    idx = 0
    while len(frames) < num_frames:
        ret, frame = cap.read()
        if not ret:
            break
        idx += 1
        if idx <= warmup:
            continue
        frames.append(frame)
    cap.release()
    return frames


# ---------------------------------------------------------------------------
# Crosswalk (zebra-stripe) GSD detection
# ---------------------------------------------------------------------------


def detect_crosswalk_gsd(
    frames: list[np.ndarray],
    stripe_width_m: float = 0.40,
    min_stripes: int = 3,
    min_stripe_px: int = 4,
    max_stripe_fraction: float = 0.25,
) -> float | None:
    """Estimate GSD from zebra-crossing stripe widths.

    Scans for white rectangular stripes on the road surface using HSV masking
    and connected-component analysis.  Groups stripe-shaped blobs by
    orientation (horizontal or vertical), measures the short-side width of
    each stripe, and returns a GSD derived from the median.

    Standard stripe widths (metres):
    - China (GB 5768):  0.40 m
    - Europe:           0.50 m
    - North America:    0.30–0.60 m

    Args:
        frames: BGR frames sampled from the live stream.
        stripe_width_m: Real-world width of one white stripe in metres.
        min_stripes: Minimum number of confirmed stripes required across all
            frames before returning a GSD estimate.
        min_stripe_px: Minimum acceptable stripe short-side in pixels.
        max_stripe_fraction: Maximum stripe short-side as a fraction of the
            smaller frame dimension (rejects very large blobs that are not
            stripes).

    Returns:
        GSD in metres per pixel, or ``None`` if no crosswalk was detected.

    Example::

        gsd = detect_crosswalk_gsd(frames, stripe_width_m=0.40)
        if gsd:
            print(f"crosswalk GSD = {gsd*100:.2f} cm/px")
    """
    all_widths_px: list[float] = []

    for frame in frames[::3]:  # sample every 3rd frame for speed
        h, w = frame.shape[:2]
        max_stripe_px = int(min(h, w) * max_stripe_fraction)

        # ── White-marking mask ────────────────────────────────────────────
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # White: low saturation, high value
        white_mask = cv2.inRange(hsv, (0, 0, 170), (180, 55, 255))

        # Morphological open to remove noise specks
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, kernel)

        # Focus on the lower two-thirds of the frame (road area, skip sky)
        white_mask[: h // 3, :] = 0

        # ── Connected component analysis ──────────────────────────────────
        n_labels, _, stats, _ = cv2.connectedComponentsWithStats(
            white_mask, connectivity=8
        )

        h_widths: list[float] = []  # short-side of horizontal stripes
        v_widths: list[float] = []  # short-side of vertical stripes

        for i in range(1, n_labels):
            bx = int(stats[i, cv2.CC_STAT_WIDTH])
            by = int(stats[i, cv2.CC_STAT_HEIGHT])
            area = int(stats[i, cv2.CC_STAT_AREA])

            short_side = min(bx, by)
            long_side  = max(bx, by)

            # Must be stripe-shaped: long-to-short ratio ≥ 3
            if short_side < 1 or long_side / short_side < 3.0:
                continue

            # Short side within the expected pixel range
            if not (min_stripe_px <= short_side <= max_stripe_px):
                continue

            # Fill ratio: solid white blob, not hollow or L-shaped
            if area < short_side * long_side * 0.35:
                continue

            if bx >= by:   # wider than tall → horizontal stripe
                h_widths.append(float(by))   # short side = height
            else:           # taller than wide → vertical stripe
                v_widths.append(float(bx))   # short side = width

        # Take the orientation group with more detections
        best = h_widths if len(h_widths) >= len(v_widths) else v_widths
        if len(best) >= 2:
            all_widths_px.extend(best)

    if len(all_widths_px) < min_stripes:
        return None

    # Reject outliers via IQR before taking the median
    arr = np.array(all_widths_px)
    q25, q75 = float(np.percentile(arr, 25)), float(np.percentile(arr, 75))
    iqr = q75 - q25
    filtered = arr[(arr >= q25 - 1.5 * iqr) & (arr <= q75 + 1.5 * iqr)]

    if len(filtered) < min_stripes:
        return None

    median_px = float(np.median(filtered))
    gsd = stripe_width_m / median_px
    print(
        f"[auto_calib] crosswalk: {len(filtered)} stripes, "
        f"median {median_px:.1f} px → GSD = {gsd * 100:.2f} cm/px"
    )
    return gsd


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def auto_calibrate(
    rtmp_url: str,
    output: str = "calibration.json",
    lane_width_m: float = 3.5,
    altitude_m: float | None = None,
    camera_hfov_deg: float = 84.0,
    num_frames: int = 90,
    debug: bool = False,
    show_preview: bool = True,
    crosswalk_stripe_width_m: float = 0.40,
) -> bool:
    """
    Automatically detect lane lines and write ``calibration.json``.

    Args:
        rtmp_url: RTMP stream URL or path to a local image/video file.
        output: Destination path for ``calibration.json``.
        lane_width_m: Real-world width of **one** lane in metres.
            Use 3.5 for urban / national roads, 3.75 for highways.
        altitude_m: Drone altitude above ground in metres.  Strongly
            recommended — used to compute the expected lane pixel width,
            which dramatically improves detection at high altitudes.
        camera_hfov_deg: Camera base horizontal FOV in degrees (default 84°).
        num_frames: Number of frames to analyse (more = more robust).
        debug: If ``True``, save a debug PNG showing detected lines.
        show_preview: If ``True``, open a preview window (requires display).
        crosswalk_stripe_width_m: Real-world width of one zebra-crossing
            stripe in metres.  Used when lane-line homography fails and the
            scene is an intersection with visible crosswalk markings.
            China standard = 0.40 m; Europe = 0.50 m.

    Returns:
        ``True`` on success, ``False`` if all detection methods failed.

    Calibration priority chain
    --------------------------
    1. Lane-line homography (best accuracy, straight road with lane markings)
    2. Crosswalk stripe GSD (intersection with zebra crossings)
    3. Lane-line GSD estimate (rough fallback for partial lane visibility)

    Example::

        ok = auto_calibrate(
            rtmp_url="rtmp://…",
            output="calibration.json",
            lane_width_m=3.5,
            altitude_m=120.0,
            crosswalk_stripe_width_m=0.40,
        )
    """
    if altitude_m is not None:
        gsd_est = 2 * altitude_m * math.tan(math.radians(camera_hfov_deg / 2)) / 1280
        exp_px  = lane_width_m / gsd_est
        print(f"[auto_calib] altitude={altitude_m:.1f}m → expected lane ≈ {exp_px:.1f} px")
    print(f"[auto_calib] grabbing {num_frames} frames from stream…")
    frames = _grab_frames(rtmp_url, num_frames)
    if not frames:
        print("[auto_calib] ERROR: could not read any frames.")
        return False

    print(f"[auto_calib] got {len(frames)} frames — detecting lane lines…")
    h, w = frames[0].shape[:2]

    debug_frame = frames[len(frames) // 2].copy() if debug or show_preview else None
    result = build_homography_from_frames(
        frames,
        lane_width_m=lane_width_m,
        altitude_m=altitude_m,
        camera_hfov_deg=camera_hfov_deg,
        debug_frame=debug_frame,
    )

    calib_source = "lane_homography"

    if result is None:
        print("[auto_calib] homography build failed — trying crosswalk detection…")

        # ── Level 2: crosswalk stripe GSD ────────────────────────────────────
        gsd = detect_crosswalk_gsd(
            frames, stripe_width_m=crosswalk_stripe_width_m
        )
        calib_source_tag = "crosswalk"

        if gsd is None:
            # ── Level 3: lane-line pixel-width GSD ───────────────────────────
            print("[auto_calib] no crosswalk found — trying lane-line GSD fallback…")
            widths = estimate_lane_pixel_widths(
                frames,
                lane_width_m=lane_width_m,
                altitude_m=altitude_m,
                camera_hfov_deg=camera_hfov_deg,
            )
            px_w = widths.get("y_near") or widths.get("y_far")
            if px_w is not None:
                gsd = lane_width_m / px_w
                calib_source_tag = "lane_gsd"
                print(
                    f"[auto_calib] lane-GSD fallback: lane ≈ {px_w:.1f} px "
                    f"→ GSD = {gsd * 100:.2f} cm/px"
                )

        if gsd is None:
            print("[auto_calib] FAILED: no usable markings detected.")
            print("  Suggestions:")
            print("  • Straight road: ensure visible lane lines (try --lane_width_m 3.75)")
            print("  • Intersection: ensure visible crosswalk stripes (try --stripe-width 0.45)")
            print("  • No markings: skip --auto-calib and use GSD mode (--no-calib)")
            return False

        calib_source = calib_source_tag

        # Build a synthetic 4-point calibration rectangle centred on the image
        # using the GSD from whichever level succeeded.
        ref_m  = crosswalk_stripe_width_m if calib_source == "crosswalk" else lane_width_m
        ref_px = ref_m / gsd              # reference dimension in pixels
        cx = w / 2.0
        rect_w_px = ref_px * (lane_width_m / ref_m)   # scale to lane_width_m
        y1, y2 = h * 0.35, h * 0.70
        depth_m = lane_width_m * (y2 - y1) / rect_w_px

        source = np.array([
            [cx - rect_w_px / 2, y1],
            [cx + rect_w_px / 2, y1],
            [cx + rect_w_px / 2, y2],
            [cx - rect_w_px / 2, y2],
        ], dtype=float).tolist()
        target = [
            [0.0,          0.0     ],
            [lane_width_m, 0.0     ],
            [lane_width_m, depth_m ],
            [0.0,          depth_m ],
        ]
    else:
        src_arr, tgt_arr = result
        source = src_arr.tolist()
        target = tgt_arr.tolist()
        depth_m = float(tgt_arr[2, 1])
        gsd     = lane_width_m / float(
            abs(src_arr[1, 0] - src_arr[0, 0]) + 1e-9
        )
        print(f"[auto_calib] homography OK — area {lane_width_m:.1f}m × {depth_m:.1f}m")
        print(f"[auto_calib] approx GSD ≈ {gsd*100:.2f} cm/px")

    calib = {
        "source_pixels":       source,
        "target_meters":       target,
        "real_width_m":        lane_width_m,
        "real_length_m":       depth_m,
        "frame_size":          [w, h],
        "auto_generated":      True,
        "calibration_source":  calib_source,   # lane_homography / crosswalk / lane_gsd
        "altitude_m":          altitude_m,     # None when unknown
        "description": (
            f"Auto calibration [{calib_source}]: "
            f"{lane_width_m:.1f} m wide × {depth_m:.1f} m long, from {rtmp_url}"
        ),
    }
    Path(output).write_text(json.dumps(calib, indent=2))
    print(f"[auto_calib] saved → {output}")

    if show_preview and debug_frame is not None:
        if result is not None:
            pass  # already drawn in build_homography_from_frames
        else:
            # draw the synthetic rectangle
            pts = np.array(source, dtype=np.int32)
            cv2.polylines(debug_frame, [pts], True, (0, 255, 255), 2)
            cv2.putText(debug_frame, "Auto-calib (GSD fallback)",
                        (12, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
        cv2.imshow("Auto Lane Calibration — press any key", debug_frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    if debug and debug_frame is not None:
        dbg_path = Path(output).with_suffix(".debug.jpg")
        cv2.imwrite(str(dbg_path), debug_frame)
        print(f"[auto_calib] debug image → {dbg_path}")

    return True


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main(
    rtmp_url: str,
    output: str = "calibration.json",
    lane_width_m: float = 3.5,
    altitude_m: float | None = None,
    camera_hfov_deg: float = 84.0,
    num_frames: int = 90,
    debug: bool = False,
    show_preview: bool = True,
) -> None:
    """Automatic lane-line calibration tool.

    Args:
        rtmp_url: RTMP stream URL (or path to a saved frame image).
        output: Path to write the calibration JSON file.
        lane_width_m: Real width of **one** lane in metres.
            Urban / national road: 3.5 m.  Highway: 3.75 m.
        altitude_m: Drone altitude in metres. Highly recommended — dramatically
            improves detection accuracy at high altitudes (> 50 m).
        camera_hfov_deg: Camera base horizontal FOV in degrees (default 84°).
        num_frames: Frames to process (more = more stable, slower start).
        debug: Save a debug JPEG showing the detected calibration zone.
        show_preview: Show a preview window with the detected zone.
    """
    ok = auto_calibrate(
        rtmp_url=rtmp_url,
        output=output,
        lane_width_m=lane_width_m,
        altitude_m=altitude_m,
        camera_hfov_deg=camera_hfov_deg,
        num_frames=num_frames,
        debug=debug,
        show_preview=show_preview,
    )
    if ok:
        print(f"\nRun speed estimation with:  --calibration {output}")
    else:
        raise SystemExit(1)


if __name__ == "__main__":
    from jsonargparse import auto_cli, set_parsing_settings
    set_parsing_settings(parse_optionals_as_positionals=True)
    auto_cli(main, as_positional=False)
