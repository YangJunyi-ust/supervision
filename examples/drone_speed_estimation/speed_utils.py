"""
Speed estimation utilities for drone-based vehicle speed measurement.

Provides:
- haversine_m: geodetic distance between two GPS points
- compute_gsd: Ground Sampling Distance from altitude + camera FOV
- ViewTransformer: perspective homography (pixel → bird's-eye metres)
- SpeedCalculator: per-tracker sliding-window speed with drone motion compensation

Example::

    from speed_utils import SpeedCalculator, compute_gsd

    gsd = compute_gsd(altitude_m=80.0, frame_width_px=1920, camera_hfov_deg=84.0)
    calc = SpeedCalculator(window_seconds=1.0)

    # each frame:
    speeds = calc.update(
        tracker_ids=detections.tracker_id,
        pixel_positions=points,    # (N, 2) BOTTOM_CENTER pixels
        telemetry=client.snapshot().dead_reckon(time.time()),
        gsd=gsd,
        fps=30,
    )
"""

from __future__ import annotations

import math
import time
from collections import defaultdict, deque
from dataclasses import dataclass

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Geodetic helpers
# ---------------------------------------------------------------------------


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Return the great-circle distance in metres between two WGS-84 points.

    Args:
        lat1: Latitude of the first point in decimal degrees.
        lon1: Longitude of the first point in decimal degrees.
        lat2: Latitude of the second point in decimal degrees.
        lon2: Longitude of the second point in decimal degrees.

    Returns:
        Distance in metres.

    Example::

        d = haversine_m(23.456, 113.444, 23.457, 113.445)
        # → ~142 m
    """
    R = 6_371_000.0  # Earth radius in metres
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def compute_gsd(
    altitude_m: float,
    frame_width_px: int,
    camera_hfov_deg: float = 84.0,
    gimbal_pitch_deg: float = -90.0,
    zoom_factor: float = 1.0,
) -> float:
    """Compute the nadir-equivalent Ground Sampling Distance (metres per pixel).

    Relationship between focal length, sensor size and hfov
    --------------------------------------------------------
    The ``camera_hfov_deg`` parameter encodes both focal length and sensor width::

        hfov = 2 × arctan(sensor_width / (2 × focal_length))
        GSD  = (altitude × sensor_width) / (focal_length × frame_width)
             = (2 × altitude × tan(hfov/2)) / frame_width   ← what we compute

    So focal length is implicitly included via hfov — you do not need to pass
    it separately.

    Zoom correction  ← KEY for variable-zoom cameras
    ---------------
    Zooming in multiplies the effective focal length, which shrinks the FOV
    and reduces GSD proportionally::

        effective_hfov = 2 × arctan(tan(base_hfov/2) / zoom_factor)
        GSD_zoomed     = GSD_base / zoom_factor

    **This is the most common source of speed estimation error on DJI drones.**
    At ×2 zoom the GSD halves; if zoom is ignored, speed is over-estimated by ×2.

    Gimbal pitch correction
    -----------------------
    When the gimbal is tilted (``gimbal_pitch_deg`` ≠ −90), the camera no
    longer points straight down.  The slant range to the ground centre
    increases, so the *effective* GSD at the image centre grows::

        slant_range = altitude / sin(|pitch|)
        GSD_tilted  = GSD_nadir / sin(|pitch|)

    Args:
        altitude_m: Drone altitude above ground in metres. Prefer ``elevation``
            (relative to takeoff) when available.
        frame_width_px: Width of the video frame in pixels.
        camera_hfov_deg: **Base** (×1 zoom) horizontal field of view in degrees.
            DJI Mavic 3 wide ≈ 84°; DJI M30T wide ≈ 83°; DJI Mini 3 ≈ 82.1°.
        gimbal_pitch_deg: Gimbal pitch angle in degrees. DJI convention:
            −90° = straight down (nadir), 0° = horizontal.
        zoom_factor: Current optical/digital zoom multiplier (1.0 = no zoom).
            Read from DJI OSD ``zoom_factor`` field, or pass via ``--zoom_factor``.

    Returns:
        GSD in metres per pixel (centre-of-frame representative value).

    Example::

        # No zoom, nadir, 80 m, 84° FOV, 1920 px
        gsd = compute_gsd(80.0, 1920)
        # → ~0.073 m/px

        # ×3 zoom (e.g. DJI M30T optical zoom)
        gsd = compute_gsd(80.0, 1920, zoom_factor=3.0)
        # → ~0.024 m/px  (1/3 of no-zoom)

        # ×3 zoom + gimbal at −60°
        gsd = compute_gsd(80.0, 1920, gimbal_pitch_deg=-60.0, zoom_factor=3.0)
        # → ~0.028 m/px
    """
    if altitude_m <= 0 or frame_width_px <= 0:
        return 0.01  # safety floor

    zoom = max(zoom_factor, 0.1)   # guard against zero/negative

    hfov_rad = math.radians(camera_hfov_deg)
    # Zoom shrinks the FOV: effective tan(hfov/2) = tan(base_hfov/2) / zoom
    gsd_nadir = (2.0 * altitude_m * math.tan(hfov_rad / 2.0)) / (frame_width_px * zoom)

    # Gimbal pitch correction
    pitch_rad = math.radians(abs(gimbal_pitch_deg))
    sin_pitch = math.sin(pitch_rad)
    if sin_pitch < 0.05:
        return gsd_nadir
    return gsd_nadir / sin_pitch


# ---------------------------------------------------------------------------
# Perspective transformer (pixel → calibrated bird's-eye plane)
# ---------------------------------------------------------------------------


class ViewTransformer:
    """Homography-based perspective transform for pixel → real-world coordinates.

    Wraps ``cv2.getPerspectiveTransform`` and ``cv2.perspectiveTransform``.
    When a four-corner road polygon and its real-world dimensions are known
    (e.g., a marked lane of known width/length), use this class to convert
    bottom-centre pixel anchors into metric positions before computing speed.

    This is optional. If no calibration polygon is available, the
    :class:`SpeedCalculator` falls back to a pure GSD scaling approach which
    is less accurate but requires no manual calibration.

    Args:
        source: ``(4, 2)`` array of pixel coordinates marking the source polygon
            (top-left, top-right, bottom-right, bottom-left order).
        target: ``(4, 2)`` array of real-world coordinates in metres (or pixels
            of a rectified bird's-eye image).

    Example::

        source = np.array([[300, 400], [900, 400], [1100, 700], [100, 700]])
        target = np.array([[0, 0], [10, 0], [10, 50], [0, 50]])
        vt = ViewTransformer(source, target)
        world_pts = vt.transform_points(pixel_pts)
    """

    def __init__(self, source: np.ndarray, target: np.ndarray) -> None:
        source = source.astype(np.float32)
        target = target.astype(np.float32)
        self.m = cv2.getPerspectiveTransform(source, target)

    def transform_points(self, points: np.ndarray) -> np.ndarray:
        """Transform pixel points to target coordinate space.

        Args:
            points: ``(N, 2)`` array of pixel coordinates.

        Returns:
            ``(N, 2)`` array in target coordinate space.

        Example::

            world_pts = vt.transform_points(np.array([[500, 600]]))
        """
        if points.size == 0:
            return points
        reshaped = points.reshape(-1, 1, 2).astype(np.float32)
        transformed = cv2.perspectiveTransform(reshaped, self.m)
        return transformed.reshape(-1, 2)


# ---------------------------------------------------------------------------
# Per-frame entry stored in the sliding window
# ---------------------------------------------------------------------------


@dataclass
class _FrameEntry:
    """One frame's worth of data for a single tracked vehicle."""

    pixel_x: float
    pixel_y: float
    # World-space position in metres (set when a ViewTransformer is available)
    world_x: float
    world_y: float
    drone_lat: float
    drone_lon: float
    ts: float  # local time.time()


# ---------------------------------------------------------------------------
# Speed calculator
# ---------------------------------------------------------------------------


class SpeedCalculator:
    """Compute true ground speed for each tracked vehicle, compensating for drone motion.

    Two modes
    ---------
    **Calibrated mode** (recommended): pass a ``ViewTransformer`` built from
    lane-marking calibration.  Vehicle pixel positions are projected to
    real-world metres before computing displacement, so speed is accurate
    regardless of zoom, altitude, or perspective distortion.

    **GSD mode** (fallback): if no transformer is given, pixel displacement
    is multiplied by a GSD value (metres/pixel) and drone motion is
    subtracted in the same pixel space.

    Args:
        window_seconds: Sliding window length in seconds.
        min_window_fraction: Minimum fraction of the full window that must be
            filled before reporting a speed (avoids noisy early estimates).
        view_transformer: Optional :class:`ViewTransformer` for calibrated
            pixel→metres mapping.  When provided, GSD is only used for
            drone motion compensation (which remains in pixel space).

    Example::

        vt = ViewTransformer(source=src_pts, target=tgt_pts)
        calc = SpeedCalculator(window_seconds=1.0, view_transformer=vt)
        speeds = calc.update(
            tracker_ids=np.array([1, 2]),
            pixel_positions=np.array([[100, 200], [300, 400]]),
            drone_lat=23.456, drone_lon=113.444,
            attitude_head=45.0,
            gsd=0.07,   # still needed for drone-motion compensation
            fps=25,
        )
    """

    def __init__(
        self,
        window_seconds: float = 1.0,
        min_window_fraction: float = 0.5,
        view_transformer: "ViewTransformer | None" = None,
    ) -> None:
        self._window_seconds = window_seconds
        self._min_window_fraction = min_window_fraction
        self._vt = view_transformer
        # tracker_id → deque[_FrameEntry]
        self._history: dict[int, deque[_FrameEntry]] = defaultdict(
            lambda: deque(maxlen=self._max_len)
        )
        self._max_len = 9999  # will be updated lazily on first update call

    def update(
        self,
        tracker_ids: np.ndarray,
        pixel_positions: np.ndarray,
        drone_lat: float,
        drone_lon: float,
        attitude_head: float,
        gsd: float,
        fps: float,
        frame_ts: float | None = None,
    ) -> dict[int, float | None]:
        """Record one frame and return the latest speed estimate per tracker.

        Args:
            tracker_ids: ``(N,)`` integer array of tracker IDs (from ByteTrack).
            pixel_positions: ``(N, 2)`` array of ``[x, y]`` pixel positions for
                each tracked vehicle (e.g. ``BOTTOM_CENTER`` anchors).
            drone_lat: Drone latitude (decimal degrees) at this frame.
            drone_lon: Drone longitude (decimal degrees) at this frame.
            attitude_head: Drone compass heading in degrees (0 = North, 90 = East).
            gsd: Ground Sampling Distance in metres per pixel.
            fps: Video frame rate (used to size the sliding window).
            frame_ts: Unix timestamp (seconds) of when this frame was *captured*
                by the RTMP reader thread.  Pass ``None`` to fall back to
                ``time.time()`` (less accurate when YOLO inference is slow).

        Returns:
            Dict mapping ``tracker_id → speed_km_h``. Speed is ``None`` when
            the window is too short for a reliable estimate.

        Example::

            speeds = calc.update(
                tracker_ids=detections.tracker_id,
                pixel_positions=anchors,
                drone_lat=state.latitude, drone_lon=state.longitude,
                attitude_head=state.attitude_head, gsd=gsd, fps=30,
                frame_ts=capture_ts,
            )
        """
        # Lazily resize deques when fps becomes known
        new_max = max(2, int(fps * self._window_seconds))
        if new_max != self._max_len:
            self._max_len = new_max
            for tid in list(self._history.keys()):
                old = self._history[tid]
                new_dq: deque[_FrameEntry] = deque(maxlen=new_max)
                new_dq.extend(old)
                self._history[tid] = new_dq

        # Use frame capture timestamp when available so that dt in the sliding
        # window reflects real elapsed capture time, not YOLO processing time.
        now = frame_ts if frame_ts is not None else time.time()
        speeds: dict[int, float | None] = {}

        # Transform all positions to world space in one batch if a transformer
        # is available (much faster than looping through each point).
        if self._vt is not None and len(pixel_positions) > 0:
            world_pts = self._vt.transform_points(
                np.array(pixel_positions, dtype=np.float32)
            )
        else:
            world_pts = None

        for i, (tid, (px, py)) in enumerate(zip(tracker_ids, pixel_positions)):
            tid = int(tid)
            wx = float(world_pts[i, 0]) if world_pts is not None else float(px)
            wy = float(world_pts[i, 1]) if world_pts is not None else float(py)
            entry = _FrameEntry(
                pixel_x=float(px),
                pixel_y=float(py),
                world_x=wx,
                world_y=wy,
                drone_lat=drone_lat,
                drone_lon=drone_lon,
                ts=now,
            )

            dq = self._history[tid]
            dq.append(entry)

            min_entries = max(2, int(self._max_len * self._min_window_fraction))
            if len(dq) < min_entries:
                speeds[tid] = None
                continue

            speeds[tid] = self._compute_speed(dq, attitude_head, gsd)

        return speeds

    def _compute_speed(
        self,
        dq: deque[_FrameEntry],
        attitude_head: float,
        gsd: float,
    ) -> float:
        """Compute speed from the sliding window entries.

        When a ViewTransformer was provided at construction, uses world-space
        (metres) coordinates directly.  Otherwise falls back to pixel
        displacement × GSD.

        Args:
            dq: Deque of frame entries for one tracker.
            attitude_head: Current drone compass heading in degrees.
            gsd: Ground Sampling Distance in metres per pixel.

        Returns:
            Speed in km/h.
        """
        oldest = dq[0]
        newest = dq[-1]

        dt = newest.ts - oldest.ts
        if dt <= 0:
            return 0.0

        # --- Drone GPS displacement in metres ---
        drone_dist_m = haversine_m(
            oldest.drone_lat, oldest.drone_lon,
            newest.drone_lat, newest.drone_lon,
        )

        if self._vt is not None:
            # ── Calibrated mode: world coords are already in metres ───────────
            # Vehicle world displacement (metres) directly from homography
            veh_dx_m = newest.world_x - oldest.world_x
            veh_dy_m = newest.world_y - oldest.world_y

            # Drone motion in world space: project along heading
            theta = math.radians(attitude_head)
            drone_dx_m = drone_dist_m * math.sin(theta)
            drone_dy_m = drone_dist_m * math.cos(theta)

            true_dx_m = veh_dx_m - drone_dx_m
            true_dy_m = veh_dy_m - drone_dy_m
        else:
            # ── GSD mode: convert pixel displacement to metres via GSD ────────
            veh_dx_px = newest.pixel_x - oldest.pixel_x
            veh_dy_px = newest.pixel_y - oldest.pixel_y

            drone_dist_px = drone_dist_m / gsd if gsd > 0 else 0.0
            theta = math.radians(attitude_head)
            drone_dx_px = drone_dist_px * math.sin(theta)
            drone_dy_px = drone_dist_px * math.cos(theta)

            true_dx_m = (veh_dx_px - drone_dx_px) * gsd
            true_dy_m = (veh_dy_px - drone_dy_px) * gsd

        dist_m = math.sqrt(true_dx_m ** 2 + true_dy_m ** 2)
        return dist_m / dt * 3.6  # → km/h

    def cleanup_stale(self, active_ids: set[int]) -> None:
        """Remove history for tracker IDs that are no longer active.

        Call this periodically (e.g., every 100 frames) to prevent unbounded
        memory growth when vehicles leave the scene and ByteTrack retires IDs.

        Args:
            active_ids: Set of tracker IDs currently present in the scene.

        Example::

            calc.cleanup_stale(set(detections.tracker_id))
        """
        stale = [tid for tid in self._history if tid not in active_ids]
        for tid in stale:
            del self._history[tid]


# ---------------------------------------------------------------------------
# GSD auto-calibrator — infers GSD from known vehicle dimensions
# ---------------------------------------------------------------------------

# Real-world widths used for GSD inference (metres), indexed by COCO class ID.
# Widths are conservative medians from traffic studies; tune if needed.
VEHICLE_REAL_WIDTH_M: dict[int, float] = {
    2: 1.85,   # car
    5: 2.55,   # bus
    7: 2.50,   # truck
}


class GSDAutoCalibrator:
    """Estimate GSD by comparing detected vehicle bounding-box widths to
    known real-world vehicle widths.

    This completely bypasses the need to know the camera zoom factor.
    As long as at least a few cars are visible in the frame, the calibrator
    converges to an accurate GSD within a few seconds — and it automatically
    re-adapts whenever the zoom changes.

    Algorithm
    ---------
    For each detected vehicle of class C:

        gsd_sample = VEHICLE_REAL_WIDTH_M[C] / bbox_width_px

    Samples are collected in a rolling window.  The **median** is used (not
    mean) to reject outliers from partial occlusions or unusual vehicles.

    Args:
        window: Rolling window length (number of samples, not seconds).
        min_samples: Minimum samples before a calibrated estimate is returned.

    Example::

        calib = GSDAutoCalibrator()

        # each frame:
        calib.feed(detections.xyxy, detections.class_id)
        gsd = calib.gsd()      # None until enough samples collected
        if gsd is not None:
            speed = calc.update(..., gsd=gsd, ...)
    """

    def __init__(self, window: int = 60, min_samples: int = 10) -> None:
        self._samples: deque[float] = deque(maxlen=window)
        self._min_samples = min_samples

    def feed(
        self,
        xyxy: np.ndarray,
        class_ids: np.ndarray | None,
    ) -> None:
        """Ingest bounding boxes from one frame to update the GSD estimate.

        Args:
            xyxy: ``(N, 4)`` bounding boxes in ``[x1, y1, x2, y2]`` format.
            class_ids: ``(N,)`` integer COCO class IDs.  Pass ``None`` to skip.

        Example::

            calib.feed(detections.xyxy, detections.class_id)
        """
        if class_ids is None or len(xyxy) == 0:
            return

        for box, cid in zip(xyxy, class_ids):
            real_w = VEHICLE_REAL_WIDTH_M.get(int(cid))
            if real_w is None:
                continue
            bbox_w_px = float(box[2] - box[0])
            if bbox_w_px < 5:          # discard degenerate boxes
                continue
            self._samples.append(real_w / bbox_w_px)

    def gsd(self) -> float | None:
        """Return the median GSD estimate (m/px), or ``None`` if not ready.

        Example::

            g = calib.gsd()
            if g:
                print(f"auto GSD = {g*100:.2f} cm/px")
        """
        if len(self._samples) < self._min_samples:
            return None
        return float(np.median(list(self._samples)))

    def reset(self) -> None:
        """Clear all collected samples.

        Example::

            calib.reset()
        """
        self._samples.clear()


# ---------------------------------------------------------------------------
# ByteTrack-based time-windowed speed estimator
# ---------------------------------------------------------------------------


@dataclass
class _TrackPoint:
    """One observation for a single ByteTrack vehicle."""

    world_x: float   # metres (or pixels when no ViewTransformer)
    world_y: float
    pixel_x: float
    pixel_y: float
    drone_lat: float
    drone_lon: float
    ts: float        # Unix capture timestamp (seconds)


class ByteTrackSpeedEstimator:
    """Time-windowed speed estimator built directly on ByteTrack outputs.

    Unlike :class:`SpeedCalculator` (which uses a fixed-size deque sized by
    an assumed FPS), this class uses wall-clock timestamps throughout so it
    converges at the same rate regardless of the actual inference FPS.

    Key differences vs. SpeedCalculator
    ------------------------------------
    * Speed is reported after ``min_seconds`` of real elapsed time (default
      0.3 s) instead of after a fixed number of frames.
    * History is pruned by time (``window_seconds``), not by frame count.
    * Per-tracker EMA smoothing (``ema_alpha``) reduces per-frame jitter.

    Args:
        window_seconds: How many seconds of history to retain per tracker.
        min_seconds: Minimum elapsed time before reporting a speed estimate.
            Lower = more responsive; higher = smoother but delayed.
        ema_alpha: Exponential moving average weight for the raw speed.
            0 < alpha <= 1; smaller = smoother but slower to react.
        view_transformer: Optional :class:`ViewTransformer` for calibrated
            pixel → real-world metre mapping.  When provided, drone-motion
            compensation is applied in metre space.  Falls back to GSD
            pixel-space calculation when ``None``.

    Example::

        est = ByteTrackSpeedEstimator(window_seconds=1.5, view_transformer=vt)

        # each frame:
        speeds = est.update(
            tracker_ids=detections.tracker_id,
            pixel_positions=anchors,
            drone_lat=state.latitude, drone_lon=state.longitude,
            attitude_head=state.attitude_head, gsd=gsd,
            frame_ts=capture_ts,
        )
    """

    def __init__(
        self,
        window_seconds: float = 1.5,
        min_seconds: float = 0.3,
        ema_alpha: float = 0.25,
        view_transformer: "ViewTransformer | None" = None,
    ) -> None:
        self._window = window_seconds
        self._min_seconds = min_seconds
        self._alpha = ema_alpha
        self._vt = view_transformer

        # tracker_id → list[_TrackPoint], ordered oldest→newest
        self._history: dict[int, list[_TrackPoint]] = defaultdict(list)
        # tracker_id → EMA-smoothed speed (km/h)
        self._smoothed: dict[int, float] = {}

    def update(
        self,
        tracker_ids: np.ndarray,
        pixel_positions: np.ndarray,
        drone_lat: float,
        drone_lon: float,
        attitude_head: float,
        gsd: float,
        frame_ts: float | None = None,
    ) -> "dict[int, float | None]":
        """Record one frame of detections and return speed per tracker.

        Args:
            tracker_ids: ``(N,)`` integer array from ByteTrack.
            pixel_positions: ``(N, 2)`` array of ``[x, y]`` pixel anchors
                (e.g. ``BOTTOM_CENTER`` of each bounding box).
            drone_lat: Drone latitude (decimal degrees) at this frame.
            drone_lon: Drone longitude (decimal degrees) at this frame.
            attitude_head: Drone compass heading in degrees (0 = North).
            gsd: Ground Sampling Distance (m/px) — used only in GSD fallback
                mode (when no ``view_transformer`` was given).
            frame_ts: Unix timestamp of when this frame was *captured* by the
                RTMP reader thread.  Falls back to ``time.time()`` when
                ``None``.

        Returns:
            Dict mapping ``tracker_id → speed_km_h``, or ``None`` when the
            window is too short to produce a reliable estimate.

        Example::

            speeds = est.update(
                tracker_ids=detections.tracker_id,
                pixel_positions=anchors,
                drone_lat=23.456, drone_lon=113.444,
                attitude_head=45.0, gsd=0.07,
                frame_ts=capture_ts,
            )
        """
        now = frame_ts if frame_ts is not None else time.time()

        # Batch-transform all pixel positions to world space in one call
        if self._vt is not None and len(pixel_positions) > 0:
            world_pts = self._vt.transform_points(
                np.array(pixel_positions, dtype=np.float32)
            )
        else:
            world_pts = None

        active_ids: set[int] = set()
        speeds: dict[int, float | None] = {}

        for i, tid in enumerate(tracker_ids):
            tid = int(tid)
            active_ids.add(tid)

            px = float(pixel_positions[i][0])
            py = float(pixel_positions[i][1])
            wx = float(world_pts[i, 0]) if world_pts is not None else px
            wy = float(world_pts[i, 1]) if world_pts is not None else py

            hist = self._history[tid]
            hist.append(_TrackPoint(
                world_x=wx, world_y=wy,
                pixel_x=px, pixel_y=py,
                drone_lat=drone_lat, drone_lon=drone_lon,
                ts=now,
            ))

            # Prune entries older than the time window
            cutoff = now - self._window
            while hist and hist[0].ts < cutoff:
                hist.pop(0)

            # Need at least 2 points spanning the minimum time threshold
            if len(hist) < 2 or (hist[-1].ts - hist[0].ts) < self._min_seconds:
                speeds[tid] = None
                continue

            speeds[tid] = self._ema_speed(tid, hist, attitude_head, gsd)

        return speeds

    def _ema_speed(
        self,
        tid: int,
        hist: "list[_TrackPoint]",
        attitude_head: float,
        gsd: float,
    ) -> float:
        """Compute raw speed from history window, apply EMA smoothing.

        Args:
            tid: Tracker ID (used to look up stored EMA state).
            hist: Ordered list of track points for this tracker.
            attitude_head: Drone compass heading in degrees.
            gsd: Ground Sampling Distance in m/px.

        Returns:
            EMA-smoothed speed in km/h.
        """
        oldest = hist[0]
        newest = hist[-1]
        dt = newest.ts - oldest.ts
        if dt <= 0:
            return self._smoothed.get(tid, 0.0)

        drone_dist_m = haversine_m(
            oldest.drone_lat, oldest.drone_lon,
            newest.drone_lat, newest.drone_lon,
        )

        if self._vt is not None:
            # Calibrated mode: world coords already in metres
            veh_dx = newest.world_x - oldest.world_x
            veh_dy = newest.world_y - oldest.world_y
            theta = math.radians(attitude_head)
            drone_dx = drone_dist_m * math.sin(theta)
            drone_dy = drone_dist_m * math.cos(theta)
            true_dx = veh_dx - drone_dx
            true_dy = veh_dy - drone_dy
        else:
            # GSD fallback: pixel displacement × GSD
            veh_dx_px = newest.pixel_x - oldest.pixel_x
            veh_dy_px = newest.pixel_y - oldest.pixel_y
            drone_px = drone_dist_m / gsd if gsd > 0 else 0.0
            theta = math.radians(attitude_head)
            drone_dx_px = drone_px * math.sin(theta)
            drone_dy_px = drone_px * math.cos(theta)
            true_dx = (veh_dx_px - drone_dx_px) * gsd
            true_dy = (veh_dy_px - drone_dy_px) * gsd

        raw_speed = math.sqrt(true_dx ** 2 + true_dy ** 2) / dt * 3.6

        # EMA: blend raw into stored smoothed value
        prev = self._smoothed.get(tid)
        smoothed = (
            self._alpha * raw_speed + (1.0 - self._alpha) * prev
            if prev is not None
            else raw_speed
        )
        self._smoothed[tid] = smoothed
        return smoothed

    def cleanup_stale(self, active_ids: "set[int]") -> None:
        """Remove history for tracker IDs no longer in the scene.

        Args:
            active_ids: Set of tracker IDs currently present.

        Example::

            est.cleanup_stale(set(detections.tracker_id))
        """
        stale = [tid for tid in self._history if tid not in active_ids]
        for tid in stale:
            del self._history[tid]
            self._smoothed.pop(tid, None)
