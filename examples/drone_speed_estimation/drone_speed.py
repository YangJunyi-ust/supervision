"""
Drone-based vehicle speed estimation from an RTMP live stream.

Pipeline
--------
1. DJI Cloud API WebSocket (daemon thread) → drone telemetry at ~0.5 Hz.
2. RTMP stream (buffer thread) → raw video frames.
3. YOLO inference (main thread, every ``frame_stride`` frames).
4. supervision ByteTrack → stable tracker IDs for each vehicle.
5. SpeedCalculator → true ground speed compensated for drone motion.
6. Annotators → bounding boxes, speed labels, trace, telemetry HUD.
7. cv2.imshow and/or VideoSink output.

Usage
-----
Live display only::

    python drone_speed.py \\
        --rtmp_url rtmp://aplay.wucepro.com/live/1581F8HGX25CC00A17UC-99-0-0 \\
        --ws_url   ws://139.9.212.1:6789/api/v1/ws/workspace/shilonghw \\
        --weights  yolov8n.pt

Record to file::

    python drone_speed.py \\
        --rtmp_url   rtmp://aplay.wucepro.com/live/1581F8HGX25CC00A17UC-99-0-0 \\
        --ws_url     ws://139.9.212.1:6789/api/v1/ws/workspace/shilonghw \\
        --weights    yolo11x.pt \\
        --target_video output.mp4 \\
        --camera_hfov  84.0 \\
        --confidence   0.3 \\
        --frame_stride 2 \\
        --device       mps
"""

from __future__ import annotations

import sys
import threading
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

import json

import supervision as sv

# Sibling modules
sys.path.insert(0, str(Path(__file__).parent))
from auto_lane_calibrate import auto_calibrate  # noqa: E402
from dji_telemetry import DJITelemetryClient, TelemetryState  # noqa: E402
from speed_utils import ByteTrackSpeedEstimator, GSDAutoCalibrator, ViewTransformer, compute_gsd  # noqa: E402

# ---------------------------------------------------------------------------
# COCO vehicle class IDs to keep (car, motorcycle, bus, truck)
# ---------------------------------------------------------------------------
VEHICLE_CLASS_IDS = {2, 3, 5, 7}

# ---------------------------------------------------------------------------
# HUD layout constants
# ---------------------------------------------------------------------------
HUD_LINE_HEIGHT = 28          # pixels between HUD text lines
HUD_MARGIN_X = 16             # pixels from left edge
HUD_MARGIN_Y = 36             # pixels from top edge
HUD_FONT = cv2.FONT_HERSHEY_SIMPLEX
HUD_FONT_SCALE = 0.65
HUD_THICKNESS = 2
HUD_COLOR = (255, 255, 255)   # white
HUD_BG_COLOR = (0, 0, 0)      # black shadow / background rect


# ---------------------------------------------------------------------------
# RTMP capture thread
# ---------------------------------------------------------------------------


def _open_stream(url: str, max_retries: int = 10, retry_delay: float = 2.0) -> cv2.VideoCapture:
    """Open an RTMP (or any OpenCV-compatible) stream, retrying on failure.

    Args:
        url: Stream URL.
        max_retries: Maximum number of connection attempts.
        retry_delay: Seconds to wait between attempts.

    Returns:
        An opened ``cv2.VideoCapture`` handle.

    Raises:
        RuntimeError: If the stream cannot be opened after all retries.
    """
    for attempt in range(1, max_retries + 1):
        cap = cv2.VideoCapture(url)
        if cap.isOpened():
            return cap
        print(f"[stream] attempt {attempt}/{max_retries} failed, retrying in {retry_delay:.0f} s…")
        cap.release()
        time.sleep(retry_delay)
    raise RuntimeError(f"Cannot connect to stream after {max_retries} attempts: {url}")


def _rtmp_reader(
    url: str,
    frame_buf: deque,         # deque(maxlen=1), stores (frame, capture_ts) tuples
    stop_event: threading.Event,
    max_reconnects: int = 10,
    reconnect_delay: float = 2.0,
) -> None:
    """Background thread: reads frames from RTMP and stores the latest in *frame_buf*.

    Each entry in *frame_buf* is a ``(frame, capture_ts)`` tuple where
    ``capture_ts`` is ``time.time()`` recorded immediately after ``cap.read()``
    returns.  This timestamp is used downstream for telemetry alignment so that
    the drone position is evaluated at *capture* time, not at *processing* time
    (which is later by however long YOLO took on the previous frame).

    Args:
        url: RTMP stream URL.
        frame_buf: ``deque(maxlen=1)`` shared with the main thread.
        stop_event: Signal to terminate this thread.
        max_reconnects: Maximum reconnection attempts after a dropout.
        reconnect_delay: Seconds between reconnect attempts.
    """
    reconnects = 0
    cap = _open_stream(url, max_retries=max_reconnects, retry_delay=reconnect_delay)

    while not stop_event.is_set():
        ret, frame = cap.read()
        capture_ts = time.time()   # record as close to cap.read() as possible
        if not ret or frame is None:
            reconnects += 1
            print(f"[stream] dropout (reconnect {reconnects}/{max_reconnects})…")
            cap.release()
            if reconnects > max_reconnects:
                print("[stream] max reconnects reached, stopping reader thread.")
                break
            time.sleep(reconnect_delay)
            try:
                cap = _open_stream(url, max_retries=3, retry_delay=reconnect_delay)
                reconnects = 0
            except RuntimeError as exc:
                print(f"[stream] reconnect failed: {exc}")
                break
            continue

        reconnects = 0
        frame_buf.append((frame, capture_ts))

    cap.release()


# ---------------------------------------------------------------------------
# HUD overlay
# ---------------------------------------------------------------------------


def _draw_hud(
    frame: np.ndarray,
    state: TelemetryState,
    gsd: float,
    telem_lag: float,
    capture_ts: float,
    gsd_source: str = "formula",
    calib_mode: str = "gsd",
) -> np.ndarray:
    """Draw a telemetry HUD in the top-left corner of *frame*.

    Args:
        frame: BGR image to annotate (modified in-place).
        state: Latest drone telemetry snapshot.
        gsd: Current Ground Sampling Distance in m/px.
        telem_lag: Seconds since the last OSD message was received.
        capture_ts: Unix timestamp when this frame was captured by the reader.

    Returns:
        The annotated frame (same object as *frame*).
    """
    status = "CONNECTED" if state.connected else "NO SIGNAL"
    proc_lag = time.time() - capture_ts
    speed_mode_str = "lane-calib" if calib_mode == "lane" else f"GSD [{gsd_source}]"
    lines = [
        f"Drone : {status}",
        f"Alt   : {state.altitude:.1f} m",
        f"Head  : {state.attitude_head:.0f} deg",
        f"Pitch : {state.attitude_pitch:.1f} deg",
        f"Zoom  : x{state.zoom_factor:.1f}",
        f"GndSpd: {state.horizontal_speed:.1f} m/s",
        f"VrtSpd: {state.vertical_speed:+.1f} m/s",
        f"SpeedM: {speed_mode_str}",
        f"GSD   : {gsd * 100:.1f} cm/px",
        f"GPS   : {state.latitude:.5f}, {state.longitude:.5f}",
        f"TelLag: {telem_lag:.1f} s",
        f"ProcLg: {proc_lag * 1000:.0f} ms",
    ]

    # Semi-transparent background
    bg_w = 310
    bg_h = HUD_LINE_HEIGHT * len(lines) + 12
    overlay = frame.copy()
    cv2.rectangle(
        overlay,
        (HUD_MARGIN_X - 8, HUD_MARGIN_Y - 24),
        (HUD_MARGIN_X + bg_w, HUD_MARGIN_Y + bg_h),
        HUD_BG_COLOR,
        cv2.FILLED,
    )
    cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)

    for i, line in enumerate(lines):
        y = HUD_MARGIN_Y + i * HUD_LINE_HEIGHT
        cv2.putText(
            frame, line,
            (HUD_MARGIN_X, y),
            HUD_FONT, HUD_FONT_SCALE, HUD_COLOR, HUD_THICKNESS, cv2.LINE_AA,
        )
    return frame


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def _load_calibration(path: str | None) -> tuple["ViewTransformer | None", "float | None"]:
    """Load a calibration.json and return (ViewTransformer, calib_altitude_m).

    Returns:
        Tuple of (transformer, altitude). Altitude is ``None`` when not stored
        in the file (older calibration files).
    """
    if path is None:
        return None, None
    try:
        data = json.loads(Path(path).read_text())
        src = np.array(data["source_pixels"], dtype=np.float32)
        tgt = np.array(data["target_meters"], dtype=np.float32)
        vt = ViewTransformer(source=src, target=tgt)
        calib_alt = data.get("altitude_m")  # may be None for old files
        auto_tag = " [auto]" if data.get("auto_generated") else ""
        src_tag  = data.get("calibration_source", "unknown")
        print(f"[calibration] loaded{auto_tag} from {path}  [{src_tag}]")
        print(f"  real area: {data.get('real_width_m')} m × {data.get('real_length_m')} m")
        if calib_alt is not None:
            print(f"  calib altitude: {calib_alt:.1f} m")
        else:
            print("  calib altitude: unknown (old file — altitude drift detection disabled)")
        return vt, calib_alt
    except Exception as exc:
        print(f"[calibration] WARNING: could not load {path}: {exc}")
        return None, None


def run(
    rtmp_url: str,
    ws_url: str,
    weights: str = "yolov8n.pt",
    camera_hfov: float = 84.0,
    zoom_factor: float = 1.0,
    calibration: str | None = None,
    auto_calibrate_lane: bool = False,
    lane_width_m: float = 3.5,
    confidence: float = 0.3,
    iou_threshold: float = 0.7,
    frame_stride: int = 2,
    target_video: str | None = None,
    device: str | None = None,
    max_reconnects: int = 10,
    reconnect_delay: float = 2.0,
    speed_window: float = 1.0,
    show_preview: bool = True,
    imgsz: int = 1280,
    altitude_tolerance: float = 8.0,
    crosswalk_stripe_width_m: float = 0.40,
    lost_track_buffer: int = 90,
    minimum_consecutive_frames: int = 2,
) -> None:
    """Run the drone vehicle speed estimation pipeline.

    Args:
        rtmp_url: RTMP stream URL from the drone.
        ws_url: DJI Cloud API WebSocket URL.
        weights: Path to YOLO weights file (e.g. ``yolov8n.pt``, ``yolo11x.pt``).
        camera_hfov: Horizontal field of view of the drone camera in degrees.
            84° suits most DJI Mavic 3 and M30T wide-lens configurations.
        confidence: YOLO detection confidence threshold.
        iou_threshold: NMS IoU threshold.
        frame_stride: Run YOLO inference every N frames; the display always
            runs at the stream frame rate.
        target_video: If provided, write the annotated stream to this path.
        device: PyTorch device string (``"cpu"``, ``"cuda:0"``, ``"mps"``).
            ``None`` lets Ultralytics choose automatically.
        max_reconnects: Maximum RTMP reconnection attempts.
        reconnect_delay: Seconds between RTMP reconnects.
        speed_window: Sliding window length in seconds for speed averaging.
        show_preview: Show an OpenCV window while running. Press ``q`` to quit.
        imgsz: Inference image size (larger = better recall for distant vehicles).
        altitude_tolerance: Max altitude deviation (m) from calibration before
            auto-disabling the homography and falling back to GSD mode.
            Set to 0 to disable drift detection.
    """
    # ── DJI WebSocket telemetry ────────────────────────────────────────────
    telem_client = DJITelemetryClient(ws_url=ws_url)
    telem_client.start()
    print(f"[telemetry] connecting to {ws_url} …")

    # ── RTMP stream reader thread ──────────────────────────────────────────
    frame_buf: deque = deque(maxlen=1)
    stop_event = threading.Event()
    reader = threading.Thread(
        target=_rtmp_reader,
        args=(rtmp_url, frame_buf, stop_event, max_reconnects, reconnect_delay),
        daemon=True,
        name="rtmp-reader",
    )
    reader.start()
    print(f"[stream] connecting to {rtmp_url} …")

    # Wait until we have at least one frame
    timeout = 30.0
    t0 = time.time()
    while not frame_buf and time.time() - t0 < timeout:
        time.sleep(0.1)
    if not frame_buf:
        stop_event.set()
        telem_client.stop()
        raise RuntimeError("Timed out waiting for first frame from RTMP stream.")

    # Infer frame dimensions from the first frame
    first_frame: np.ndarray = frame_buf[-1][0]
    frame_h, frame_w = first_frame.shape[:2]
    fps_stream = 25.0  # RTMP streams often don't expose fps; assume 25

    print(f"[stream] connected  {frame_w}×{frame_h} (assumed {fps_stream:.0f} fps)")

    # ── supervision annotators ─────────────────────────────────────────────
    thickness = sv.calculate_optimal_line_thickness((frame_w, frame_h))
    text_scale = sv.calculate_optimal_text_scale((frame_w, frame_h))

    box_annotator = sv.BoxAnnotator(thickness=thickness)
    label_annotator = sv.LabelAnnotator(
        text_scale=text_scale,
        text_thickness=thickness,
        text_position=sv.Position.TOP_LEFT,
    )
    trace_annotator = sv.TraceAnnotator(
        thickness=thickness,
        trace_length=int(fps_stream * 2),
        position=sv.Position.BOTTOM_CENTER,
    )

    # ── YOLO model ────────────────────────────────────────────────────────
    model = YOLO(weights)
    infer_kwargs: dict = dict(
        conf=confidence,
        iou=iou_threshold,
        imgsz=imgsz,
        verbose=False,
    )
    if device:
        infer_kwargs["device"] = device

    # ── ByteTrack ─────────────────────────────────────────────────────────
    # lost_track_buffer: keep a track alive for N frames even when the
    # detector misses the vehicle, preventing ID reassignment on brief
    # occlusions (tree shadows, buildings, etc.).
    # minimum_consecutive_frames=2: require 2 consecutive hits before
    # confirming a new track, reducing spurious one-frame false positives.
    byte_track = sv.ByteTrack(
        frame_rate=int(fps_stream),
        track_activation_threshold=confidence,
        lost_track_buffer=lost_track_buffer,
        minimum_consecutive_frames=minimum_consecutive_frames,
    )

    # ── Auto lane calibration (if requested and no manual file given) ─────
    _auto_calib_path = "calibration_auto.json"
    if auto_calibrate_lane and calibration is None:
        # Wait up to 8 s for the telemetry thread to deliver a valid altitude.
        # Altitude is critical for the auto-calibrator to compute the expected
        # lane pixel width at the current drone height.
        print("[drone_speed] waiting for telemetry altitude before auto-calibration…")
        _alt_wait = 0.0
        while _alt_wait < 8.0:
            _snap = telem_client.snapshot()
            _alt  = _snap.elevation if _snap.elevation > 5.0 else _snap.altitude
            if _alt > 5.0:
                break
            time.sleep(0.5)
            _alt_wait += 0.5
        _calib_alt = _alt if _alt > 5.0 else None
        if _calib_alt:
            print(f"[drone_speed] using altitude={_calib_alt:.1f} m for lane calibration")
        else:
            print("[drone_speed] WARNING: no altitude received — calibration may be less accurate")

        print("[drone_speed] running automatic lane calibration…")
        ok = auto_calibrate(
            rtmp_url=rtmp_url,
            output=_auto_calib_path,
            lane_width_m=lane_width_m,
            altitude_m=_calib_alt,
            camera_hfov_deg=camera_hfov,
            num_frames=90,
            debug=False,
            show_preview=False,
            crosswalk_stripe_width_m=crosswalk_stripe_width_m,
        )
        if ok:
            calibration = _auto_calib_path
            print(f"[drone_speed] auto calibration saved → {_auto_calib_path}")
        else:
            print("[drone_speed] WARNING: auto calibration failed — falling back to GSD mode")

    # ── Lane calibration (highest accuracy if available) ──────────────────
    view_transformer, calib_altitude = _load_calibration(calibration)
    # Track whether calibration was ever valid so we can print once when disabling
    _calib_disabled_by_drift = False

    # ── Speed estimator + GSD auto-calibrator ─────────────────────────────
    # ByteTrackSpeedEstimator uses wall-clock time windows so it converges
    # at the same rate regardless of YOLO inference FPS (unlike the old
    # frame-count-based SpeedCalculator which could take 4+ s at low FPS).
    speed_calc = ByteTrackSpeedEstimator(
        window_seconds=speed_window,
        min_seconds=max(0.2, speed_window * 0.25),
        ema_alpha=0.25,
        view_transformer=view_transformer,
    )
    gsd_calib = GSDAutoCalibrator(window=120, min_samples=10)
    calib_mode = "lane" if view_transformer is not None else "gsd"

    # ── VideoSink (optional) ──────────────────────────────────────────────
    video_info = sv.VideoInfo(
        width=frame_w,
        height=frame_h,
        fps=int(fps_stream),
        total_frames=0,
    )
    sink: sv.VideoSink | None = None
    if target_video:
        sink = sv.VideoSink(target_path=target_video, video_info=video_info)
        sink.__enter__()
        print(f"[output] recording to {target_video}")

    # ── Main processing loop ───────────────────────────────────────────────
    latest_detections = sv.Detections.empty()
    latest_speeds: dict[int, float | None] = {}
    frame_idx = 0
    cleanup_interval = 100  # frames between stale-tracker cleanups

    print("[main] pipeline running — press  q  to quit")

    try:
        while True:
            if not frame_buf:
                time.sleep(0.005)
                continue

            frame, capture_ts = frame_buf.pop()

            # ── Telemetry aligned to frame capture time ───────────────────
            # Use capture_ts (when cap.read() returned in the reader thread)
            # rather than time.time() (now, after YOLO on the previous frame).
            # This removes the YOLO-inference-time offset from drone position.
            telem: TelemetryState = telem_client.snapshot_at(capture_ts)

            # ── GSD ───────────────────────────────────────────────────────
            # Prefer elevation (relative to takeoff) if available
            alt = telem.elevation if telem.elevation > 1.0 else telem.altitude

            # ── Altitude drift detection ──────────────────────────────────
            # When the drone has climbed or descended significantly from the
            # altitude at which the lane calibration was captured, the
            # homography becomes inaccurate.  Disable it automatically.
            if (
                view_transformer is not None
                and calib_altitude is not None
                and altitude_tolerance > 0
                and alt > 1.0   # only act when telemetry is live
                and abs(alt - calib_altitude) > altitude_tolerance
                and not _calib_disabled_by_drift
            ):
                _calib_disabled_by_drift = True
                view_transformer = None
                calib_mode = "gsd"
                speed_calc._vt = None  # hot-swap the transformer out
                print(
                    f"[calibration] DISABLED — altitude drift "
                    f"{alt:.1f} m vs calib {calib_altitude:.1f} m "
                    f"(tolerance ±{altitude_tolerance:.0f} m). "
                    "Switched to GSD mode."
                )
            # ── GSD resolution priority ───────────────────────────────────
            # 1. Bbox auto-calibration (adapts to any zoom automatically)
            # 2. OSD zoom telemetry (if bbox calib not ready yet)
            # 3. Manual --zoom_factor override (always wins if != 1.0)
            gsd_auto = gsd_calib.gsd()
            if zoom_factor != 1.0:
                # explicit manual override
                effective_zoom = zoom_factor
                gsd = compute_gsd(
                    altitude_m=max(alt, 5.0),
                    frame_width_px=frame_w,
                    camera_hfov_deg=camera_hfov,
                    gimbal_pitch_deg=telem.attitude_pitch,
                    zoom_factor=effective_zoom,
                )
                gsd_source = "manual"
            elif gsd_auto is not None:
                # auto-calibrated from vehicle bounding boxes — zoom-independent
                gsd = gsd_auto
                gsd_source = "auto"
            else:
                # fall back to formula with OSD zoom until enough samples arrive
                effective_zoom = telem.zoom_factor
                gsd = compute_gsd(
                    altitude_m=max(alt, 5.0),
                    frame_width_px=frame_w,
                    camera_hfov_deg=camera_hfov,
                    gimbal_pitch_deg=telem.attitude_pitch,
                    zoom_factor=effective_zoom,
                )
                gsd_source = "formula"

            # ── Inference (every frame_stride frames) ─────────────────────
            if frame_idx % frame_stride == 0:
                results = model(frame, **infer_kwargs)[0]
                detections = sv.Detections.from_ultralytics(results)

                # Filter to vehicle classes only
                if detections.class_id is not None and len(detections) > 0:
                    mask = np.isin(detections.class_id, list(VEHICLE_CLASS_IDS))
                    detections = detections[mask]

                detections = byte_track.update_with_detections(detections=detections)
                latest_detections = detections

                # Feed bbox widths to GSD auto-calibrator (uses car/bus/truck widths)
                gsd_calib.feed(detections.xyxy, detections.class_id)

                # ── Speed estimation ──────────────────────────────────────
                if detections.tracker_id is not None and len(detections) > 0:
                    anchors = detections.get_anchors_coordinates(
                        anchor=sv.Position.BOTTOM_CENTER
                    )
                    latest_speeds = speed_calc.update(
                        tracker_ids=detections.tracker_id,
                        pixel_positions=anchors,
                        drone_lat=telem.latitude,
                        drone_lon=telem.longitude,
                        attitude_head=telem.attitude_head,
                        gsd=gsd,
                        frame_ts=capture_ts,   # use frame capture time, not now
                    )
                else:
                    latest_speeds = {}

            # ── Cleanup stale trackers ────────────────────────────────────
            if frame_idx % cleanup_interval == 0 and latest_detections.tracker_id is not None:
                speed_calc.cleanup_stale(set(latest_detections.tracker_id.tolist()))

            # ── Build labels ──────────────────────────────────────────────
            labels = _build_labels(latest_detections, latest_speeds)

            # ── Annotate ──────────────────────────────────────────────────
            annotated = frame.copy()
            annotated = trace_annotator.annotate(annotated, latest_detections)
            annotated = box_annotator.annotate(annotated, latest_detections)
            annotated = label_annotator.annotate(annotated, latest_detections, labels)
            annotated = _draw_hud(
                annotated, telem, gsd,
                telem_lag=telem_client.telem_lag(),
                capture_ts=capture_ts,
                gsd_source=gsd_source,
                calib_mode=calib_mode,
            )

            if sink is not None:
                sink.write_frame(annotated)

            if show_preview:
                cv2.imshow("Drone Speed Estimation", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            frame_idx += 1

    finally:
        stop_event.set()
        reader.join(timeout=5)
        telem_client.stop()
        if sink is not None:
            sink.__exit__(None, None, None)
        if show_preview:
            cv2.destroyAllWindows()
        print("[main] stopped.")


def _build_labels(
    detections: sv.Detections,
    speeds: dict[int, float | None],
) -> list[str]:
    """Build per-detection label strings.

    Args:
        detections: Current tracked detections.
        speeds: Mapping from tracker_id to speed in km/h (or None).

    Returns:
        List of label strings, one per detection.
    """
    labels = []
    if detections.tracker_id is None:
        return [""] * len(detections)

    for tracker_id in detections.tracker_id:
        tid = int(tracker_id)
        speed = speeds.get(tid)
        if speed is None:
            labels.append(f"#{tid}")
        else:
            labels.append(f"#{tid}  {int(speed)} km/h")
    return labels


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main(
    rtmp_url: str,
    ws_url: str,
    weights: str = "yolov8n.pt",
    camera_hfov: float = 84.0,
    zoom_factor: float = 1.0,
    calibration: str | None = None,
    auto_calibrate_lane: bool = False,
    lane_width_m: float = 3.5,
    confidence: float = 0.3,
    iou_threshold: float = 0.7,
    frame_stride: int = 2,
    target_video: str | None = None,
    device: str | None = None,
    max_reconnects: int = 10,
    reconnect_delay: float = 2.0,
    speed_window: float = 1.0,
    show_preview: bool = True,
    imgsz: int = 1280,
    altitude_tolerance: float = 8.0,
    crosswalk_stripe_width_m: float = 0.40,
    lost_track_buffer: int = 90,
    minimum_consecutive_frames: int = 2,
) -> None:
    """Drone vehicle speed estimation from RTMP live stream + DJI WebSocket.

    Args:
        rtmp_url: RTMP stream URL from the drone camera. Example:
            ``rtmp://aplay.wucepro.com/live/1581F8HGX25CC00A17UC-99-0-0``
        ws_url: DJI Cloud API WebSocket URL. Example:
            ``ws://139.9.212.1:6789/api/v1/ws/workspace/shilonghw``
        weights: Ultralytics YOLO weights file.  Use ``yolov8n.pt`` for speed
            or ``yolo11x.pt`` / ``yolov8x.pt`` for accuracy.
        camera_hfov: Base (×1 zoom) horizontal field of view in degrees.
            DJI Mavic 3 wide lens ≈ 84°; DJI M30T wide ≈ 83°; DJI Mini 3 ≈ 82.1°.
        zoom_factor: Manual zoom override (default 1.0 = use telemetry value).
            Set this when the DJI OSD does not report zoom automatically, e.g.
            ``--zoom_factor 2.0`` when the camera is at ×2 optical zoom.
            If set to anything other than 1.0, telemetry zoom is ignored.
        calibration: Path to a ``calibration.json`` file generated by
            ``calibrate_lane.py`` or ``auto_lane_calibrate.py``.  When
            provided, vehicle positions are projected to real-world metres
            using a lane-marking homography, which is much more accurate than
            GSD estimation.  Recommended whenever you can see clear lane
            markings in the video.
        auto_calibrate_lane: When ``True`` and no ``--calibration`` file is
            given, automatically detect lane lines in the first ~90 frames of
            the stream and build a calibration.  Saves
            ``calibration_auto.json`` in the working directory.  Falls back
            to GSD mode if detection fails.  Requires clearly visible lane
            markings.
        lane_width_m: Real-world width of **one** lane in metres used for
            ``--auto_calibrate_lane``.  3.5 = urban / national road (default),
            3.75 = highway.
        confidence: Minimum detection confidence (0–1).
        iou_threshold: NMS IoU threshold (0–1, ignored by NMS-free models).
        frame_stride: Run YOLO every N frames (display always at full frame rate).
            ``1`` = every frame (accurate), ``2–4`` = faster on slow hardware.
        target_video: Optional output video file path. Annotated frames are
            written here in addition to (or instead of) the preview window.
        device: Inference device: ``"cpu"``, ``"cuda:0"``, ``"mps"``, or
            ``None`` for auto-selection.
        max_reconnects: How many times to attempt RTMP reconnection on dropout.
        reconnect_delay: Seconds to wait between reconnect attempts.
        speed_window: Sliding window in seconds over which speed is averaged.
            Larger values smooth speed estimates; smaller values are more
            responsive to acceleration/deceleration events.
        show_preview: Open a live OpenCV display window.  Set to ``False`` when
            running headless (e.g. on a server recording only to ``target_video``).
        imgsz: YOLO inference image size in pixels. 640 is fast; 1280 improves
            recall for small vehicles on high-altitude footage.
    """
    run(
        rtmp_url=rtmp_url,
        ws_url=ws_url,
        weights=weights,
        camera_hfov=camera_hfov,
        zoom_factor=zoom_factor,
        calibration=calibration,
        auto_calibrate_lane=auto_calibrate_lane,
        lane_width_m=lane_width_m,
        confidence=confidence,
        iou_threshold=iou_threshold,
        frame_stride=frame_stride,
        target_video=target_video,
        device=device,
        max_reconnects=max_reconnects,
        reconnect_delay=reconnect_delay,
        speed_window=speed_window,
        show_preview=show_preview,
        imgsz=imgsz,
        altitude_tolerance=altitude_tolerance,
        crosswalk_stripe_width_m=crosswalk_stripe_width_m,
        lost_track_buffer=lost_track_buffer,
        minimum_consecutive_frames=minimum_consecutive_frames,
    )


if __name__ == "__main__":
    from jsonargparse import auto_cli, set_parsing_settings

    set_parsing_settings(parse_optionals_as_positionals=True)
    auto_cli(main, as_positional=False)
