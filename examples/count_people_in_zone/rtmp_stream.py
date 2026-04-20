"""
Real-time people counting in zones from an RTMP live stream.

Usage
-----
Step 1 – capture a frame to draw zones on::

    python examples/count_people_in_zone/rtmp_stream.py \\
      --source_stream "rtmp://aplay.wucepro.com/live/1581F8HGX25CC00A17UC-99-0-0" \\
      --capture_frame_to "/Users/yangsheng/Desktop/stream_frame.jpg"

    Then use draw_zones.py on the saved image to create a zones JSON.

Step 2 – run live detection::

    python examples/count_people_in_zone/rtmp_stream.py \\
      --source_stream "rtmp://aplay.wucepro.com/live/1581F8HGX25CC00A17UC-99-0-0" \\
      --zone_configuration_path "/Users/yangsheng/Desktop/my_zones.json" \\
      --source_weights_path yolo26x.pt \\
      --confidence_threshold 0.10 \\
      --imgsz 1280 \\
      --max_det 800 \\
      --frame_stride 6 \\
      --device mps
"""

import sys
import threading
import time
from collections import deque

import cv2
from ultralytics import YOLO

import supervision as sv

# Re-use all detection / annotation helpers from the file-based script.
sys.path.insert(0, __file__.rsplit("/", 1)[0])
from ultralytics_example import (  # noqa: E402
    annotate,
    detect,
    initiate_annotators,
    load_zones_config,
)


def _open_stream(url: str, max_retries: int = 10, retry_delay: float = 2.0) -> cv2.VideoCapture:
    """Open an RTMP (or any OpenCV-compatible) stream, retrying on failure."""
    for attempt in range(1, max_retries + 1):
        cap = cv2.VideoCapture(url)
        if cap.isOpened():
            return cap
        print(f"[rtmp_stream] connection attempt {attempt}/{max_retries} failed, retrying in {retry_delay}s…")
        cap.release()
        time.sleep(retry_delay)
    raise RuntimeError(f"Could not connect to stream after {max_retries} attempts: {url}")


def capture_frame(source_stream: str, capture_frame_to: str) -> None:
    """Connect to the stream, grab one frame, save it and exit."""
    print(f"[rtmp_stream] connecting to {source_stream} …")
    cap = _open_stream(source_stream)
    ret, frame = cap.read()
    cap.release()
    if not ret or frame is None:
        raise RuntimeError("Connected but could not read a frame from the stream.")
    cv2.imwrite(capture_frame_to, frame)
    h, w = frame.shape[:2]
    print(f"[rtmp_stream] frame saved → {capture_frame_to}  ({w}×{h})")
    print("Now run draw_zones.py on that image to create your zones JSON, then run")
    print("this script again without --capture_frame_to to start live detection.")


def run(
    source_stream: str,
    zone_configuration_path: str,
    source_weights_path: str = "yolo26x.pt",
    confidence_threshold: float = 0.10,
    iou_threshold: float = 0.7,
    imgsz: int = 1280,
    device: str | None = None,
    max_det: int = 800,
    agnostic_nms: bool = False,
    augment: bool = False,
    min_person_area: int | None = None,
    use_sahi: bool = False,
    sahi_slice_wh: tuple[int, int] = (640, 640),
    sahi_overlap_wh: tuple[int, int] = (128, 128),
    frame_stride: int = 1,
    max_reconnects: int = 10,
    reconnect_delay: float = 2.0,
) -> None:
    """
    Run real-time people counting in zones from an RTMP stream.

    Inference runs in a background thread so the display loop is never blocked
    and the video stays smooth even when the model is slow.

    Args:
        source_stream: RTMP (or any OpenCV-compatible) stream URL.
        zone_configuration_path: Path to the zone JSON (``{"polygons": [...]}``).
        source_weights_path: Ultralytics weights file, e.g. ``yolo26x.pt``.
        confidence_threshold: Detection confidence threshold.
        iou_threshold: IoU threshold (used for NMS-based models; ignored by YOLO26).
        imgsz: Inference image size.
        device: Inference device (``mps``, ``cuda:0``, ``cpu``, or None for auto).
        max_det: Maximum detections per frame.
        agnostic_nms: Class-agnostic NMS flag.
        augment: Test-time augmentation (slower, may improve recall).
        min_person_area: Drop boxes smaller than this area in pixels.
        use_sahi: Enable sliced (SAHI-style) inference via ``sv.InferenceSlicer``.
        sahi_slice_wh: Tile size ``(width, height)`` when ``use_sahi=True``.
        sahi_overlap_wh: Overlap ``(width, height)`` in pixels between tiles.
        frame_stride: Send a frame to the inference thread every N frames.
        max_reconnects: Maximum stream reconnection attempts after a dropout.
        reconnect_delay: Seconds to wait between reconnection attempts.
    """
    polygons = load_zones_config(zone_configuration_path)
    model = YOLO(source_weights_path)

    print(f"[rtmp_stream] connecting to {source_stream} …")
    cap = _open_stream(source_stream, max_retries=max_reconnects, retry_delay=reconnect_delay)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1920
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 1080
    resolution_wh = (width, height)

    zones, zone_annotators, box_annotators = initiate_annotators(
        polygons=polygons, resolution_wh=resolution_wh
    )

    # ── shared state between main thread (display) and inference thread ──
    # inference_queue: at most 1 pending frame to avoid building up a backlog
    inference_queue: deque = deque(maxlen=1)
    # latest_detections is written by inference thread, read by main thread
    latest_detections: list[sv.Detections] = [sv.Detections.empty()]
    stop_event = threading.Event()

    def _inference_worker() -> None:
        while not stop_event.is_set():
            if not inference_queue:
                time.sleep(0.005)
                continue
            frame_to_infer = inference_queue.pop()
            result = detect(
                frame_to_infer,
                model,
                confidence_threshold,
                iou_threshold,
                imgsz=imgsz,
                device=device,
                max_det=max_det,
                agnostic_nms=agnostic_nms,
                augment=augment,
                min_person_area=min_person_area,
                use_sahi=use_sahi,
                sahi_slice_wh=sahi_slice_wh,
                sahi_overlap_wh=sahi_overlap_wh,
            )
            latest_detections[0] = result

    worker = threading.Thread(target=_inference_worker, daemon=True)
    worker.start()

    idx = 0
    reconnects = 0

    print("[rtmp_stream] press  q  in the display window to quit.")
    while True:
        ret, frame = cap.read()

        if not ret or frame is None:
            reconnects += 1
            print(f"[rtmp_stream] stream dropout (reconnect {reconnects}/{max_reconnects}) …")
            cap.release()
            if reconnects > max_reconnects:
                print("[rtmp_stream] max reconnects reached, exiting.")
                break
            time.sleep(reconnect_delay)
            try:
                cap = _open_stream(source_stream, max_retries=3, retry_delay=reconnect_delay)
                reconnects = 0
            except RuntimeError as exc:
                print(f"[rtmp_stream] reconnect failed: {exc}")
                break
            continue

        reconnects = 0

        # Every frame_stride frames, push the frame to the inference thread.
        # The thread runs asynchronously; the main loop always uses the most
        # recent detection result available, without waiting.
        if idx % frame_stride == 0:
            inference_queue.append(frame.copy())

        annotated_frame = annotate(
            frame=frame,
            zones=zones,
            zone_annotators=zone_annotators,
            box_annotators=box_annotators,
            detections=latest_detections[0],
        )

        cv2.imshow("RTMP People Count", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        idx += 1

    stop_event.set()
    worker.join(timeout=5)
    cap.release()
    cv2.destroyAllWindows()


def main(
    source_stream: str,
    zone_configuration_path: str | None = None,
    capture_frame_to: str | None = None,
    source_weights_path: str = "yolo26x.pt",
    confidence_threshold: float = 0.10,
    iou_threshold: float = 0.7,
    imgsz: int = 1280,
    device: str | None = None,
    max_det: int = 800,
    agnostic_nms: bool = False,
    augment: bool = False,
    min_person_area: int | None = None,
    use_sahi: bool = False,
    sahi_slice_wh: tuple[int, int] = (640, 640),
    sahi_overlap_wh: tuple[int, int] = (128, 128),
    frame_stride: int = 1,
    max_reconnects: int = 10,
    reconnect_delay: float = 2.0,
) -> None:
    """
    Entry point for RTMP real-time people counting.

    Args:
        source_stream: RTMP stream URL.
        zone_configuration_path: Path to zone JSON. Required unless
            ``capture_frame_to`` is set.
        capture_frame_to: If provided, grab one frame from the stream, save it to
            this path, and exit. Use the saved image with ``draw_zones.py`` to
            create a zone configuration.
        source_weights_path: Ultralytics weights file (default ``yolo26x.pt``).
        confidence_threshold: Detection confidence threshold.
        iou_threshold: IoU threshold (ignored by YOLO26 NMS-free models).
        imgsz: Inference image size.
        device: Inference device (``mps``, ``cuda:0``, ``cpu``, or None for auto).
        max_det: Max detections per frame.
        agnostic_nms: Class-agnostic NMS.
        augment: Test-time augmentation.
        min_person_area: Minimum box area filter in pixels.
        use_sahi: Enable tiled (SAHI-style) inference.
        sahi_slice_wh: Tile size for SAHI.
        sahi_overlap_wh: Tile overlap for SAHI.
        frame_stride: Inference every N frames.
        max_reconnects: Max stream reconnection attempts.
        reconnect_delay: Seconds between reconnection attempts.
    """
    if capture_frame_to is not None:
        capture_frame(source_stream, capture_frame_to)
        return

    if zone_configuration_path is None:
        raise ValueError(
            "--zone_configuration_path is required when not using --capture_frame_to. "
            "Run with --capture_frame_to first to grab a frame, draw zones on it, "
            "then pass the zones JSON here."
        )

    run(
        source_stream=source_stream,
        zone_configuration_path=zone_configuration_path,
        source_weights_path=source_weights_path,
        confidence_threshold=confidence_threshold,
        iou_threshold=iou_threshold,
        imgsz=imgsz,
        device=device,
        max_det=max_det,
        agnostic_nms=agnostic_nms,
        augment=augment,
        min_person_area=min_person_area,
        use_sahi=use_sahi,
        sahi_slice_wh=sahi_slice_wh,
        sahi_overlap_wh=sahi_overlap_wh,
        frame_stride=frame_stride,
        max_reconnects=max_reconnects,
        reconnect_delay=reconnect_delay,
    )


if __name__ == "__main__":
    from jsonargparse import auto_cli, set_parsing_settings

    set_parsing_settings(parse_optionals_as_positionals=True)
    auto_cli(main, as_positional=False)
