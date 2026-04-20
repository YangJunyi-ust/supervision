"""
Live vehicle detection and annotation on RTMP / RTSP / HTTP / webcam.

Requires: ultralytics, supervision, opencv-python (same as this folder's examples).

Example::

    conda activate ddm_learning
    cd examples/traffic_analysis
    python live_vehicle_stream.py "rtmp://..." --device mps

Larger Ultralytics checkpoints (better accuracy, slower). First run downloads weights::

    python live_vehicle_stream.py "rtmp://..." --device mps --weights yolov8x.pt
    python live_vehicle_stream.py "rtmp://..." --device mps --weights yolo11l.pt
    python live_vehicle_stream.py "rtmp://..." --device mps --weights yolo11x.pt

Default ``--weights`` is ``yolov8l.pt`` (between small and xlarge). Use ``yolov8s.pt`` if FPS is too low.
"""

from __future__ import annotations

import argparse

import cv2
import numpy as np
from ultralytics import YOLO

import supervision as sv

# COCO: car, motorcycle, bus, truck
VEHICLE_CLASS_IDS = [2, 3, 5, 7]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stream_url",
        help="Stream URL (rtmp://, rtsp://, http://) or 0 for default webcam",
    )
    parser.add_argument(
        "--weights",
        default="yolov8l.pt",
        help=(
            "Ultralytics .pt name or path. Larger: yolov8l (default), yolov8x; "
            "YOLO11: yolo11m.pt, yolo11l.pt, yolo11x.pt. Smaller/faster: yolov8s.pt"
        ),
    )
    parser.add_argument(
        "--device",
        default="mps",
        help="Inference device: mps (Apple GPU), cuda, or cpu",
    )
    parser.add_argument("--conf", type=float, default=0.3)
    parser.add_argument("--iou", type=float, default=0.5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    url: str | int = args.stream_url
    if url.isdigit():
        url = int(url)

    model = YOLO(args.weights)
    tracker = sv.ByteTrack()
    box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()
    fps_monitor = sv.FPSMonitor()

    cap = cv2.VideoCapture(url)
    if not cap.isOpened():
        raise SystemExit(
            f"无法打开视频流: {args.stream_url}\n"
            "若 RTMP 失败：确认本机 OpenCV 带 FFmpeg；或先用 ffmpeg 转成本地 RTSP / 用 VLC 测该地址是否可播。"
        )

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("读帧失败或流已结束，重试拉流可在外层加循环。")
                break

            fps_monitor.tick()
            results = model(
                frame,
                verbose=False,
                conf=args.conf,
                iou=args.iou,
                device=args.device,
            )[0]
            detections = sv.Detections.from_ultralytics(results)
            if len(detections):
                mask = np.isin(detections.class_id, np.array(VEHICLE_CLASS_IDS))
                detections = detections[mask]
            detections = tracker.update_with_detections(detections)

            out = box_annotator.annotate(scene=frame.copy(), detections=detections)
            labels = [f"#{int(tid)}" for tid in detections.tracker_id]
            out = label_annotator.annotate(
                scene=out, detections=detections, labels=labels
            )
            out = sv.draw_text(
                scene=out,
                text=f"{fps_monitor.fps:.1f} FPS",
                text_anchor=sv.Point(10, 30),
                background_color=sv.Color.BLACK,
                text_color=sv.Color.WHITE,
            )

            cv2.imshow("vehicles", out)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
