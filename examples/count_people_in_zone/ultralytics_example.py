import json

import cv2
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO

import supervision as sv

COLORS = sv.ColorPalette.DEFAULT


def load_zones_config(file_path: str) -> list[np.ndarray]:
    """
    Load polygon zone configurations from a JSON file.

    This function reads a JSON file which contains polygon coordinates, and
    converts them into a list of NumPy arrays. Each polygon is represented as
    a NumPy array of coordinates.

    Args:
    file_path (str): The path to the JSON configuration file.

    Returns:
    List[np.ndarray]: A list of polygons, each represented as a NumPy array.
    """
    with open(file_path) as file:
        data = json.load(file)
        return [np.array(polygon, np.int32) for polygon in data["polygons"]]


def initiate_annotators(
    polygons: list[np.ndarray], resolution_wh: tuple[int, int]
) -> tuple[list[sv.PolygonZone], list[sv.PolygonZoneAnnotator], list[sv.BoxAnnotator]]:
    line_thickness = sv.calculate_optimal_line_thickness(resolution_wh=resolution_wh)
    text_scale = sv.calculate_optimal_text_scale(resolution_wh=resolution_wh)

    zones = []
    zone_annotators = []
    box_annotators = []

    for index, polygon in enumerate(polygons):
        zone = sv.PolygonZone(polygon=polygon)
        zone_annotator = sv.PolygonZoneAnnotator(
            zone=zone,
            color=COLORS.by_idx(index),
            thickness=line_thickness,
            text_thickness=line_thickness * 2,
            text_scale=text_scale * 2,
        )
        box_annotator = sv.BoxAnnotator(
            color=COLORS.by_idx(index), thickness=line_thickness
        )
        zones.append(zone)
        zone_annotators.append(zone_annotator)
        box_annotators.append(box_annotator)

    return zones, zone_annotators, box_annotators


def detect(
    frame: np.ndarray,
    model: YOLO,
    confidence_threshold: float = 0.5,
    iou_threshold: float = 0.7,
    imgsz: int = 1280,
    device: str | None = None,
    max_det: int = 500,
    agnostic_nms: bool = False,
    augment: bool = False,
    min_person_area: int | None = None,
    use_sahi: bool = False,
    sahi_slice_wh: tuple[int, int] = (640, 640),
    sahi_overlap_wh: tuple[int, int] = (128, 128),
) -> sv.Detections:
    """
    Detect objects in a frame using a YOLO model, filtering detections by class ID and
        confidence threshold.

    Args:
        frame (np.ndarray): The frame to process, expected to be a NumPy array.
        model (YOLO): The YOLO model used for processing the frame.
        confidence_threshold (float): The confidence threshold for filtering
            detections. Default is 0.5.
        iou_threshold (float): The IoU threshold for non-maximum suppression. For
            end-to-end models such as YOLO26, Ultralytics may ignore this (NMS-free).
        imgsz (int): Letterbox size for inference. Higher values (e.g. 1920) help small
            / distant people in HD video but increase compute and memory.
        device (str | None): Ultralytics device, e.g. ``\"mps\"``, ``\"cuda:0\"``, or
            ``\"cpu\"``. If None, Ultralytics picks a default.
        max_det (int): Max boxes per image. Dense crowds need higher values (e.g. 800);
            default Ultralytics is often 300, which can drop people when crowded.
        agnostic_nms (bool): Class-agnostic NMS; try True if you use a multi-class
            model and see missing overlaps. For person-only COCO models it is minor.
        augment (bool): Ultralytics test-time augmentation; slower but can improve
            recall on hard frames (dense crowds, motion blur).
        min_person_area (int | None): If set, drop boxes smaller than this many
            pixels (width * height). Use when lowering ``confidence_threshold`` to
            suppress speckle false positives while keeping small true distant people
            if the threshold is chosen conservatively.
        use_sahi (bool): If True, use sliced inference (SAHI-style) via
            ``sv.InferenceSlicer``. Splits the frame into overlapping tiles and merges
            results with NMS. Greatly improves recall for small / distant people in
            high-resolution video but is 4-10× slower.
        sahi_slice_wh (tuple[int, int]): Tile size ``(width, height)`` in pixels when
            ``use_sahi=True``. Smaller tiles help very small targets; 640 is a good
            start.
        sahi_overlap_wh (tuple[int, int]): Overlap in pixels ``(width, height)``
            between adjacent tiles. E.g. ``(128, 128)`` on 640 tiles ≈ 20% overlap.
            Higher values reduce missed detections near tile edges.

    Returns:
        sv.Detections: Filtered detections after processing the frame with the YOLO
            model.

    Note:
        This function is specifically tailored for a YOLO model and assumes class ID 0
            for filtering. YOLO26 and similar one-to-one detectors are trained without
            classical NMS at inference; tune recall mainly with ``confidence_threshold``,
            ``imgsz``, and ``max_det``.
    """
    kwargs: dict = {
        "conf": confidence_threshold,
        "iou": iou_threshold,
        "imgsz": imgsz,
        "verbose": False,
        "max_det": max_det,
        "agnostic_nms": agnostic_nms,
        "augment": augment,
    }
    if device:
        kwargs["device"] = device

    if use_sahi:
        def _callback(img: np.ndarray) -> sv.Detections:
            results = model(img, **kwargs)[0]
            dets = sv.Detections.from_ultralytics(results)
            return dets[dets.class_id == 0]

        slicer = sv.InferenceSlicer(
            callback=_callback,
            slice_wh=sahi_slice_wh,
            overlap_wh=sahi_overlap_wh,
            iou_threshold=iou_threshold,
        )
        detections = slicer(frame)
        filter_by_confidence = detections.confidence >= confidence_threshold
        detections = detections[filter_by_confidence]
    else:
        results = model(frame, **kwargs)[0]
        detections = sv.Detections.from_ultralytics(results)
        filter_by_class = detections.class_id == 0
        filter_by_confidence = detections.confidence >= confidence_threshold
        detections = detections[filter_by_class & filter_by_confidence]

    if min_person_area is not None and len(detections):
        xyxy = detections.xyxy
        w = xyxy[:, 2] - xyxy[:, 0]
        h = xyxy[:, 3] - xyxy[:, 1]
        large_enough = (w * h) >= min_person_area
        detections = detections[large_enough]
    return detections


def annotate(
    frame: np.ndarray,
    zones: list[sv.PolygonZone],
    zone_annotators: list[sv.PolygonZoneAnnotator],
    box_annotators: list[sv.BoxAnnotator],
    detections: sv.Detections,
) -> np.ndarray:
    """
    Annotate a frame with zone and box annotations based on given detections.

    Args:
        frame (np.ndarray): The original frame to be annotated.
        zones (List[sv.PolygonZone]): A list of polygon zones used for detection.
        zone_annotators (List[sv.PolygonZoneAnnotator]): A list of annotators for
            drawing zone annotations.
        box_annotators (List[sv.BoxAnnotator]): A list of annotators for
            drawing box annotations.
        detections (sv.Detections): Detections to be used for annotation.

    Returns:
        np.ndarray: The annotated frame.
    """
    annotated_frame = frame.copy()
    for zone, zone_annotator, box_annotator in zip(
        zones, zone_annotators, box_annotators
    ):
        detections_in_zone = detections[zone.trigger(detections=detections)]
        annotated_frame = zone_annotator.annotate(scene=annotated_frame)
        annotated_frame = box_annotator.annotate(
            scene=annotated_frame, detections=detections_in_zone
        )
    return annotated_frame


def main(
    zone_configuration_path: str,
    source_video_path: str,
    source_weights_path: str = "yolo26x.pt",
    target_video_path: str | None = None,
    confidence_threshold: float = 0.3,
    iou_threshold: float = 0.7,
    imgsz: int = 1280,
    device: str | None = None,
    max_det: int = 500,
    agnostic_nms: bool = False,
    augment: bool = False,
    min_person_area: int | None = None,
    use_sahi: bool = False,
    sahi_slice_wh: tuple[int, int] = (640, 640),
    sahi_overlap_wh: tuple[int, int] = (128, 128),
    frame_stride: int = 1,
):
    """
    Counting people in zones with YOLO and Supervision.

    Args:
        zone_configuration_path: Path to the zone configuration JSON file
        source_video_path: Path to the source video file
        source_weights_path: Ultralytics weights, e.g. ``yolo26x.pt`` (default),
            ``yolo26l.pt``, or a local ``.pt`` path. Requires a recent ``ultralytics``
            release that ships YOLO26.
        target_video_path: Path to the target video file (output)
        confidence_threshold: Confidence threshold for the model
        iou_threshold: IOU threshold for the model
        imgsz: Letterbox inference size (try 1920 for small/distant people)
        device: Inference device (``mps``, ``cuda:0``, ``cpu``, or empty for auto)
        max_det: Max detections per frame; raise for dense crowds (e.g. 800)
        agnostic_nms: Passed to Ultralytics NMS (see ``detect`` docstring)
        augment: Enable TTA in ``detect`` (slower, can help recall)
        min_person_area: Optional minimum box area in pixels (see ``detect``)
        use_sahi: Enable sliced (SAHI-style) inference via ``sv.InferenceSlicer``
        sahi_slice_wh: Tile size ``(width, height)`` in pixels when ``use_sahi=True``
        sahi_overlap_wh: Overlap ``(width, height)`` in pixels between tiles
        frame_stride: Only run inference every N frames; skipped frames reuse the
            last detection result. ``1`` = every frame, ``4`` = every 4th frame.
            Output video always contains all frames.
    """
    video_info = sv.VideoInfo.from_video_path(source_video_path)
    polygons = load_zones_config(zone_configuration_path)
    zones, zone_annotators, box_annotators = initiate_annotators(
        polygons=polygons, resolution_wh=video_info.resolution_wh
    )

    model = YOLO(source_weights_path)

    frames_generator = sv.get_video_frames_generator(source_video_path)
    last_detections: sv.Detections = sv.Detections.empty()

    if target_video_path is not None:
        with sv.VideoSink(target_video_path, video_info) as sink:
            for idx, frame in enumerate(
                tqdm(frames_generator, total=video_info.total_frames)
            ):
                if idx % frame_stride == 0:
                    last_detections = detect(
                        frame,
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
                annotated_frame = annotate(
                    frame=frame,
                    zones=zones,
                    zone_annotators=zone_annotators,
                    box_annotators=box_annotators,
                    detections=last_detections,
                )
                sink.write_frame(annotated_frame)
    else:
        for idx, frame in enumerate(
            tqdm(frames_generator, total=video_info.total_frames)
        ):
            if idx % frame_stride == 0:
                last_detections = detect(
                    frame,
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
            annotated_frame = annotate(
                frame=frame,
                zones=zones,
                zone_annotators=zone_annotators,
                box_annotators=box_annotators,
                detections=last_detections,
            )
            cv2.imshow("Processed Video", annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cv2.destroyAllWindows()


if __name__ == "__main__":
    from jsonargparse import auto_cli, set_parsing_settings

    set_parsing_settings(parse_optionals_as_positionals=True)
    auto_cli(main, as_positional=False)
