# Drone Vehicle Speed Estimation

Measure true ground speed of vehicles from a moving DJI drone, using:

- **RTMP live stream** for real-time video
- **DJI Cloud API WebSocket** for drone telemetry (altitude, heading, GPS)
- **YOLO + supervision ByteTrack** for vehicle detection and tracking
- **Lane-marking calibration** (recommended) or GSD formula for pixel→metres conversion

## How It Works

```
RTMP stream ──► YOLO detect ──► ByteTrack ──► SpeedCalculator ──► annotated display
                                                    ▲
DJI WebSocket ──► TelemetryState (alt, GPS, heading)
```

**Speed mode A — lane calibration (recommended, most accurate):**

1. Run `calibrate_lane.py` once to click 4 corners of a road section whose
   real-world dimensions are known (e.g., two lane widths × 20 m).
2. A perspective homography maps any pixel coordinate → real-world metres.
3. Vehicle displacement in metres / dt = speed, independent of altitude & zoom.

**Speed mode B — GSD formula (fallback):**

1. GSD (m/px) = `2 × altitude × tan(hfov / (2 × zoom)) / frame_width`
2. Vehicle pixel displacement → metres via GSD; drone GPS motion subtracted.

## Setup

```bash
pip install -r requirements.txt
```

## Quick Start — with Lane Calibration (Recommended)

### Step 1: Calibrate

```bash
python calibrate_lane.py \
    --rtmp_url rtmp://aplay.wucepro.com/live/1581F8HGX25CC00A17UC-99-0-0 \
    --real_width 7.0 \      # two 3.5 m lanes = 7.0 m
    --real_length 20.0 \    # road section length you can estimate visually
    --output calibration.json
```

A window opens with a frame from the stream.  Click **4 corners** of the
road section in this order: **top-left → top-right → bottom-right → bottom-left**
(i.e., far-left → far-right → near-right → near-left).  Press **Enter** to save.

Common `--real_width` values for Chinese roads:

| Road type        | 1 lane | 2 lanes | 3 lanes |
|------------------|--------|---------|---------|
| Urban / national | 3.5 m  | 7.0 m   | 10.5 m  |
| Highway          | 3.75 m | 7.5 m   | 11.25 m |

### Step 2: Run with calibration

```bash
/opt/anaconda3/envs/car_speed/bin/python drone_speed.py \
    --rtmp_url rtmp://aplay.wucepro.com/live/1581F8HGX25CC00A17UC-99-0-0 \
    --ws_url   ws://139.9.212.1:6789/api/v1/ws/workspace/shilonghw \
    --calibration calibration.json \
    --weights  yolo26s.pt \
    --confidence 0.10 \
    --frame_stride 1 \
    --imgsz 1536 \
    --device mps
```

The HUD will show `SpeedM: lane-calib` when calibrated mode is active.

## Without Calibration (GSD fallback)

```bash
/opt/anaconda3/envs/car_speed/bin/python drone_speed.py \
    --rtmp_url <URL> --ws_url <WS> --weights yolo26s.pt
```

## Usage

### Live display

```bash
python drone_speed.py \
    --rtmp_url rtmp://aplay.wucepro.com/live/1581F8HGX25CC00A17UC-99-0-0 \
    --ws_url   ws://139.9.212.1:6789/api/v1/ws/workspace/shilonghw \
    --weights  yolov8n.pt \
    --camera_hfov 84.0 \
    --confidence  0.3
```

Press **q** in the display window to quit.

### Record to file (headless)

```bash
python drone_speed.py \
    --rtmp_url     rtmp://aplay.wucepro.com/live/1581F8HGX25CC00A17UC-99-0-0 \
    --ws_url       ws://139.9.212.1:6789/api/v1/ws/workspace/shilonghw \
    --weights      yolo11x.pt \
    --target_video output.mp4 \
    --show_preview False \
    --camera_hfov  84.0 \
    --frame_stride 2 \
    --device       cuda:0
```

### All options

| Argument | Default | Description |
|---|---|---|
| `--rtmp_url` | required | RTMP stream URL |
| `--ws_url` | required | DJI Cloud API WebSocket URL |
| `--weights` | `yolov8n.pt` | YOLO weights file |
| `--camera_hfov` | `84.0` | Camera horizontal FOV in degrees |
| `--confidence` | `0.3` | Detection confidence threshold |
| `--iou_threshold` | `0.7` | NMS IoU threshold |
| `--frame_stride` | `2` | Run YOLO every N frames |
| `--target_video` | `None` | Output video path |
| `--device` | auto | `cpu`, `cuda:0`, `mps` |
| `--speed_window` | `1.0` | Speed averaging window (seconds) |
| `--show_preview` | `True` | Show OpenCV window |
| `--imgsz` | `1280` | YOLO inference image size |
| `--max_reconnects` | `10` | RTMP reconnect attempts |
| `--reconnect_delay` | `2.0` | Seconds between reconnects |

## Camera FOV Calibration

`camera_hfov` controls the GSD calculation. Common DJI values:

| Drone / Lens | HFOV |
|---|---|
| DJI Mavic 3 (wide) | 84° |
| DJI M30T (wide) | 83° |
| DJI Mini 3 Pro (wide) | 82.1° |
| DJI Phantom 4 Pro | 84° |
| DJI Mavic 2 Pro | 77° |

For best accuracy, calibrate using a known-distance ground reference:

1. Fly to a fixed altitude over a painted road marking of known length.
2. Count pixels from one end to the other.
3. `GSD = real_length_m / pixel_count`
4. Back-calculate `hfov = 2 × atan(GSD × frame_width / (2 × altitude))` in degrees.

## Notes

- DJI Cloud API OSD data arrives at **~0.5 Hz** (one message every 2 seconds). Between updates, drone position is dead-reckoned from `horizontal_speed` + `attitude_head`. Speed estimates are most accurate when the drone is hovering or flying straight.
- Detected classes are limited to COCO IDs **2, 3, 5, 7** (car, motorcycle, bus, truck). Adjust `VEHICLE_CLASS_IDS` in `drone_speed.py` for custom models.
- For very high altitudes (> 150 m) or oblique camera angles, consider using a `ViewTransformer` calibrated against known ground points — see `speed_utils.ViewTransformer`.
