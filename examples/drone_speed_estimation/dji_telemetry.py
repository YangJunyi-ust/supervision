"""
DJI Cloud API WebSocket telemetry client.

Connects to a DJI Cloud API WebSocket endpoint, receives device OSD messages,
and exposes the latest drone state via a thread-safe snapshot() call.

The DJI Cloud API pushes `device_osd` messages at ~0.5 Hz containing:
  latitude, longitude, altitude, attitude_head, horizontal_speed, vertical_speed

Between OSD updates (every ~2 s), the caller can dead-reckon the drone
position using horizontal_speed + attitude_head from the last snapshot.

Timestamp synchronisation notes
--------------------------------
* ``received_at`` is set from the DJI OSD message's own ``timestamp`` field
  (Unix milliseconds) converted to seconds.  Falling back to ``time.time()``
  only when the field is absent.  This avoids clock-skew between the local
  machine and the drone/server.
* ``snapshot_at(frame_ts)`` interpolates the stored telemetry history to the
  requested time, so callers can pass the frame's *capture* timestamp (not the
  processing timestamp) and get a correctly time-aligned drone state.

Example::

    client = DJITelemetryClient(
        ws_url="ws://139.9.212.1:6789/api/v1/ws/workspace/shilonghw"
    )
    client.start()

    # Use the frame's capture timestamp for alignment:
    state = client.snapshot_at(frame_capture_ts)
    print(state.altitude, state.attitude_head)

    client.stop()
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque

log = logging.getLogger(__name__)

# How many seconds of telemetry history to keep for time-aligned look-up.
_HISTORY_SECONDS = 10.0


@dataclass
class TelemetryState:
    """Snapshot of drone telemetry at one point in time.

    Attributes:
        latitude: WGS-84 latitude in decimal degrees.
        longitude: WGS-84 longitude in decimal degrees.
        altitude: Absolute altitude above sea level in metres.
        elevation: Altitude relative to takeoff point in metres.
        attitude_head: Compass heading (yaw) in degrees (0 = North, 90 = East).
        horizontal_speed: Ground speed in m/s.
        vertical_speed: Climb rate in m/s (positive = ascending).
        timestamp: Unix timestamp (seconds) of this state (used for ordering).
        received_at: Unix timestamp (seconds) from the DJI OSD ``timestamp``
            field (converted from ms), or local ``time.time()`` if unavailable.
        connected: True when the WebSocket connection is alive.
    """

    latitude: float = 0.0
    longitude: float = 0.0
    altitude: float = 0.0
    elevation: float = 0.0
    attitude_head: float = 0.0
    attitude_pitch: float = -90.0   # degrees; DJI: −90 = nadir, 0 = horizontal
    attitude_roll: float = 0.0      # degrees; for future perspective correction
    zoom_factor: float = 1.0        # current optical/digital zoom multiplier
    horizontal_speed: float = 0.0
    vertical_speed: float = 0.0
    timestamp: float = field(default_factory=time.time)
    received_at: float = field(default_factory=time.time)
    connected: bool = False

    def dead_reckon(self, now: float) -> "TelemetryState":
        """Return a copy with GPS position extrapolated to *now*.

        Uses ``horizontal_speed`` and ``attitude_head`` from this state to
        project latitude/longitude forward (or backward) in time.

        Args:
            now: Target Unix timestamp in seconds.

        Returns:
            A new TelemetryState with updated latitude, longitude and timestamp.

        Example::

            aligned = state.dead_reckon(frame_capture_ts)
        """
        dt = now - self.received_at
        if dt == 0 or self.horizontal_speed == 0:
            return TelemetryState(**{**self.__dict__, "timestamp": now})

        dist_m = self.horizontal_speed * dt
        bearing_rad = math.radians(self.attitude_head)

        earth_radius_m = 6_371_000.0
        delta_lat = (dist_m * math.cos(bearing_rad)) / earth_radius_m
        delta_lon = (dist_m * math.sin(bearing_rad)) / (
            earth_radius_m * math.cos(math.radians(self.latitude))
        )

        return TelemetryState(
            latitude=self.latitude + math.degrees(delta_lat),
            longitude=self.longitude + math.degrees(delta_lon),
            altitude=self.altitude,
            elevation=self.elevation,
            attitude_head=self.attitude_head,
            attitude_pitch=self.attitude_pitch,
            attitude_roll=self.attitude_roll,
            zoom_factor=self.zoom_factor,
            horizontal_speed=self.horizontal_speed,
            vertical_speed=self.vertical_speed,
            timestamp=now,
            received_at=self.received_at,
            connected=self.connected,
        )


class DJITelemetryClient:
    """Async WebSocket client for DJI Cloud API telemetry, run in a daemon thread.

    The WebSocket loop runs inside a dedicated asyncio event loop on a background
    thread so that the main (OpenCV) thread is never blocked.

    Args:
        ws_url: Full WebSocket URL, e.g.
            ``ws://139.9.212.1:6789/api/v1/ws/workspace/shilonghw``.
        reconnect_delay: Seconds to wait before reconnecting after a disconnect.
        ping_interval: WebSocket ping interval in seconds (``None`` to disable).

    Example::

        client = DJITelemetryClient(ws_url="ws://host/api/v1/ws/workspace/abc")
        client.start()
        state = client.snapshot_at(time.time())   # time-aligned snapshot
        client.stop()
    """

    def __init__(
        self,
        ws_url: str,
        reconnect_delay: float = 3.0,
        ping_interval: float | None = 20.0,
    ) -> None:
        self._ws_url = ws_url
        self._reconnect_delay = reconnect_delay
        self._ping_interval = ping_interval

        self._lock = threading.Lock()
        self._state = TelemetryState()

        # Ordered history: newest at the right (deque[-1]).
        # Each entry is a TelemetryState with received_at set to the DJI timestamp.
        self._history: Deque[TelemetryState] = deque()

        # threading.Event is safe to use from any thread.
        self._shutdown = threading.Event()

        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the background WebSocket thread.

        Safe to call multiple times; subsequent calls are no-ops if already
        running.

        Example::

            client.start()
        """
        if self._thread is not None and self._thread.is_alive():
            return
        self._shutdown.clear()
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=self._run_loop, daemon=True, name="dji-telemetry-ws"
        )
        self._thread.start()
        log.info("[DJI WS] background thread started for %s", self._ws_url)

    def stop(self) -> None:
        """Signal the WebSocket loop to shut down and join the thread.

        Uses a threading.Event so the asyncio loop can exit cleanly without
        raising ``RuntimeError: Event loop stopped before Future completed``.

        Example::

            client.stop()
        """
        self._shutdown.set()
        if self._loop is not None and not self._loop.is_closed():
            # Schedule the async stop sentinel on the loop's thread.
            self._loop.call_soon_threadsafe(self._loop.stop)
        if self._thread is not None:
            self._thread.join(timeout=5)

    def snapshot(self) -> TelemetryState:
        """Return the latest telemetry state, dead-reckoned to *now*.

        For precise time alignment (e.g. matching a video frame's capture
        time) prefer :meth:`snapshot_at`.

        Example::

            state = client.snapshot()
        """
        return self.snapshot_at(time.time())

    def snapshot_at(self, ts: float) -> TelemetryState:
        """Return telemetry dead-reckoned to the requested timestamp *ts*.

        Searches the stored history for the two OSD messages that bracket *ts*
        and linearly interpolates between them.  Falls back to the nearest
        boundary state with dead reckoning when *ts* is outside the stored
        window.

        This lets callers pass a video frame's *capture* timestamp so the
        drone position is aligned with the frame, not with the current
        processing time.

        Args:
            ts: Target Unix timestamp in seconds.

        Returns:
            A :class:`TelemetryState` projected to time *ts*.

        Example::

            # frame_capture_ts is when the RTMP reader called cap.read()
            state = client.snapshot_at(frame_capture_ts)
        """
        with self._lock:
            if not self._history:
                # No data yet — return default state
                return TelemetryState(**{**self._state.__dict__, "timestamp": ts})

            # Find bracketing entries
            hist = list(self._history)

        # All OSD messages are older than ts: dead-reckon forward from latest
        if ts >= hist[-1].received_at:
            return hist[-1].dead_reckon(ts)

        # ts is older than our earliest history: dead-reckon backward from earliest
        if ts <= hist[0].received_at:
            return hist[0].dead_reckon(ts)

        # Binary search for the two bracketing states
        lo, hi = 0, len(hist) - 1
        while lo + 1 < hi:
            mid = (lo + hi) // 2
            if hist[mid].received_at <= ts:
                lo = mid
            else:
                hi = mid

        s0, s1 = hist[lo], hist[hi]
        span = s1.received_at - s0.received_at
        if span <= 0:
            return s0.dead_reckon(ts)

        # Linear interpolation weight
        alpha = (ts - s0.received_at) / span
        lat = s0.latitude + alpha * (s1.latitude - s0.latitude)
        lon = s0.longitude + alpha * (s1.longitude - s0.longitude)
        alt = s0.altitude + alpha * (s1.altitude - s0.altitude)
        elev = s0.elevation + alpha * (s1.elevation - s0.elevation)
        head = s0.attitude_head + alpha * (s1.attitude_head - s0.attitude_head)
        pitch = s0.attitude_pitch + alpha * (s1.attitude_pitch - s0.attitude_pitch)
        roll = s0.attitude_roll + alpha * (s1.attitude_roll - s0.attitude_roll)
        zoom = s0.zoom_factor + alpha * (s1.zoom_factor - s0.zoom_factor)
        h_spd = s0.horizontal_speed + alpha * (s1.horizontal_speed - s0.horizontal_speed)
        v_spd = s0.vertical_speed + alpha * (s1.vertical_speed - s0.vertical_speed)

        return TelemetryState(
            latitude=lat,
            longitude=lon,
            altitude=alt,
            elevation=elev,
            attitude_head=head,
            attitude_pitch=pitch,
            attitude_roll=roll,
            zoom_factor=zoom,
            horizontal_speed=h_spd,
            vertical_speed=v_spd,
            timestamp=ts,
            received_at=ts,
            connected=s1.connected,
        )

    def telem_lag(self) -> float:
        """Return seconds since the last OSD message was received.

        Values above ~3 s indicate the telemetry stream is stale.

        Example::

            lag = client.telem_lag()
        """
        with self._lock:
            return time.time() - self._state.received_at

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _run_loop(self) -> None:
        """Entry point for the background daemon thread."""
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._connect_loop())
        except RuntimeError as exc:
            # Raised when loop.stop() is called while run_until_complete is
            # active — expected during clean shutdown.
            if "Event loop stopped before Future" not in str(exc):
                log.warning("[DJI WS] loop exited with error: %s", exc)
        finally:
            try:
                self._loop.close()
            except Exception:
                pass

    async def _connect_loop(self) -> None:
        """Reconnect indefinitely until shutdown is requested."""
        try:
            import websockets
        except ImportError as exc:
            raise ImportError(
                "websockets is required: pip install websockets"
            ) from exc

        while not self._shutdown.is_set():
            try:
                log.info("[DJI WS] connecting to %s", self._ws_url)
                async with websockets.connect(
                    self._ws_url,
                    ping_interval=self._ping_interval,
                    open_timeout=10,
                ) as ws:
                    with self._lock:
                        self._state.connected = True
                    log.info("[DJI WS] connected")
                    await self._receive_loop(ws)
            except Exception as exc:
                if self._shutdown.is_set():
                    break
                log.warning("[DJI WS] disconnected: %s", exc)
            finally:
                with self._lock:
                    self._state.connected = False

            if self._shutdown.is_set():
                break

            log.info("[DJI WS] reconnecting in %.1f s…", self._reconnect_delay)
            # Use asyncio.sleep so the loop stays responsive to loop.stop()
            try:
                await asyncio.sleep(self._reconnect_delay)
            except asyncio.CancelledError:
                break

    async def _receive_loop(self, ws) -> None:
        """Read messages from the WebSocket and update internal state."""
        async for raw in ws:
            if self._shutdown.is_set():
                break
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                log.debug("[DJI WS] non-JSON message: %r", raw[:200])
                continue

            biz_code = msg.get("biz_code") or msg.get("bizCode", "")
            if biz_code == "device_osd":
                self._handle_osd(msg)
            else:
                log.debug("[DJI WS] ignored biz_code=%r", biz_code)

    @staticmethod
    def _extract_zoom(data: dict) -> float | None:
        """Search for zoom/focal-length data across all payload sub-keys in *data*.

        DJI Cloud API OSD messages embed camera payload data under keys that
        follow the pattern ``"{type}-{subtype}-{gimbalindex}"``, e.g.
        ``"43-0-0"`` or ``"52-0-0"``.  The zoom information lives there, not
        in the ``"host"`` sub-key.

        Args:
            data: The ``data`` dict from a ``device_osd`` message.

        Returns:
            Zoom factor as a float, or ``None`` if not found.
        """
        import re
        payload_key_re = re.compile(r"^\d+-\d+-\d+$")
        for key, val in data.items():
            if not payload_key_re.match(key) or not isinstance(val, dict):
                continue
            # Field names seen across DJI drone models
            for field in (
                "zoom_factor",
                "optical_zoom_factor",
                "optical_zoom",
                "zoom_ratio",
                "focal_length",   # some models report mm; needs separate handling
            ):
                raw = val.get(field)
                if raw is None:
                    continue
                try:
                    z = float(raw)
                    if 0.1 <= z <= 200:
                        return z
                except (ValueError, TypeError):
                    pass
        return None

    def _handle_osd(self, msg: dict) -> None:
        """Parse a device_osd message and update stored state and history.

        Timestamp priority:
        1. DJI OSD ``timestamp`` field (Unix ms from DJI server clock) → most
           accurate for inter-message alignment.
        2. Local ``time.time()`` at receive — fallback when field is absent.
        """
        data = msg.get("data", {})
        host = data.get("host", {})

        # ── Resolve timestamp ────────────────────────────────────────────────
        local_now = time.time()
        dji_ts_ms = msg.get("timestamp")
        if dji_ts_ms is not None:
            try:
                dji_ts_s = float(dji_ts_ms) / 1000.0
                # Sanity-check: reject if more than 60 s from local clock
                # (avoids using a stale or epoch-zero value)
                if abs(dji_ts_s - local_now) < 60.0:
                    received_at = dji_ts_s
                else:
                    received_at = local_now
            except (ValueError, TypeError):
                received_at = local_now
        else:
            received_at = local_now

        with self._lock:
            s = self._state
            s.latitude = float(host.get("latitude", s.latitude))
            s.longitude = float(host.get("longitude", s.longitude))
            s.altitude = float(host.get("height", host.get("altitude", s.altitude)))
            s.elevation = float(host.get("elevation", s.elevation))
            s.attitude_head = float(host.get("attitude_head", s.attitude_head))
            s.attitude_pitch = float(host.get("attitude_pitch", s.attitude_pitch))
            s.attitude_roll = float(host.get("attitude_roll", s.attitude_roll))
            # Try host-level zoom fields first, then scan payload sub-keys
            raw_zoom = (
                host.get("zoom_factor")
                or host.get("optical_zoom_factor")
                or host.get("optical_zoom")
                or host.get("zoom_ratio")
                or data.get("zoom_factor")
            )
            if raw_zoom is not None:
                try:
                    z = float(raw_zoom)
                    if 0.1 <= z <= 200:
                        s.zoom_factor = z
                except (ValueError, TypeError):
                    pass
            else:
                # Search all payload sub-keys (e.g. "43-0-0", "52-0-0")
                z = self._extract_zoom(data)
                if z is not None:
                    s.zoom_factor = z
            s.horizontal_speed = float(host.get("horizontal_speed", s.horizontal_speed))
            s.vertical_speed = float(host.get("vertical_speed", s.vertical_speed))
            s.received_at = received_at
            s.timestamp = received_at
            s.connected = True

            # Store a snapshot in history for time-aligned look-ups
            entry = TelemetryState(**s.__dict__)
            self._history.append(entry)

            # Prune entries older than _HISTORY_SECONDS
            cutoff = local_now - _HISTORY_SECONDS
            while self._history and self._history[0].received_at < cutoff:
                self._history.popleft()

        log.debug(
            "[DJI WS] OSD  ts=%.3f  lat=%.6f lon=%.6f alt=%.1f m  "
            "head=%.0f°  h_spd=%.1f m/s  lag=%.2f s",
            received_at,
            s.latitude,
            s.longitude,
            s.altitude,
            s.attitude_head,
            s.horizontal_speed,
            local_now - received_at,
        )
