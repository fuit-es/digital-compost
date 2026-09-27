"""Streams the simulation to browser viewers over WebSocket.

The simulation stays in Python; browsers only draw. Every server -> browser
message is binary: a little-endian u32 header length, a JSON header, padding
to a 4-byte boundary, then an optional payload (RGBA pixels or float32 records).

Messages (header "t"):
    hello   world size, simulation config (for visual parameters)
    add     new chunk: size, physics vertices, sound curve; payload = RGBA pixels
    rm      removed chunk ids
    glue    new glue: position, meal colors, glow color
    glue_rm removed glue ids
    burst   worm consumption bursts (particles are animated in the browser)
    thumb   worm history thumbnail; payload = RGBA pixels
    panels  worm history panels
    f       frame: flags, glue pulses; payload = float32 records per chunk
            [id, x, y, angle, scale, volume, tint] (tint = packed RGB, -1 if not glued)

Browser -> server messages are JSON text:
    {"t": "size", "w", "h"}   viewport size (the world follows it)
    {"t": "mouse", "x", "y"}  cursor in world coordinates, or x = null when it leaves
    {"t": "key", "k"}         keyboard shortcut / pill bar button
"""
import json
import struct
import threading
from collections import deque
from queue import Queue, Empty
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np
import pygame

from utils.geometry_utils import transform_curve_to_pixels
from utils.logging_utils import get_logger

_logger = get_logger(__name__)

_RECORD_FIELDS = 7


def _pack(header: Dict[str, Any], payload: bytes = b"") -> bytes:
    """Encode one binary message: u32 header length | JSON header | pad to 4 | payload."""
    head = json.dumps(header, separators=(",", ":"), default=str).encode()
    pad = b" " * ((-(4 + len(head))) % 4)
    return struct.pack("<I", len(head)) + head + pad + payload


class _Client:
    """One connected browser: ordered events plus the latest frame (older frames are dropped)."""

    def __init__(self, ws) -> None:
        self.ws = ws
        self.lock = threading.Lock()
        self.wake = threading.Event()
        self.events: Deque[bytes] = deque()
        self.frame: Optional[bytes] = None

    def push(self, events: List[bytes], frame: Optional[bytes]) -> None:
        with self.lock:
            self.events.extend(events)
            if frame is not None:
                self.frame = frame
        self.wake.set()

    def take(self) -> Tuple[List[bytes], Optional[bytes]]:
        with self.lock:
            events = list(self.events)
            self.events.clear()
            frame, self.frame = self.frame, None
        return events, frame


class ViewerHub:
    """Diffs the simulation each frame and fans the changes out to connected browsers."""

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        panel_cfg = config["simulation"]["ui"]["worm_history_panel"]
        self._thumb_size = int(panel_cfg["chunk_preview_size"])

        self._lock = threading.Lock()
        self._clients: List[_Client] = []
        self._pending: List[_Client] = []  # Connected, waiting for a full sync from the main thread

        # Browser input, read by the main loop
        self._commands: Queue = Queue()
        self.mouse_pos: Optional[Tuple[float, float]] = None
        self.requested_size: Optional[Tuple[int, int]] = None

        # What the browsers currently know
        self._chunks: Dict[int, Any] = {}   # uid -> Chunk
        self._glues: Dict[int, Any] = {}    # uid -> Glue
        self._thumbs: Dict[int, bytes] = {}  # history entry uid -> thumb message
        self._panels_key: Optional[Tuple] = None
        self._panels_msg: bytes = _pack({"t": "panels", "worms": []})

    # ----------------------------------------------------------------
    # Connection handling (runs on the WebSocket handler thread)
    # ----------------------------------------------------------------
    def serve(self, ws) -> None:
        """Serve one browser until it disconnects."""
        client = _Client(ws)
        with self._lock:
            self._pending.append(client)
        _logger.info("Viewer connected")
        try:
            while True:
                client.wake.wait(0.05)
                client.wake.clear()
                while True:
                    message = ws.receive(timeout=0)
                    if message is None:
                        break
                    self._handle_message(message)
                events, frame = client.take()
                for event in events:
                    ws.send(event)
                if frame is not None:
                    ws.send(frame)
        except Exception as exc:
            _logger.info("Viewer disconnected (%s)", exc.__class__.__name__)
        finally:
            with self._lock:
                for group in (self._clients, self._pending):
                    if client in group:
                        group.remove(client)

    def _handle_message(self, message: Any) -> None:
        try:
            msg = json.loads(message)
        except (TypeError, ValueError):
            return
        kind = msg.get("t")
        if kind == "mouse":
            x, y = msg.get("x"), msg.get("y")
            self.mouse_pos = None if x is None or y is None else (float(x), float(y))
        elif kind == "key":
            self._commands.put(str(msg.get("k", "")).lower())
        elif kind == "size":
            w, h = int(msg.get("w", 0)), int(msg.get("h", 0))
            if w > 0 and h > 0:
                self.requested_size = (w, h)

    def drain_commands(self) -> List[str]:
        """Keys pressed in any browser since the last call (main thread)."""
        keys = []
        while True:
            try:
                keys.append(self._commands.get_nowait())
            except Empty:
                return keys

    # ----------------------------------------------------------------
    # Publishing (main thread)
    # ----------------------------------------------------------------
    def publish(self, sim, fps: float) -> None:
        """Send this frame's changes to every browser; fully sync newly connected ones."""
        with self._lock:
            pending, self._pending = self._pending, []
            clients = list(self._clients)
        streaming = bool(clients or pending)

        events: List[bytes] = []

        # Chunks
        current = {chunk.uid: chunk for chunk in sim.chunks}
        removed = [uid for uid in self._chunks if uid not in current]
        added = [chunk for uid, chunk in current.items() if uid not in self._chunks]
        self._chunks = current
        if clients:
            events.extend(self._chunk_add_msg(chunk) for chunk in added)
            if removed:
                events.append(_pack({"t": "rm", "ids": removed}))

        # Glues
        current_glues = {glue.uid: glue for glue in sim.glues}
        removed_glues = [uid for uid in self._glues if uid not in current_glues]
        added_glues = [glue for uid, glue in current_glues.items() if uid not in self._glues]
        self._glues = current_glues
        if clients:
            events.extend(self._glue_msg(glue) for glue in added_glues)
            if removed_glues:
                events.append(_pack({"t": "glue_rm", "ids": removed_glues}))

        # Consumption bursts (always drained so they don't pile up)
        bursts = sim.worm_manager.take_bursts()
        if clients and bursts:
            events.append(_pack({"t": "burst", "bursts": bursts}))

        # Worm history panels
        panel_events = self._update_panels(sim.worm_manager) if streaming else []
        if clients:
            events.extend(panel_events)

        frame = self._frame_msg(sim, fps) if streaming else None

        for client in clients:
            client.push(events, frame)
        for client in pending:
            client.push(self._full_sync(sim), frame)

        if pending:
            with self._lock:
                self._clients.extend(pending)

    def _full_sync(self, sim) -> List[bytes]:
        """Everything a freshly connected browser needs to draw the current state."""
        messages = [_pack({"t": "hello", "world": [sim.width, sim.height], "config": self.config})]
        messages.extend(self._chunk_add_msg(chunk) for chunk in self._chunks.values())
        messages.extend(self._glue_msg(glue) for glue in self._glues.values())
        messages.extend(self._thumbs.values())
        messages.append(self._panels_msg)
        return messages

    def _chunk_add_msg(self, chunk) -> bytes:
        surface = chunk.texture_surface
        w, h = surface.get_size()
        header: Dict[str, Any] = {
            "t": "add",
            "id": chunk.uid,
            "w": w,
            "h": h,
            "verts": [round(c, 1) for vertex in chunk.vertices for c in vertex],
        }
        if chunk.curve_data is not None and chunk.original_line_width is not None:
            pts, rgba, surf_w, surf_h, padding = chunk.curve_data
            px, py, _, _ = transform_curve_to_pixels(np.asarray(pts), surf_w, surf_h, padding)
            header["curve"] = {
                "pts": np.round(np.column_stack([px, py]).ravel(), 1).tolist(),
                "rgba": [float(c) for c in rgba],
                "lw": chunk.original_line_width,
            }
        return _pack(header, pygame.image.tobytes(surface, "RGBA"))

    @staticmethod
    def _glue_msg(glue) -> bytes:
        return _pack({
            "t": "glue",
            "id": glue.uid,
            "x": glue.position[0],
            "y": glue.position[1],
            "meals": [list(rgb) for rgb in glue.meal_rgbs],
            "glow": list(glue.glow_rgb),
        })

    def _update_panels(self, worm_manager) -> List[bytes]:
        """Return thumb + panels messages when any visible history changed."""
        worms = [w for w in worm_manager.worms if w.panel_enabled and w.history]
        key = tuple((w.uid, len(w.history), w.appetite_min, w.appetite_max) for w in worms)
        if key == self._panels_key:
            return []
        self._panels_key = key

        messages = []
        visible = {}
        for worm in worms:
            for entry in worm.history:
                msg = self._thumbs.get(entry.uid)
                if msg is None:
                    size = (self._thumb_size, self._thumb_size)
                    thumb = pygame.transform.smoothscale(entry.chunk_surface, size)
                    msg = _pack({"t": "thumb", "id": entry.uid, "w": size[0], "h": size[1]},
                                pygame.image.tobytes(thumb, "RGBA"))
                    messages.append(msg)
                visible[entry.uid] = msg
        self._thumbs = visible

        self._panels_msg = _pack({"t": "panels", "worms": [w.panel_data() for w in worms]})
        messages.append(self._panels_msg)
        return messages

    def _frame_msg(self, sim, fps: float) -> bytes:
        tints = {}
        for glue in sim.glues:
            for glued in glue.glued_chunks:
                r, g, b = glued.tint_rgb
                tints[id(glued.chunk)] = (r << 16) | (g << 8) | b
        volumes = sim.audio_manager.volumes()

        records = np.empty((len(sim.chunks), _RECORD_FIELDS), dtype=np.float32)
        for i, chunk in enumerate(sim.chunks):
            body = chunk.body
            pos = body.position
            records[i] = (
                chunk.uid, pos.x, pos.y, body.angle, chunk.render_scale,
                volumes.get(chunk, 0.0), tints.get(id(chunk), -1),
            )

        header = {
            "t": "f",
            "world": [sim.width, sim.height],
            "fps": round(fps),
            "busy": sim.queue_manager.active_batch_count > 0,
            "glues": [[g.uid, round(g.pulse_time, 3), g.is_full] for g in sim.glues],
            "flags": {
                "dark": sim.dark_mode,
                "torus": sim.boundary_manager.torus_world,
                "history": sim.worm_manager.history_panel_enabled,
                "glue": sim.glue_visuals_enabled,
                "debug": sim.debug_mode,
                "ui": sim.ui_enabled,
                "indicator": sim.processing_indicator_enabled,
                "rec_in": sim.input_manager.recording,
                "rec_out": sim.export_manager.compost_recording,
            },
        }
        return _pack(header, records.tobytes())
