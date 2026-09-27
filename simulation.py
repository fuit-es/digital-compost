"""Main simulation orchestrator."""
import pygame
import pymunk
from typing import List, Dict, Any, Set, Optional, Tuple

from entities import Worm, Glue, Chunk
from managers import (
    AudioManager,
    BoundaryManager,
    ExportManager,
    InputManager,
    QueueManager,
    WormManager,
)
from utils.logging_utils import get_logger

_logger = get_logger(__name__)


class Simulation:
    """
    Orchestrates the Felzenszwalb-based chunk simulation.
    Delegates responsibilities to specialized managers.
    Rendering happens in the browser viewer (see network/viewer_hub.py).
    """

    def __init__(
        self,
        config: Dict[str, Any],
        space: pymunk.Space,
    ) -> None:
        """
        Initialize the simulation with provided parameters.

        Args:
            config: The entire configuration loaded from YAML.
            space: The physics space.
        """
        self.config = config
        self.space = space
        self.debug_mode = False
        self.dark_mode = config["simulation"].get("dark_mode", False)
        self.glue_visuals_enabled = config["worm"]["glue"]["visual"].get("enabled", True)
        self.ui_enabled = config["simulation"]["ui"].get("enabled", True)
        self.processing_indicator_enabled = config["simulation"]["ui"]["processing_indicator"].get("enabled", True)

        # Local references
        self.width = config["simulation"]["window"]["width"]
        self.height = config["simulation"]["window"]["height"]
        self.colors = config["colors"]

        # Core state
        self.chunks: List[Chunk] = []
        self.glues: List[Glue] = []
        self.should_quit = False  # Signal for main loop to stop

        # Initialize managers
        self.audio_manager = AudioManager(config)
        self.boundary_manager = BoundaryManager(config, space, self.width, self.height)
        self.worm_manager = WormManager(config)
        self.export_manager = ExportManager(config, self.width, self.height)
        self.input_manager = InputManager(config)
        self.queue_manager = QueueManager(config, space, self.width, self.height)

        # Connect queue manager to chunks list and batch completion callback
        self.queue_manager.set_chunks_list(self.chunks)
        self.queue_manager.set_batch_complete_callback(self._on_batch_complete)

    # ----------------------------------------------------------------
    # Delegation methods
    # ----------------------------------------------------------------
    def handle_audio_hover(self, mouse_pos: Optional[Tuple[float, float]]) -> None:
        """Handle proximity-based audio playback (None when no cursor is over the viewer)."""
        self.audio_manager.handle_audio_hover(mouse_pos, self.chunks)

    def toggle_torus_world(self) -> None:
        """Toggle between torus world and rigid wall boundaries."""
        self.boundary_manager.toggle_mode()

    def toggle_history_panel(self) -> None:
        """Toggle history panel visibility."""
        self.worm_manager.toggle_history_panel()

    def _toggle_audio_in_recording(self) -> None:
        """Toggle audio input recording."""
        is_recording, saved_path = self.input_manager.toggle_recording()
        if not is_recording and saved_path:
            self.submit_audio(saved_path)

    def _toggle_compost_recording(self) -> None:
        """Toggle compost output recording."""
        self.export_manager.toggle_compost_recording(self.audio_manager)

    def on_upload_received(self, image_bytes: bytes, target_width: int = None, target_height: int = None) -> None:
        """Handle incoming image upload (from HTTP server)."""
        self.queue_manager.on_upload_received(image_bytes, target_width, target_height)

    def process_pending_uploads(self, max_items: int = None) -> None:
        """Process pending uploads from HTTP queue."""
        if max_items is None:
            max_items = self.config.get("queue", {}).get("max_http_upload_drain", 4)
        self.queue_manager._process_pending_uploads(max_items)

    def receive_chunks(self) -> None:
        """Receive processed chunks from background."""
        self.queue_manager.receive_chunks()

    def cleanup(self) -> None:
        """Clean up resources before shutdown."""
        self.queue_manager.cleanup()
        self.audio_manager.cleanup()

    # ----------------------------------------------------------------
    # Callback for batch completion
    # ----------------------------------------------------------------
    def _on_batch_complete(self, upload_type: str) -> None:
        """Called when a processing batch completes."""
        _logger.info("Spawning worms for %s upload", upload_type)
        glued_ids = self.get_all_glued_chunk_ids()
        available = len(self.chunks) - len(glued_ids)
        self.worm_manager.spawn_worms_for_upload(upload_type, len(self.chunks), available)

    # ----------------------------------------------------------------
    # Viewer commands
    # ----------------------------------------------------------------
    def handle_key(self, key: str) -> None:
        """Apply a keyboard shortcut or pill bar button sent by the viewer."""
        if key == "d":
            self.dark_mode = not self.dark_mode
        elif key == "t":
            self.toggle_torus_world()
        elif key == "h":
            self.toggle_history_panel()
        elif key == "g":
            self.glue_visuals_enabled = not self.glue_visuals_enabled
        elif key == "i":
            self._toggle_audio_in_recording()
        elif key == "o":
            self._toggle_compost_recording()
        elif key == "c":
            self.clear_chunks()
        elif key == "q":
            self.should_quit = True
        elif key == "escape":
            self.ui_enabled = not self.ui_enabled
        elif key == "e":
            self.export_glues()
        elif key == "k":
            self.debug_mode = not self.debug_mode
        elif key == "p":
            self.processing_indicator_enabled = not self.processing_indicator_enabled

    def resize(self, width: int, height: int) -> None:
        """Change the world size to match the viewer."""
        if (width, height) == (self.width, self.height):
            return
        _logger.info("World resized to %dx%d", width, height)
        self.width = width
        self.height = height
        # Glues read the world size from config for torus flocking
        self.config["simulation"]["window"]["width"] = width
        self.config["simulation"]["window"]["height"] = height
        self.boundary_manager.resize(width, height)
        self.export_manager.width, self.export_manager.height = width, height
        self.queue_manager.width, self.queue_manager.height = width, height

    # ----------------------------------------------------------------
    # Upload methods
    # ----------------------------------------------------------------
    def submit_image(self, image: pygame.Surface, target_width: int = None, target_height: int = None) -> None:
        """Submit an image for background processing."""
        self.queue_manager.submit_image(image, target_width, target_height)

    def submit_audio(self, audio_path: str) -> None:
        """Submit an audio file for background processing."""
        self.queue_manager.submit_audio(audio_path)

    # ----------------------------------------------------------------
    # Core update methods
    # ----------------------------------------------------------------
    def update_chunks(self) -> None:
        """Advance chunk animations and process worm and glue behavior."""
        # Apply torus wrapping (internally checks if enabled)
        self.boundary_manager.handle_torus_wrapping()

        # Cache torus_world state for this frame
        torus_world = self.boundary_manager.torus_world

        # Compute glued chunk IDs and available chunks
        glued_chunk_ids = self.get_all_glued_chunk_ids()
        available_chunks = self.get_available_chunks(glued_chunk_ids)

        # Check worm lifecycle with accurate available food count
        self.worm_manager.check_worm_completion(len(available_chunks), self.glues)

        for chunk in self.chunks:
            chunk.update_shrink()

        # Update worm + glues (delegated to WormManager)
        self.worm_manager.update_worm_and_glues(
            self.glues,
            available_chunks,
            self.remove_chunk,
            torus_world
        )

        # Auto-export check (delegated to WormManager)
        expired_glues, schedule_worms = self.worm_manager.get_glues_for_auto_export(
            self.glues,
            len(self.chunks)
        )
        if expired_glues:
            self._do_export(expired_glues, schedule_worms)

    # ----------------------------------------------------------------
    # Chunk availability methods
    # ----------------------------------------------------------------
    def get_all_glued_chunk_ids(self) -> Set[int]:
        """Get the IDs of all chunks that are already glued by any glue."""
        glued_chunk_ids = set()
        for glue in self.glues:
            glued_chunk_ids.update(id(glued.chunk) for glued in glue.glued_chunks)
        return glued_chunk_ids

    def get_available_chunks(self, glued_chunk_ids: Set[int]) -> List[Chunk]:
        """Get chunks available for worm/glue interaction (not already glued)."""
        return [chunk for chunk in self.chunks if id(chunk) not in glued_chunk_ids]

    # ----------------------------------------------------------------
    # Export methods
    # ----------------------------------------------------------------
    def export_glues(self, glues: Optional[List[Glue]] = None, schedule_worms: bool = True) -> None:
        """Export glues to files."""
        # Ensure any dead worm's glue is promoted before building export list
        available_chunks = self.get_available_chunks(self.get_all_glued_chunk_ids())
        self.worm_manager.check_worm_completion(len(available_chunks), self.glues)
        glues_to_export = list(glues) if glues else list(self.glues)
        self._do_export(glues_to_export, schedule_worms)

    def _do_export(self, glues_to_export: List[Glue], schedule_worms: bool) -> None:
        """Perform the actual export."""
        # Schedule worms based on glues removed, not PNGs saved —
        # an empty glue (attracted nothing) still completes a cycle.
        num_glues = len(glues_to_export)

        def on_export_complete(count: int) -> None:
            if schedule_worms and num_glues > 0:
                available = len(self.chunks) - len(self.get_all_glued_chunk_ids())
                self.worm_manager.schedule_worms(num_glues, available_chunks=available)

        self.export_manager.export_glues(
            glues=glues_to_export,
            glues_list=self.glues,
            remove_chunk=self.remove_chunk,
            on_complete=on_export_complete
        )
        # Clean up worm state via WormManager (not ExportManager)
        self.worm_manager.cleanup_after_export(glues_to_export)

    # ----------------------------------------------------------------
    # Clear methods
    # ----------------------------------------------------------------
    def remove_chunk(self, chunk) -> bool:
        """Remove a single chunk from the simulation (physics, tracking, audio)."""
        try:
            if chunk.shape in self.space.shapes or chunk.body in self.space.bodies:
                self.space.remove(chunk.shape, chunk.body)
        except Exception:
            pass
        try:
            self.chunks.remove(chunk)
        except ValueError:
            return False
        self.audio_manager.on_chunk_removed(chunk)
        return True

    def clear_chunks(self) -> None:
        """Remove all chunks from the simulation and reset worms and glues."""
        for chunk in self.chunks:
            try:
                self.space.remove(chunk.shape, chunk.body)
            except Exception:
                pass
        _logger.info("Removed %d chunks from the simulation", len(self.chunks))
        self.chunks.clear()
        self.audio_manager.stop_all()
        self.worm_manager.clear()
        self.glues.clear()
