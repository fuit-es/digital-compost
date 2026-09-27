import itertools
import pygame
import pymunk
import math
import random
from typing import List, Tuple, Dict, Any, Optional
import numpy as np
from utils.logging_utils import get_logger
from utils.math_utils import lerp

_logger = get_logger(__name__)

# Stable ids shared with the browser renderer
_next_uid = itertools.count(1)


class Chunk:
    """Handles a single polygon chunk with physics and a surface (drawn by the browser viewer)."""

    @classmethod
    def from_segment(
        cls,
        surface_bytes: bytes,
        surface_size: Tuple[int, int],
        vertices: List[Tuple[float, float]],
        config: Dict[str, Any],
        space: "pymunk.Space",
        spawn_position: Tuple[float, float],
        spawn_velocity: Tuple[float, float],
        downsized: bool = False,
        cached_hsv_color: Optional[Tuple[float, float, float]] = None,
        audio_path: Optional[str] = None,
        curve_data: Optional[Tuple] = None,
        original_line_width: Optional[int] = None,
    ) -> "Chunk":
        """
        Create a Chunk from segmentation result data.
        Fast and main-thread safe - bypasses heavy processing.

        Args:
            surface_bytes: RGBA pixel data as bytes (from numpy array.tobytes())
            surface_size: (width, height)
            vertices: Convex hull vertices centered at origin
            config: Configuration dictionary
            space: Pymunk physics space
            downsized: Whether this chunk was downsampled
            cached_hsv_color: Pre-calculated HSV color (H, S, V) each 0-1
            audio_path: Path to audio file for sound chunks
            curve_data: Curve rendering data for sound chunks
            original_line_width: Line width for sound chunk redrawing

        Returns:
            Chunk: Fully initialized chunk with physics body
        """
        width, height = surface_size

        # Reconstruct pygame surface from bytes
        rgba_array = np.frombuffer(surface_bytes, dtype=np.uint8).reshape((height, width, 4))

        # Create pygame surface
        surface = pygame.Surface((width, height), pygame.SRCALPHA)
        # Convert numpy array to pygame surface
        # pygame expects (width, height, channels) but we have (height, width, channels)
        rgb_transposed = rgba_array[:, :, :3].swapaxes(0, 1)
        alpha_transposed = rgba_array[:, :, 3].swapaxes(0, 1)
        pygame.surfarray.blit_array(surface, rgb_transposed)
        # Set alpha channel
        alpha_surface = pygame.surfarray.pixels_alpha(surface)
        alpha_surface[:] = alpha_transposed
        del alpha_surface  # Release lock

        # Create instance without calling __init__
        chunk = cls.__new__(cls)

        # Set required attributes manually (mirroring __init__)
        chunk.uid = next(_next_uid)
        chunk.config = config
        chunk.space = space
        chunk.segment_surface = surface
        # Surface sent to the viewer; render_scale is relative to it
        chunk.texture_surface = surface
        chunk.render_scale = 1.0
        chunk.audio_path = audio_path
        chunk.original_line_width = original_line_width

        # Handle curve_data - convert list back to numpy array if needed
        if curve_data is not None:
            pts, rgba, surf_w, surf_h, padding = curve_data
            if isinstance(pts, list):
                pts = np.array(pts)
            chunk.curve_data = (pts, rgba, surf_w, surf_h, padding)
        else:
            chunk.curve_data = None

        # Handle gradual shrinkage for downsized chunks
        if downsized:
            compress_cfg = config.get("image", {}).get("compression", {})
            downsample_factor = compress_cfg.get("downsample_factor", 0.5)
            shrink_secs = float(compress_cfg.get("shrink_duration_seconds", 2.0))
            physics_vertices = [(x * downsample_factor, y * downsample_factor) for x, y in vertices]
            chunk._shrink_start_ms = pygame.time.get_ticks()
            chunk._shrink_target_scale = downsample_factor
            chunk._shrink_duration_ms = shrink_secs * 1000
        else:
            physics_vertices = vertices
            chunk._shrink_start_ms = None
            chunk._shrink_target_scale = 1.0
            chunk._shrink_duration_ms = 0

        chunk.vertices = physics_vertices

        # Use pre-calculated HSV color if provided
        if cached_hsv_color is not None:
            chunk.cached_hsv_color = cached_hsv_color
        else:
            from utils.color_utils import calculate_chunk_color
            chunk.cached_hsv_color = calculate_chunk_color(surface)

        chunk._setup_physics_body(physics_vertices, spawn_position, spawn_velocity)

        return chunk

    @classmethod
    def spawn_batch_explosion(
        cls,
        batch_items: List,
        config: Dict[str, Any],
        space: "pymunk.Space",
        screen_width: int,
        screen_height: int,
    ) -> List["Chunk"]:
        """
        Spawn all chunks from a batch as an explosion: reconstruct spatial layout
        at a random screen position, then assign outward velocities.

        Args:
            batch_items: List of ChunkData from background processing
            config: Configuration dictionary
            space: Pymunk physics space
            screen_width: Screen width for bounds clamping
            screen_height: Screen height for bounds clamping

        Returns:
            List of created Chunk instances
        """
        explosion_cfg = (
            config.get("simulation", {})
            .get("physics", {})
            .get("chunk", {})
            .get("explosion", {})
        )
        speed_min = float(explosion_cfg.get("speed_min", 50))
        speed_max = float(explosion_cfg.get("speed_max", 200))

        # Collect source offsets (fall back to (0, 0) if missing)
        offsets = [
            item.source_offset if item.source_offset is not None else (0.0, 0.0)
            for item in batch_items
        ]

        # Compute batch centroid and relative offsets
        cx = sum(o[0] for o in offsets) / len(offsets)
        cy = sum(o[1] for o in offsets) / len(offsets)
        rel_offsets = [(o[0] - cx, o[1] - cy) for o in offsets]

        # Compute bounding box of relative offsets to constrain spawn center
        rel_xs = [r[0] for r in rel_offsets]
        rel_ys = [r[1] for r in rel_offsets]
        min_rel_x, max_rel_x = min(rel_xs), max(rel_xs)
        min_rel_y, max_rel_y = min(rel_ys), max(rel_ys)

        # Allowed spawn center range (ensure all chunks stay on screen)
        spawn_min_x = -min_rel_x
        spawn_max_x = screen_width - max_rel_x
        spawn_min_y = -min_rel_y
        spawn_max_y = screen_height - max_rel_y

        if spawn_min_x > spawn_max_x:
            spawn_x = (spawn_min_x + spawn_max_x) / 2
        else:
            spawn_x = random.uniform(spawn_min_x, spawn_max_x)

        if spawn_min_y > spawn_max_y:
            spawn_y = (spawn_min_y + spawn_max_y) / 2
        else:
            spawn_y = random.uniform(spawn_min_y, spawn_max_y)

        # Create all chunks with explosion positions and velocities
        chunks = []
        for item, rel in zip(batch_items, rel_offsets):
            pos_x = spawn_x + rel[0]
            pos_y = spawn_y + rel[1]

            # Velocity: outward from spawn center
            dist = math.sqrt(rel[0] ** 2 + rel[1] ** 2)
            if dist > 1e-6:
                dir_x = rel[0] / dist
                dir_y = rel[1] / dist
            else:
                angle = random.uniform(0, 2 * math.pi)
                dir_x = math.cos(angle)
                dir_y = math.sin(angle)

            speed = random.uniform(speed_min, speed_max)

            chunk = cls.from_segment(
                surface_bytes=item.surface_bytes,
                surface_size=item.surface_size,
                vertices=item.vertices,
                config=config,
                space=space,
                downsized=item.downsized,
                cached_hsv_color=item.cached_hsv_color,
                audio_path=item.audio_path,
                curve_data=item.curve_data,
                original_line_width=item.original_line_width,
                spawn_position=(pos_x, pos_y),
                spawn_velocity=(dir_x * speed, dir_y * speed),
            )
            chunks.append(chunk)

        return chunks

    def _setup_physics_body(
        self,
        vertices: List[Tuple[float, float]],
        position: Tuple[float, float],
        velocity: Tuple[float, float],
    ) -> None:
        """
        Set up pymunk body and shape for the chunk.

        Args:
            vertices: Polygon vertices for the physics shape
            position: Initial position
            velocity: Initial velocity
        """
        physics_cfg = self.config.get("simulation", {}).get("physics", {}).get("chunk", {})
        mass = physics_cfg.get("mass", 1.0)
        elasticity = physics_cfg.get("elasticity", 1.0)
        friction = physics_cfg.get("friction", 5.0)
        shape_filter_group = physics_cfg.get("shape_filter_group", 1)

        moment = pymunk.moment_for_poly(mass, vertices)
        self.body = pymunk.Body(mass, moment)
        self.body.position = position
        self.body.velocity = velocity

        # Create shape
        self.shape = pymunk.Poly(self.body, vertices)
        self.shape.elasticity = elasticity
        self.shape.friction = friction
        self.shape.filter = pymunk.ShapeFilter(group=shape_filter_group)

        # Add to physics space
        self.space.add(self.body, self.shape)

    def update_shrink(self) -> None:
        """Advance the gradual shrinkage of an entropy-downsized chunk."""
        if self._shrink_start_ms is None:
            return
        elapsed = pygame.time.get_ticks() - self._shrink_start_ms
        t = min(1.0, elapsed / self._shrink_duration_ms) if self._shrink_duration_ms > 0 else 1.0
        self.render_scale = lerp(1.0, self._shrink_target_scale, t)
        if t >= 1.0:
            # Bake the final size so exports and worm history use the shrunk surface
            w, h = self.texture_surface.get_size()
            new_size = (max(1, int(w * self.render_scale)), max(1, int(h * self.render_scale)))
            self.segment_surface = pygame.transform.smoothscale(self.texture_surface, new_size)
            self._shrink_start_ms = None

    @property
    def extent(self) -> float:
        """Largest on-screen dimension of the chunk (pixels)."""
        return max(self.texture_surface.get_size()) * self.render_scale


class GluedChunk:
    """A chunk that has been attracted to glue, tinted with its meal color."""
    def __init__(self, chunk: Chunk, tint_rgb: tuple):
        self.chunk = chunk
        self.tint_rgb = tint_rgb