import itertools
import pygame
import random
import math
from typing import List, Tuple, Dict, Any, Optional, Set, Deque
from collections import deque
import numpy as np
from utils.color_utils import calculate_color_contrast, calculate_color_contrast_many, hsv_to_rgb_int
from utils.logging_utils import get_logger
from utils.math_utils import compute_flocking_forces
from entities.chunk import Chunk, GluedChunk

_logger = get_logger(__name__)

# Stable ids shared with the browser renderer
_next_uid = itertools.count(1)


class WormHistory:
    """Represents a single chunk's history entry for the worm."""
    def __init__(self, chunk_surface: pygame.Surface, hsv_color: Tuple[float, float, float],
                 wild: bool = False, stretch: bool = False, contrast_pct: Optional[float] = None):
        self.uid = next(_next_uid)
        self.chunk_surface = chunk_surface
        self.hsv_color = hsv_color
        self.rgb_color = hsv_to_rgb_int(hsv_color)
        self.wild = wild
        self.stretch = stretch
        self.contrast_pct = contrast_pct  # Cone distance % for wild/stretch picks


class Glue:
    """Manages a collection of chunks attracted by a single glue point."""
    def __init__(self, position: Tuple[float, float], worm_history: List[WormHistory], config: Dict[str, Any], source_worm: Optional['Worm'] = None):
        self.uid = next(_next_uid)
        self.position = position
        self.worm_history = worm_history
        self.source_worm = source_worm
        self.config = config
        self.glued_chunks: List[GluedChunk] = []
        self.attracted_chunk_ids: Set[int] = set()  # Track IDs of chunks we've already attracted
        self.evaluated_chunk_ids: Set[int] = set()  # Track IDs of chunks we've already evaluated
        # Timestamp (ms) when this glue first became full ("capped"). None if not yet capped
        self.capped_at_ms: Optional[int] = None
        self.sound_chunk_count: int = 0  # Track sound chunks for efficient cap checking

        # Get glue configuration
        self.glue_cfg = config["worm"]["glue"]
        self.max_glued_chunks = self.glue_cfg["max_glued_chunks"]
        self.sound_cap = self.glue_cfg.get("sound_cap")  # None means no limit
        self.vicinity_cfg = self.glue_cfg["color_attraction"]
        self.min_difference = self.vicinity_cfg["min_difference"]
        self.max_difference = self.vicinity_cfg["max_difference"]

        # Flocking parameters (shared by all glued chunks)
        flocking_cfg = self.glue_cfg["flocking"]
        self.maxspeed = flocking_cfg["maxspeed"]
        self.maxforce = flocking_cfg["maxforce"]
        self.desired_separation = flocking_cfg["desired_separation"]
        self.separate_force_mult = flocking_cfg["separate_force_mult"]
        self.seek_force_mult = flocking_cfg["seek_force_mult"]

        # World dimensions for torus wrapping (updated in place when the viewer resizes)
        self._win_cfg = config["simulation"]["window"]

        # Pulse clock for the glow (the viewer draws it from glue.visual config)
        self.pulse_speed = self.glue_cfg.get("visual", {}).get("pulse_speed", 0.05)
        self.pulse_time = 0
        # Compute colors from worm history
        self._init_history_colors()

    def _init_history_colors(self) -> None:
        """Prepare meal colors and mixed glow color from worm history."""
        # Collect HSV colors from history
        hsv_colors = [entry.hsv_color for entry in self.worm_history if entry and entry.hsv_color]
        if not hsv_colors:
            # Fallback to a default tint if no history; should not happen due to constructor guard
            self.glow_rgb = (255, 255, 255)
            self.meal_rgbs = []
            return

        # Convert each HSV to RGB for wheel segments
        self.meal_rgbs = [hsv_to_rgb_int(hsv) for hsv in hsv_colors]

        # Average HSV for glow color, handling circular hue
        # Hue is in degrees [0,360)
        sum_sin = 0.0
        sum_cos = 0.0
        sum_s = 0.0
        sum_v = 0.0
        n = float(len(hsv_colors))
        for (h, s, v) in hsv_colors:
            rad = h * math.pi / 180.0
            sum_sin += math.sin(rad)
            sum_cos += math.cos(rad)
            sum_s += s
            sum_v += v
        mean_h_rad = math.atan2(sum_sin / n, sum_cos / n)
        mean_h = (mean_h_rad * 180.0 / math.pi) % 360.0
        mean_s = sum_s / n
        mean_v = sum_v / n
        self.glow_rgb = hsv_to_rgb_int((mean_h, mean_s, mean_v))

    def _build_candidate_lists(
        self,
        available_chunks: List[Chunk]
    ) -> Dict[int, Deque[Tuple[float, Chunk]]]:
        """
        Build candidate lists per meal color, sorted by color difference.

        Each new chunk goes to the meal with the smallest color difference within
        the vicinity thresholds (first meal wins ties).

        Returns:
            Dict mapping meal_index to deque of (difference, chunk) tuples
        """
        meal_to_candidates: Dict[int, List[Tuple[float, Chunk]]] = {}

        new_chunks = [
            c for c in available_chunks
            if id(c) not in self.attracted_chunk_ids and id(c) not in self.evaluated_chunk_ids
        ]
        if new_chunks and self.worm_history:
            # One vectorized contrast pass per meal: (meals x new chunks)
            new_hsv = [c.cached_hsv_color for c in new_chunks]
            diffs = np.array([
                calculate_color_contrast_many(entry.hsv_color, new_hsv, self.config)
                for entry in self.worm_history
            ])
            in_range = (diffs >= self.min_difference) & (diffs <= self.max_difference)
            masked = np.where(in_range, diffs, np.inf)
            best_meal = np.argmin(masked, axis=0)
            best_diff = masked[best_meal, np.arange(len(new_chunks))]
            for j in np.nonzero(np.isfinite(best_diff))[0]:
                meal_to_candidates.setdefault(int(best_meal[j]), []).append((float(best_diff[j]), new_chunks[j]))

        self.evaluated_chunk_ids.update(id(c) for c in new_chunks)

        # Sort and convert to deques
        result: Dict[int, Deque[Tuple[float, Chunk]]] = {}
        for idx, candidates in meal_to_candidates.items():
            result[idx] = deque(sorted(candidates, key=lambda t: t[0]))

        return result

    def _record_cap_time(self) -> None:
        """Record the timestamp when glue became full."""
        if self.capped_at_ms is None:
            try:
                self.capped_at_ms = pygame.time.get_ticks()
            except Exception:
                self.capped_at_ms = 0

    def _is_sound_capped(self) -> bool:
        """Check if sound cap has been reached."""
        return self.sound_cap is not None and self.sound_chunk_count >= self.sound_cap

    def try_attract_chunks(self, available_chunks: List[Chunk], available_ids: Set[int]) -> None:
        """Attract chunks in a balanced way across history colors (round-robin)."""
        if len(self.glued_chunks) >= self.max_glued_chunks:
            self._record_cap_time()
            return

        # Clean up stale IDs from removed chunks
        self.evaluated_chunk_ids &= available_ids
        self.attracted_chunk_ids &= available_ids

        # Build candidate lists per meal color
        meal_candidates = self._build_candidate_lists(available_chunks)
        if not meal_candidates:
            return

        # Rank meals from lowest candidate count to highest
        ordered_meals = sorted(meal_candidates.keys(), key=lambda k: len(meal_candidates[k]))

        # Round-robin pick until capacity reached
        capacity_remaining = self.max_glued_chunks - len(self.glued_chunks)
        while capacity_remaining > 0:
            progressed = False
            sound_capped = self._is_sound_capped()

            for meal_idx in ordered_meals:
                candidates = meal_candidates[meal_idx]

                # Skip sound chunks if sound-capped
                while candidates and sound_capped and candidates[0][1].audio_path:
                    candidates.popleft()
                if not candidates:
                    continue

                # Pick next candidate
                _, chosen = candidates.popleft()
                meal_rgb = self.meal_rgbs[meal_idx]
                self.glued_chunks.append(GluedChunk(chosen, meal_rgb))
                self.attracted_chunk_ids.add(id(chosen))
                if chosen.audio_path:
                    self.sound_chunk_count += 1
                capacity_remaining -= 1
                progressed = True

                if capacity_remaining <= 0:
                    break

            if not progressed:
                break

        # Record cap time if now full
        if len(self.glued_chunks) >= self.max_glued_chunks:
            self._record_cap_time()

    def update(self, torus_world: bool = False) -> None:
        """Update the behavior of all glued chunks."""
        self.pulse_time += self.pulse_speed

        n = len(self.glued_chunks)
        if n == 0:
            return

        # Gather positions and velocities from physics bodies
        positions = np.empty((n, 2), dtype=np.float64)
        velocities = np.empty((n, 2), dtype=np.float64)
        for i, gc in enumerate(self.glued_chunks):
            p = gc.chunk.body.position
            v = gc.chunk.body.velocity
            positions[i] = (p.x, p.y)
            velocities[i] = (v.x, v.y)

        # Compute all forces in one vectorized call
        world_size = (self._win_cfg["width"], self._win_cfg["height"]) if torus_world else None
        forces = compute_flocking_forces(
            positions, velocities, self.position,
            self.desired_separation, self.maxspeed, self.maxforce,
            self.separate_force_mult, self.seek_force_mult,
            world_size,
        )

        # Apply combined forces back to physics bodies
        for i, gc in enumerate(self.glued_chunks):
            gc.chunk.body.apply_force_at_world_point(
                (forces[i, 0], forces[i, 1]), gc.chunk.body.position
            )

    @property
    def is_full(self) -> bool:
        return len(self.glued_chunks) >= self.max_glued_chunks


class Worm:
    """
    A worm that consumes chunks based on their color contrast and records their colors.
    """
    def __init__(self, config: Dict[str, Any],
                 appetite_min: float = 0.0, appetite_max: float = 1.0,
                 available_chunks: int = 0):
        """
        Initialize a new worm.

        Args:
            config (Dict[str, Any]): The configuration dictionary.
            appetite_min: Inherited minimum contrast (cone distance %).
            appetite_max: Inherited maximum contrast (cone distance %).
            available_chunks: Number of available (non-glued) chunks at spawn time.
        """
        self.uid = next(_next_uid)
        self.config = config
        worm_cfg = config["worm"]
        hard_max = worm_cfg["max_meals"]
        min_meals = worm_cfg.get("min_meals", 5)
        max_meal_pct = worm_cfg.get("max_meal_pct", 0.02)
        # Dynamic max: % of available chunks, floored by min_meals, capped by hard max
        pct_limit = max(min_meals, int(available_chunks * max_meal_pct))
        self.max_meals = min(hard_max, pct_limit)
        appetite_cfg = worm_cfg["appetite"]
        self.appetite_min = appetite_min
        self.appetite_max = appetite_max
        # Panel configuration (layout and styling are done by the viewer)
        panel_cfg = config["simulation"]["ui"]["worm_history_panel"]
        self.panel_enabled = panel_cfg["enabled"]
        self.show_hsv_text = panel_cfg.get("show_hsv_text", True)
        self.show_contrast_markers = panel_cfg.get("show_contrast_markers", True)

        # Jitter: chance scales as appetite band narrows
        self.jitter_wide = appetite_cfg.get("jitter_wide", 0.05)
        self.jitter_narrow = appetite_cfg.get("jitter_narrow", 0.50)
        self.jitter_curve = appetite_cfg.get("jitter_curve", 3)

        # State
        self.history: List[WormHistory] = []
        self.last_color: Optional[Tuple[float, float, float]] = None
        self.is_dead = False
        self._next_is_wild: bool = False
        self._next_is_stretch: bool = False
        self._next_contrast_pct: Optional[float] = None
        # Last chunk reference and glue
        self.last_chunk: Optional[Chunk] = None
        self.glue: Optional[Glue] = None

        # Glue configuration
        self.glue_enabled = worm_cfg.get("glue", {}).get("enabled", False)

    @property
    def jitter_chance(self) -> float:
        spread = self.appetite_max - self.appetite_min
        return self.jitter_wide + (self.jitter_narrow - self.jitter_wide) * ((1.0 - spread) ** self.jitter_curve)

    def consume_chunk(self, chunk: Chunk) -> None:
        """
        Consume a chunk, recording its color information.

        Args:
            chunk (Chunk): The chunk to consume.
        """
        if self.is_dead:
            return

        self.last_chunk = chunk
        hsv_color = chunk.cached_hsv_color

        self.last_color = hsv_color

        # Create history entry with HSV color
        is_stretch = self._next_is_stretch
        history_entry = WormHistory(
            chunk.segment_surface, hsv_color,
            wild=self._next_is_wild,
            stretch=is_stretch,
            contrast_pct=self._next_contrast_pct,
        )
        self._next_is_wild = False
        self._next_is_stretch = False
        self._next_contrast_pct = None
        self.history.append(history_entry)

        # Stretch meal is the last meal — die immediately after eating it
        if is_stretch or len(self.history) >= self.max_meals:
            self.mark_as_dead()

    def mark_as_dead(self) -> None:
        """Mark the worm as dead and create glue at the last meal site if enabled."""
        if self.is_dead:
            return
        self.is_dead = True
        _logger.debug("Worm died after consuming %d chunks", len(self.history))

        # Create glue at the last meal site if enabled and we have at least one meal
        if self.glue_enabled and self.last_chunk is not None and len(self.history) > 0:
            position = (self.last_chunk.body.position.x, self.last_chunk.body.position.y)
            self.glue = Glue(position, self.history, self.config, source_worm=self)
            _logger.debug("Created glue at (%.1f, %.1f)", position[0], position[1])

    def panel_data(self) -> Dict[str, Any]:
        """
        Describe the worm's history panel for the viewer.

        Returns:
            Dict with the appetite header and one row per meal: thumbnail id,
            swatch color, optional borders, contrast % or HSV text.
        """
        header = f"[{self.appetite_min:.0%}-{self.appetite_max:.0%}] j:{self.jitter_chance:.0%}"
        # Compute consecutive distances and find min/max indices
        cons_dists = []
        cons_min_idx = -1
        cons_max_idx = -1
        if len(self.history) >= 2:
            cons_dists = [
                calculate_color_contrast(
                    self.history[i].hsv_color, self.history[i + 1].hsv_color, self.config
                )
                for i in range(len(self.history) - 1)
            ]
            min_d = min(cons_dists)
            max_d = max(cons_dists)
            if min_d != max_d:
                cons_min_idx = cons_dists.index(min_d) + 1
                cons_max_idx = cons_dists.index(max_d) + 1

        rows = []
        for idx, entry in enumerate(self.history):
            row: Dict[str, Any] = {"thumb": entry.uid, "rgb": list(entry.rgb_color)}

            # Chunk preview border: wild (red) or stretch (yellow)
            if self.show_contrast_markers:
                if entry.wild:
                    row["thumb_border"] = [255, 60, 60]
                elif entry.stretch:
                    row["thumb_border"] = [255, 200, 40]

            # Swatch border and contrast percentage overlay
            is_min = idx == cons_min_idx
            is_max = idx == cons_max_idx
            swatch_pct = None
            pct_color = None
            if self.show_contrast_markers:
                # Swatch border: green for max, blue for min consecutive distance
                if is_max:
                    row["swatch_border"] = [60, 255, 120]
                elif is_min:
                    row["swatch_border"] = [100, 160, 255]

                # Contrast % text: wild/stretch show their own, min/max show consecutive
                if entry.wild:
                    swatch_pct, pct_color = entry.contrast_pct, (255, 60, 60)
                elif entry.stretch:
                    swatch_pct, pct_color = entry.contrast_pct, (255, 200, 40)
                elif is_max:
                    swatch_pct, pct_color = cons_dists[idx - 1], (60, 255, 120)
                elif is_min:
                    swatch_pct, pct_color = cons_dists[idx - 1], (100, 160, 255)

            if swatch_pct is not None:
                row["pct"] = f"{swatch_pct:.0%}"
                row["pct_color"] = list(pct_color)
            elif self.show_hsv_text:
                h, s, v = entry.hsv_color
                row["hsv"] = f"H:{h:.0f}° S:{s:.2f} V:{v:.2f}"
            rows.append(row)

        return {"id": self.uid, "header": header, "rows": rows}

    def select_next_chunk(self, chunks: List[Chunk]) -> Optional[Chunk]:
        """
        Select the next chunk to consume based on color contrast rules.

        - First chunk: random pick.
        - Subsequent: must fall within [appetite_min, appetite_max] of last meal.
        - Stretching: when no valid food in range, eat the nearest out-of-range
          chunk as a last meal, then die immediately.

        Args:
            chunks: Available chunks to choose from.

        Returns:
            The selected chunk, or None if no valid chunk found.
        """
        if self.is_dead:
            return None

        # Die immediately if already at capacity
        if len(self.history) >= self.max_meals:
            self.mark_as_dead()
            return None

        # No food available — die regardless of meal count
        if not chunks:
            self.mark_as_dead()
            return None

        # First chunk: select randomly
        if self.last_color is None:
            return random.choice(chunks)

        # Jitter: chance scales as appetite band narrows (spread 1.0->wide, 0.0->narrow)
        if random.random() < self.jitter_chance:
            best_chunk = max(
                chunks,
                key=lambda c: calculate_color_contrast(self.last_color, c.cached_hsv_color, self.config)
            )
            best_diff = calculate_color_contrast(self.last_color, best_chunk.cached_hsv_color, self.config)
            self._next_is_wild = True
            self._next_contrast_pct = best_diff
            _logger.debug("Worm jitter! eating furthest chunk, contrast=%.4f", best_diff)
            return best_chunk

        # Find valid chunks within appetite range, weighted by contrast
        valid_chunks = []
        weights = []
        for chunk in chunks:
            diff = calculate_color_contrast(self.last_color, chunk.cached_hsv_color, self.config)
            if self.appetite_min <= diff <= self.appetite_max:
                valid_chunks.append(chunk)
                weights.append(diff)
        if valid_chunks:
            try:
                return random.choices(valid_chunks, weights=weights, k=1)[0]
            except ValueError:
                return random.choice(valid_chunks)

        # No valid chunks — stretch (eat one last out-of-range meal, then die)
        return self._stretch_and_pick(chunks)

    def _stretch_and_pick(self, chunks: List[Chunk]) -> Optional[Chunk]:
        """
        Pick the nearest out-of-range chunk as a last meal (worm dies after eating it).

        Finds the chunk whose contrast to last_color is closest to the
        current [appetite_min, appetite_max] range boundary.
        Does not modify appetite — the worm dies after this meal.
        """
        best_chunk = None
        best_gap = float("inf")
        best_diff = 0.0

        for chunk in chunks:
            diff = calculate_color_contrast(self.last_color, chunk.cached_hsv_color, self.config)
            if diff < self.appetite_min:
                gap = self.appetite_min - diff
            else:
                gap = diff - self.appetite_max
            if gap < best_gap:
                best_gap = gap
                best_chunk = chunk
                best_diff = diff

        if best_chunk is None:
            self.mark_as_dead()
            return None

        self._next_is_stretch = True
        self._next_contrast_pct = best_diff
        _logger.debug(
            "Worm stretch pick: contrast=%.4f, range=[%.4f, %.4f] (meal %d)",
            best_diff, self.appetite_min, self.appetite_max, len(self.history) + 1,
        )
        return best_chunk
