import pygame
import random
from typing import List, Tuple, Dict, Any, Optional
from utils import calculate_chunk_color, calculate_color_contrast, hsv_to_rgb_int, rgb_to_hex
from chunk_thingie import Chunk


class WormHistory:
    """Represents a single chunk's history entry for the worm."""
    def __init__(self, chunk_surface: pygame.Surface, hsv_color: Tuple[float, float, float]):
        self.chunk_surface = chunk_surface
        self.hsv_color = hsv_color
        self.rgb_color = hsv_to_rgb_int(hsv_color)
        self.hex_color = rgb_to_hex(self.rgb_color)


class Worm:
    """
    A worm that consumes chunks based on their color contrast and records their colors.
    """
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize a new worm.
        
        Args:
            config (Dict[str, Any]): The configuration dictionary.
        """
        worm_cfg = config["worm"]
        self.max_chunks = worm_cfg["max_chunks"]
        self.min_difference = worm_cfg["min_difference"]  # Minimum color difference required
        self.max_difference = worm_cfg["max_difference"]  # Maximum color difference allowed
        
        # Panel configuration
        self.panel_x = worm_cfg["display_area"]["x"]
        self.panel_y = worm_cfg["display_area"]["y"]
        self.panel_width = worm_cfg["display_area"]["width"]
        self.panel_height = worm_cfg["display_area"]["height"]
        
        # History panel settings
        panel_cfg = worm_cfg["history_panel"]
        self.chunk_preview_size = panel_cfg["chunk_preview_size"]
        self.text_height = panel_cfg["text_height"]
        self.spacing = panel_cfg["spacing"]
        
        # Colors
        self.panel_bg_color = tuple(config["colors"]["PANEL_BG"])
        
        # State
        self.history: List[WormHistory] = []
        self.last_color: Optional[Tuple[float, float, float]] = None
        self.is_dead = False

    def is_valid_next_chunk(self, chunk: Chunk) -> bool:
        """
        Check if chunk has color difference within the allowed range compared to the previous chunk.
        
        Args:
            chunk (Chunk): The chunk to evaluate.
            
        Returns:
            bool: True if the chunk's color difference is between min and max thresholds.
        """
        if self.last_color is None:
            return True
            
        # Calculate this chunk's color
        chunk_color = calculate_chunk_color(chunk.segment_surface)
        
        # Calculate color difference between chunks
        difference = calculate_color_contrast(self.last_color, chunk_color)
        
        # Print debug info
        print(f"Last color (HSV): {self.last_color}")
        print(f"Current color (HSV): {chunk_color}")
        print(f"Color difference: {difference:.2f}")
        print(f"Valid range: {self.min_difference} to {self.max_difference}")
        
        # Check if difference is within thresholds
        return self.min_difference <= difference <= self.max_difference

    def consume_chunk(self, chunk: Chunk) -> None:
        """
        Consume a chunk, recording its color information.
        
        Args:
            chunk (Chunk): The chunk to consume.
        """
        if self.is_dead:
            return
            
        # Calculate and store HSV color
        hsv_color = calculate_chunk_color(chunk.segment_surface)
        self.last_color = hsv_color
        print(f"Consumed chunk with HSV color: {hsv_color}")
        
        # Create history entry with HSV color
        history_entry = WormHistory(chunk.segment_surface, hsv_color)
        self.history.append(history_entry)
        
        # Check if worm should die
        if len(self.history) >= self.max_chunks:
            self.is_dead = True

    def draw(self, surface: pygame.Surface) -> None:
        """
        Draw the worm's history panel.
        
        Args:
            surface (pygame.Surface): The surface to draw on.
        """
        # Draw panel background
        panel_rect = pygame.Rect(self.panel_x, self.panel_y, self.panel_width, self.panel_height)
        pygame.draw.rect(surface, self.panel_bg_color, panel_rect)
        
        # Draw each history entry
        font = pygame.font.SysFont(None, self.text_height)
        y_offset = self.panel_y + self.spacing
        
        for entry in self.history:
            # Draw chunk preview
            preview = pygame.transform.scale(entry.chunk_surface, 
                                          (self.chunk_preview_size, self.chunk_preview_size))
            preview_rect = preview.get_rect(topleft=(self.panel_x + self.spacing, y_offset))
            surface.blit(preview, preview_rect)
            
            # Draw color swatch
            swatch_rect = pygame.Rect(preview_rect.right + self.spacing, y_offset,
                                    self.chunk_preview_size, self.chunk_preview_size)
            pygame.draw.rect(surface, entry.rgb_color, swatch_rect)
            
            # Draw hex code
            hex_text = font.render(entry.hex_color, True, (255, 255, 255))
            text_rect = hex_text.get_rect(midleft=(swatch_rect.right + self.spacing,
                                                 y_offset + self.chunk_preview_size // 2))
            surface.blit(hex_text, text_rect)
            
            y_offset += self.chunk_preview_size + self.spacing

    def select_next_chunk(self, chunks: List[Chunk]) -> Optional[Chunk]:
        """
        Select the next chunk to consume based on color contrast rules.
        
        Args:
            chunks (List[Chunk]): Available chunks to choose from.
            
        Returns:
            Optional[Chunk]: The selected chunk, or None if no valid chunk found.
        """
        if self.is_dead or not chunks:
            return None
            
        # For first chunk, select randomly
        if self.last_color is None:
            return random.choice(chunks)
            
        # Find all valid chunks based on contrast
        valid_chunks = [chunk for chunk in chunks if self.is_valid_next_chunk(chunk)]
        if not valid_chunks:
            # If no valid chunks are found, the worm dies
            print("No valid chunks found - worm dies!")
            self.is_dead = True
            return None
            
        return random.choice(valid_chunks) 