import pygame
import pymunk
import random
import os
import math
import numpy as np
from tkinter import filedialog
from typing import List, Tuple, Dict, Any
from skimage.segmentation import felzenszwalb
from skimage.color import rgba2rgb, rgb2gray
from skimage.util import img_as_ubyte
from skimage.transform import resize
from scipy.spatial import ConvexHull, QhullError
from utils import scale_image_to_fit, quit_program
import sys

# ------------------------------------------------------------
#  ExactMaskChunk: Handles a single polygon chunk with physics and a surface.
# ------------------------------------------------------------
class ExactMaskChunk:
    """Handles a single polygon chunk with physics and a surface."""

    def __init__(
        self,
        segment_surface: pygame.Surface,
        vertices: List[Tuple[float, float]],
        config: Dict[str, Any],
        space: pymunk.Space,
        downsized: bool = False  # Indicates if the segment has been downsized
    ) -> None:
        """
        Initialize an ExactMaskChunk instance.
        
        Args:
            segment_surface (pygame.Surface): Cropped surface of the image segment.
            vertices (List[Tuple[float, float]]): The polygon vertices for the chunk.
            config (Dict[str, Any]): The entire config dictionary loaded from YAML.
            space (pymunk.Space): The Pymunk physics space.
            downsized (bool): Indicates if the segment has been downsized.
        """
        self.config = config
        self.space = space

        sim_cfg = config["simulation"]
        self.width = sim_cfg["window"]["width"]
        self.height = sim_cfg["window"]["height"]
        self.ui_bar_height = sim_cfg["window"]["ui_bar_height"]
        vel_min = sim_cfg["chunk_velocity_min"]
        vel_max = sim_cfg["chunk_velocity_max"]

        if downsized:
            # Resize the segment_surface to a smaller size
            downsample_factor = config["compression"].get("downsample_factor", 0.5)
            new_width = max(1, int(segment_surface.get_width() * downsample_factor))
            new_height = max(1, int(segment_surface.get_height() * downsample_factor))
            self.segment_surface = pygame.transform.scale(segment_surface, (new_width, new_height))
            print(f"Downsized chunk to ({new_width}, {new_height})")
            # Optionally, scale down the vertices as well
            self.vertices = [(x * downsample_factor, y * downsample_factor) for (x, y) in vertices]
        else:
            self.segment_surface = segment_surface
            self.vertices = vertices

        # Compute a valid random position using bounding-box constraints
        random_x, random_y = self._get_random_position_in_bounds(self.vertices)

        # Setup the chunk's body and shape
        mass = 1.0
        moment = pymunk.moment_for_poly(mass, self.vertices)
        self.body = pymunk.Body(mass, moment)
        self.body.position = (random_x, random_y)
        self.body.velocity = (
            random.uniform(vel_min, vel_max),
            random.uniform(vel_min, vel_max),
        )

        self.shape = pymunk.Poly(self.body, self.vertices)
        self.shape.elasticity = 1.0
        self.shape.friction = 0.0
        self.shape.filter = pymunk.ShapeFilter(group=1)

        self.space.add(self.body, self.shape)

        if downsized:
            print(f"Created a downsized chunk at position ({random_x}, {random_y}) with size {self.segment_surface.get_size()}.")
        else:
            print(f"Created a chunk at position ({random_x}, {random_y}) with size {self.segment_surface.get_size()}.")

    def _get_random_position_in_bounds(
        self,
        vertices: List[Tuple[float, float]]
    ) -> Tuple[float, float]:
        """
        Compute a valid random (x,y) for this chunk, ensuring it remains within screen boundaries.

        Args:
            vertices (List[Tuple[float, float]]): Polygon vertices for this chunk.

        Returns:
            (random_x, random_y): A valid position where the chunk can be placed.
        """
        # 1) Compute bounding box
        xs = [v[0] for v in vertices]
        ys = [v[1] for v in vertices]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)

        # 2) Determine allowable placement range
        allowed_min_x = -min_x
        allowed_max_x = self.width - max_x
        allowed_min_y = self.ui_bar_height - min_y
        allowed_max_y = self.height - max_y

        # 3) Generate a valid random position
        if allowed_min_x > allowed_max_x or allowed_min_y > allowed_max_y:
            # If polygon is too large, clamp to a default position
            random_x = max(0, allowed_min_x)
            random_y = max(self.ui_bar_height, allowed_min_y)
        else:
            random_x = random.uniform(allowed_min_x, allowed_max_x)
            random_y = random.uniform(allowed_min_y, allowed_max_y)

        return random_x, random_y

    def draw(self, surface: pygame.Surface, debug_mode: bool = False) -> None:
        """
        Draw the chunk on a Pygame surface.
        
        Args:
            surface (pygame.Surface): The surface to draw on.
            debug_mode (bool): Whether to draw debug outlines.
        """
        pos = self.body.position
        angle_degrees = math.degrees(-self.body.angle)

        rotated_surface = pygame.transform.rotate(self.segment_surface, angle_degrees)
        rect = rotated_surface.get_rect(center=(int(pos.x), int(pos.y)))
        surface.blit(rotated_surface, rect)

        if debug_mode:
            colors = self.config["colors"]
            cos_a = math.cos(self.body.angle)
            sin_a = math.sin(self.body.angle)
            transformed_vertices = [
                (
                    v.x * cos_a - v.y * sin_a + pos.x,
                    v.x * sin_a + v.y * cos_a + pos.y
                )
                for v in self.shape.get_vertices()
            ]
            pygame.draw.polygon(surface, colors["RED"], transformed_vertices, width=1)

# ------------------------------------------------------------
#  ChunkFactory: Responsible for creating ExactMaskChunk objects
# ------------------------------------------------------------
class ChunkFactory:
    """
    Converts labeled segments from an image into ExactMaskChunk objects.
    """

    def __init__(self, config: Dict[str, Any], space: pymunk.Space) -> None:
        """
        Args:
            config (Dict[str, Any]): Global config dictionary.
            space (pymunk.Space): Pymunk space for physics objects.
        """
        self.config = config
        self.space = space

    def create_chunks_from_segments(
        self,
        img: pygame.Surface,
        segments: np.ndarray,
        labels_to_downsize: List[int]
    ) -> List[ExactMaskChunk]:
        """
        Create chunk objects from the labeled image segments.

        Args:
            img (pygame.Surface): The original (scaled) image surface.
            segments (np.ndarray): Label matrix from Felzenszwalb.
            labels_to_downsize (List[int]): List of segment labels that should be downsized.

        Returns:
            List[ExactMaskChunk]: A list of chunk objects.
        """
        chunks: List[ExactMaskChunk] = []

        unique_labels = np.unique(segments)
        for segment_label in unique_labels:
            mask = (segments == segment_label)
            coords = np.argwhere(mask)
            if len(coords) < 3:
                continue

            centroid = coords.mean(axis=0)  # (y, x)
            cy, cx = centroid
            min_y, min_x = coords.min(axis=0)
            max_y, max_x = coords.max(axis=0)

            width = max_x - min_x + 1
            height = max_y - min_y + 1

            # Extract the sub-surface
            raw_segment_surface = pygame.Surface((width, height), pygame.SRCALPHA)
            for y, x in coords:
                raw_segment_surface.set_at((x - min_x, y - min_y), img.get_at((x, y)))

            # Shift coords so centroid is (0,0)
            vertices = [(x - cx, y - cy) for (y, x) in coords]
            vertices_array = np.array(vertices)

            # Attempt to form a hull
            try:
                hull = ConvexHull(vertices_array)
            except QhullError:
                continue
            hull_vertices = [tuple(vertices_array[i]) for i in hull.vertices]

            # Calculate bounding box of hull
            hull_xs = [v[0] for v in hull_vertices]
            hull_ys = [v[1] for v in hull_vertices]
            hull_width = max(hull_xs) - min(hull_xs)
            hull_height = max(hull_ys) - min(hull_ys)

            surface_width = max(1, int(math.ceil(hull_width)))
            surface_height = max(1, int(math.ceil(hull_height)))
            centered_surface = pygame.Surface((surface_width, surface_height), pygame.SRCALPHA)

            offset_x = surface_width / 2
            offset_y = surface_height / 2

            # Blit pixels onto centered surface
            for y in range(height):
                for x in range(width):
                    color = raw_segment_surface.get_at((x, y))
                    if color.a > 0:
                        blit_x = int(x - (cx - min_x) + offset_x)
                        blit_y = int(y - (cy - min_y) + offset_y)
                        if 0 <= blit_x < surface_width and 0 <= blit_y < surface_height:
                            centered_surface.set_at((blit_x, blit_y), color)

            # Check if this segment is marked for downsizing
            downsized = segment_label in labels_to_downsize

            # Create the chunk
            chunk = ExactMaskChunk(
                segment_surface=centered_surface,
                vertices=hull_vertices,
                config=self.config,
                space=self.space,
                downsized=downsized  # Indicate if this chunk should be downsized
            )
            chunks.append(chunk)

        return chunks

# ------------------------------------------------------------
#  ImageProcessor: Responsible for loading & segmenting images
# ------------------------------------------------------------
class ImageProcessor:
    """
    Handles image loading, optional scaling, Felzenszwalb segmentation,
    and compression of low complexity, large segments by marking them for downsizing.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Args:
            config (Dict[str, Any]): The entire config dictionary (for segmentation params, compression thresholds, etc.)
        """
        self.config = config

    def load_image_via_dialog(self) -> pygame.Surface:
        """
        Opens a file dialog to load an image via Tkinter.
        
        Returns:
            pygame.Surface: The loaded image (with alpha), or None if canceled.
        """
        file_path = filedialog.askopenfilename(
            title="Select an image",
            filetypes=[("Image files", ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.gif"))],
        )
        if file_path and os.path.exists(file_path):
            try:
                return pygame.image.load(file_path).convert_alpha()
            except pygame.error as e:
                print(f"Unable to load image: {e}")
        return None

    def segment_image(self, image: pygame.Surface) -> Tuple[np.ndarray, List[int]]:
        """
        Perform Felzenszwalb segmentation on a Pygame image surface,
        then mark segments that are too large and not very complex for downsizing.

        Args:
            image (pygame.Surface): The surface to segment.

        Returns:
            Tuple[np.ndarray, List[int]]: The labeled segments array and list of segment labels to downsize.
        """
        # Convert Pygame surface to numpy array
        numpy_image = pygame.surfarray.array3d(image).swapaxes(0, 1)

        # RGBA → RGB if needed
        if numpy_image.shape[-1] == 4:
            numpy_image = rgba2rgb(numpy_image)
            numpy_image = img_as_ubyte(numpy_image)
            numpy_image = numpy_image[:, :, :3]

        felz_cfg = self.config["segmentation"]["felzenszwalb"]
        segments = felzenszwalb(
            numpy_image,
            scale=felz_cfg["scale"],
            sigma=felz_cfg["sigma"],
            min_size=felz_cfg["min_size"],
        )

        # Compress segments that are too large and not very complex by marking them
        compress_cfg = self.config.get("compression", {})
        max_size = compress_cfg.get("max_size", 5000)          # Default threshold
        min_entropy = compress_cfg.get("min_entropy", 4.0)     # Default entropy threshold

        # Convert RGB image to grayscale for entropy calculation
        grayscale_image = rgb2gray(numpy_image)
        grayscale_image = img_as_ubyte(grayscale_image)  # Convert to 0-255

        unique_labels = np.unique(segments)
        labels_to_downsize: List[int] = []

        for label in unique_labels:
            mask = segments == label
            size = np.sum(mask)
            if size < 3:
                continue  # Ignore very small segments

            # Calculate entropy
            entropy = self._calculate_entropy(grayscale_image, mask)
            print(f"Segment {label}: size={size}, entropy={entropy}")

            if size > max_size and entropy < min_entropy:
                # Mark this segment for downsizing
                labels_to_downsize.append(label)
                print(f"Marking segment {label} for downsizing.")

        return segments, labels_to_downsize

    def _calculate_entropy(self, grayscale_image: np.ndarray, mask: np.ndarray) -> float:
        """
        Calculate the entropy of the given mask in the grayscale image.

        Args:
            grayscale_image (np.ndarray): Grayscale version of the original image.
            mask (np.ndarray): Boolean mask where True represents the segment.

        Returns:
            float: Entropy value.
        """
        # Extract the pixel values within the mask
        pixel_values = grayscale_image[mask]

        if pixel_values.size == 0:
            return 0.0

        # Calculate histogram
        histogram, _ = np.histogram(pixel_values, bins=256, range=(0, 256), density=True)
        histogram = histogram[histogram > 0]  # Remove zero entries to avoid log2 issues

        # Calculate entropy
        entropy = -np.sum(histogram * np.log2(histogram))
        return entropy

# ------------------------------------------------------------
#  UIManager: Responsible for drawing UI elements
# ------------------------------------------------------------
class UIManager:
    """
    Handles creation and rendering of the UI (buttons, bar, debug text).
    """

    def __init__(
        self,
        config: Dict[str, Any],
        screen: pygame.Surface,
        font: pygame.font.Font
    ) -> None:
        """
        Args:
            config (Dict[str, Any]): Global config dict (for colors, button dims, etc.)
            screen (pygame.Surface): The main window surface for drawing.
            font (pygame.font.Font): The font object for text rendering.
        """
        self.config = config
        self.screen = screen
        self.font = font

        sim_cfg = config["simulation"]
        self.width = sim_cfg["window"]["width"]
        self.height = sim_cfg["window"]["height"]
        self.ui_bar_height = sim_cfg["window"]["ui_bar_height"]
        self.colors = config["colors"]

        # Button dims
        self.button_width = sim_cfg["buttons"]["width"]
        self.button_height = sim_cfg["buttons"]["height"]
        self.button_gap = sim_cfg["buttons"]["gap"]

        self.buttons = self._create_buttons()

    def _create_buttons(self) -> List[pygame.Rect]:
        """Create UI button rectangles."""
        upload_rect = pygame.Rect(
            10,
            (self.ui_bar_height - self.button_height) // 2,
            self.button_width,
            self.button_height
        )
        clear_rect = pygame.Rect(
            upload_rect.right + self.button_gap,
            (self.ui_bar_height - self.button_height) // 2,
            self.button_width,
            self.button_height
        )
        quit_rect = pygame.Rect(
            clear_rect.right + self.button_gap,
            (self.ui_bar_height - self.button_height) // 2,
            self.button_width,
            self.button_height
        )
        return [upload_rect, clear_rect, quit_rect]

    def draw_ui(self, debug_mode: bool) -> None:
        """
        Draw the UI bar, buttons, and optional debug text.

        Args:
            debug_mode (bool): Whether to show debug mode status on the UI bar.
        """
        ui_bar_rect = pygame.Rect(0, 0, self.width, self.ui_bar_height)
        pygame.draw.rect(self.screen, self.colors["UI_BG"], ui_bar_rect)

        button_texts = ["Upload Image", "Clear Canvas", "Quit"]
        for rect, text in zip(self.buttons, button_texts):
            self._draw_button(rect, text)

        # Show debug mode
        debug_text = f"Debug: {'ON' if debug_mode else 'OFF'} (Press D to toggle)"
        debug_surface = self.font.render(debug_text, True, self.colors["WHITE"])
        self.screen.blit(debug_surface, (self.width - 300, (self.ui_bar_height - self.button_height) // 2))

    def _draw_button(self, rect: pygame.Rect, text: str) -> None:
        """Draw a single button."""
        pygame.draw.rect(self.screen, self.colors["DARK_GRAY"], rect, border_radius=5)
        text_surface = self.font.render(text, True, self.colors["WHITE"])
        text_rect = text_surface.get_rect(center=rect.center)
        self.screen.blit(text_surface, text_rect)

# ------------------------------------------------------------
#  Simulation: Orchestrates the physics, UI, and chunk management
# ------------------------------------------------------------
class Simulation:
    """
    Orchestrates the Felzenszwalb-based chunk simulation using the above components.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        space: pymunk.Space,
        screen: pygame.Surface,
        font: pygame.font.Font
    ) -> None:
        """
        Initialize the simulation with provided parameters.

        Args:
            config (Dict[str, Any]): The entire configuration loaded from YAML.
            space (pymunk.Space): The physics space.
            screen (pygame.Surface): The main Pygame window surface.
            font (pygame.font.Font): Font for rendering text/UI.
        """
        self.config = config
        self.space = space
        self.screen = screen
        self.font = font
        self.debug_mode = False

        # Build sub-components
        self.ui_manager = UIManager(config, screen, font)
        self.chunk_factory = ChunkFactory(config, space)
        self.image_processor = ImageProcessor(config)

        # Local references
        self.width = config["simulation"]["window"]["width"]
        self.height = config["simulation"]["window"]["height"]
        self.ui_bar_height = config["simulation"]["window"]["ui_bar_height"]
        self.colors = config["colors"]

        self.chunks: List[ExactMaskChunk] = []

        # Create static boundaries
        self._create_boundaries()

    def _create_boundaries(self) -> None:
        """
        Create static boundaries for the simulation area.
        """
        top = self.ui_bar_height
        static_lines = [
            pymunk.Segment(self.space.static_body, (0, top), (self.width, top), 1),
            pymunk.Segment(self.space.static_body, (0, top), (0, self.height), 1),
            pymunk.Segment(self.space.static_body, (0, self.height), (self.width, self.height), 1),
            pymunk.Segment(self.space.static_body, (self.width, top), (self.width, self.height), 1),
        ]
        for line in static_lines:
            line.elasticity = 1.0
            line.friction = 0.0
        self.space.add(*static_lines)

    def draw_ui(self) -> None:
        """
        Draw the UI bar, buttons, and debug status.
        """
        self.ui_manager.draw_ui(self.debug_mode)

    def update_chunks(self) -> None:
        """
        Update (draw) all chunks in the simulation.
        """
        for chunk in self.chunks:
            chunk.draw(self.screen, debug_mode=self.debug_mode)

    def handle_button_click(self, mouse_pos: Tuple[int, int]) -> None:
        """
        Handle button clicks based on mouse position.

        Args:
            mouse_pos (Tuple[int, int]): The (x,y) position of the mouse click.
        """
        upload_rect, clear_rect, quit_rect = self.ui_manager.buttons
        if upload_rect.collidepoint(mouse_pos):
            self.upload_image()
        elif clear_rect.collidepoint(mouse_pos):
            self.clear_chunks()
        elif quit_rect.collidepoint(mouse_pos):
            quit_program()

    def upload_image(self) -> None:
        """
        Let the user pick an image and create chunks from its Felzenszwalb segments.
        """
        image = self.image_processor.load_image_via_dialog()
        if image:
            # Scale the image to fit the screen
            max_width = self.width
            max_height = self.height - self.ui_bar_height
            image = scale_image_to_fit(image, max_width, max_height)
            print(f"Image loaded and scaled to {image.get_size()}.")

            # Segment the image (with size reduction)
            segments, labels_to_downsize = self.image_processor.segment_image(image)
            print(f"Segmentation completed. {len(labels_to_downsize)} segments marked for downsizing.")

            # Create chunks and store them
            new_chunks = self.chunk_factory.create_chunks_from_segments(image, segments, labels_to_downsize)
            self.chunks.extend(new_chunks)
            print(f"Created {len(new_chunks)} new chunks.")

    def clear_chunks(self) -> None:
        """
        Remove all chunks from the simulation.
        """
        for chunk in self.chunks:
            self.space.remove(chunk.shape, chunk.body)
        print(f"Removed {len(self.chunks)} chunks from the simulation.")
        self.chunks.clear()
