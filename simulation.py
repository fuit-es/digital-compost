import pygame
import pymunk
import random
import os
from tkinter import filedialog
from typing import List, Tuple
import numpy as np
from skimage.segmentation import felzenszwalb
from skimage.color import rgba2rgb
from skimage.util import img_as_ubyte
import math
from scipy.spatial import ConvexHull, QhullError
from utils import scale_image_to_fit, quit_program

class ExactMaskChunk:
    """
    A class to manage chunks with exact masks from the image.
    Each chunk is represented by a convex polygon and associated image segment.
    """

    def __init__(self, segment_surface: pygame.Surface, vertices: List[Tuple[float, float]], sim_params: dict, space: pymunk.Space):
        """
        Initialize an ExactMaskChunk instance.
        """
        self.segment_surface = segment_surface
        self.width = sim_params['width']
        self.height = sim_params['height']
        self.ui_bar_height = sim_params['ui_bar_height']
        self.colors = sim_params['colors']
        self.space = space

        # Compute the bounding box of the polygon
        xs = [v[0] for v in vertices]
        ys = [v[1] for v in vertices]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)

        # Calculate allowed position ranges to keep the chunk entirely within boundaries
        allowed_min_x = -min_x
        allowed_max_x = self.width - max_x
        allowed_min_y = self.ui_bar_height - min_y
        allowed_max_y = self.height - max_y

        if allowed_min_x > allowed_max_x or allowed_min_y > allowed_max_y:
            # If polygon is too large, place it at a default position
            random_x = max(0, allowed_min_x)
            random_y = max(self.ui_bar_height, allowed_min_y)
        else:
            random_x = random.uniform(allowed_min_x, allowed_max_x)
            random_y = random.uniform(allowed_min_y, allowed_max_y)

        mass = 1
        moment = pymunk.moment_for_poly(mass, vertices)
        self.body = pymunk.Body(mass, moment)
        self.body.position = (random_x, random_y)
        self.body.velocity = (random.uniform(-200, 200), random.uniform(-200, 200))

        # Create shape and set a filter so chunks do not collide with each other
        self.shape = pymunk.Poly(self.body, vertices)
        self.shape.elasticity = 1.0
        self.shape.friction = 0.0
        # All chunks share the same group, preventing chunk-chunk collisions
        self.shape.filter = pymunk.ShapeFilter(group=1)

        self.space.add(self.body, self.shape)

    def draw(self, surface: pygame.Surface, debug_mode: bool = False) -> None:
        """
        Draw the chunk on the Pygame surface.
        """
        pos = self.body.position
        angle = -self.body.angle
        angle_degrees = angle * (180.0 / np.pi)

        # Rotate the surface about its center by angle_degrees (counterclockwise)
        rotated_surface = pygame.transform.rotate(self.segment_surface, angle_degrees)
        rect = rotated_surface.get_rect(center=(int(pos.x), int(pos.y)))
        surface.blit(rotated_surface, rect)

        if debug_mode:
            # Draw polygon outline for debugging
            cos_a = math.cos(self.body.angle)
            sin_a = math.sin(self.body.angle)
            vertices = self.shape.get_vertices()
            transformed_vertices = [
                (v.x * cos_a - v.y * sin_a + pos.x, v.x * sin_a + v.y * cos_a + pos.y)
                for v in vertices
            ]
            pygame.draw.polygon(surface, self.colors["RED"], transformed_vertices, 1)

class Simulation:
    """
    A class to manage the overall simulation of Felzenszwalb-based chunks.
    Handles UI, image upload, segmentation into chunks, and physics simulation.
    """
    def __init__(self, sim_params: dict, space: pymunk.Space, screen: pygame.Surface, font: pygame.font.Font):
        """
        Initialize the simulation with provided parameters.
        """
        self.sim_params = sim_params
        self.width = sim_params['width']
        self.height = sim_params['height']
        self.ui_bar_height = sim_params['ui_bar_height']
        self.colors = sim_params['colors']
        self.button_height = sim_params['button_dimensions']['height']
        self.button_width = sim_params['button_dimensions']['width']
        self.button_gap = sim_params['button_dimensions']['gap']
        self.debug_mode = False
        self.chunks: List[ExactMaskChunk] = []
        self.space = space
        self.screen = screen
        self.font = font
        self.buttons = self.create_buttons()
        self.create_boundaries()


    def create_buttons(self) -> List[pygame.Rect]:
        """
        Create button rectangles for the UI.
        """
        upload_rect = pygame.Rect(10, (self.ui_bar_height - self.button_height) // 2, self.button_width, self.button_height)
        clear_rect = pygame.Rect(upload_rect.right + self.button_gap, (self.ui_bar_height - self.button_height) // 2, self.button_width, self.button_height)
        quit_rect = pygame.Rect(clear_rect.right + self.button_gap, (self.ui_bar_height - self.button_height) // 2, self.button_width, self.button_height)
        return [upload_rect, clear_rect, quit_rect]

    def create_boundaries(self) -> None:
        """
        Create static boundaries for the simulation area.
        Chunks will still collide with these boundaries.
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
        self.space.add(*static_lines)

    def draw_ui(self) -> None:
        """
        Draw the UI bar and buttons.
        """
        ui_bar_rect = pygame.Rect(0, 0, self.width, self.ui_bar_height)
        pygame.draw.rect(self.screen, self.colors["UI_BG"], ui_bar_rect)
        button_texts = ["Upload Image", "Clear Canvas", "Quit"]
        for rect, text in zip(self.buttons, button_texts):
            self.draw_button(rect, text)

        # Display Debug mode status
        debug_text = f"Debug: {'ON' if self.debug_mode else 'OFF'} (Press D to toggle)"
        debug_surface = self.font.render(debug_text, True, self.colors["WHITE"])
        self.screen.blit(debug_surface, (self.width - 300, (self.ui_bar_height - self.button_height)//2))

    def draw_button(self, rect: pygame.Rect, text: str) -> None:
        """
        Draw a button with the specified text.
        """
        pygame.draw.rect(self.screen, self.colors["DARK_GRAY"], rect, border_radius=5)
        text_surface = self.font.render(text, True, self.colors["WHITE"])
        text_rect = text_surface.get_rect(center=rect.center)
        self.screen.blit(text_surface, text_rect)

    def create_chunks_from_segments(self, img: pygame.Surface, segments: np.ndarray) -> None:
        """
        Create chunks from the segments using the exact mask.
        """
        for segment_label in np.unique(segments):
            mask = (segments == segment_label)
            coords = np.argwhere(mask)

            # If there are fewer than 3 pixels, can't form a polygon
            if len(coords) < 3:
                continue
            
            # Check if points are collinear
            # points = np.array(coords)
            # if np.linalg.matrix_rank(points - points.mean(axis=0)) < 2:
            #     # Points are collinear (matrix rank < 2)
            #     continue

            centroid = coords.mean(axis=0)
            cy, cx = centroid
            min_y, min_x = coords.min(axis=0)
            max_y, max_x = coords.max(axis=0)

            width = max_x - min_x + 1
            height = max_y - min_y + 1

            raw_segment_surface = pygame.Surface((width, height), pygame.SRCALPHA)
            for y, x in coords:
                raw_segment_surface.set_at((x - min_x, y - min_y), img.get_at((x, y)))

            vertices = [(x - cx, y - cy) for y, x in coords]
            vertices_array = np.array(vertices)

            # Check again if vertices can form a non-degenerate polygon
            # Attempt to create a hull and if it fails (due to less than 3 points, or colinear points), skip
            try:
                hull = ConvexHull(vertices_array)
            except QhullError:
                # Couldn't form a valid polygon, skip this segment
                continue

            hull_vertices = [tuple(vertices_array[v]) for v in hull.vertices]

            hull_xs = [v[0] for v in hull_vertices]
            hull_ys = [v[1] for v in hull_vertices]
            hull_min_x, hull_max_x = min(hull_xs), max(hull_xs)
            hull_min_y, hull_max_y = min(hull_ys), max(hull_ys)

            hull_width = hull_max_x - hull_min_x
            hull_height = hull_max_y - hull_min_y

            surface_width = int(math.ceil(hull_width))
            surface_height = int(math.ceil(hull_height))
            surface_width = max(surface_width, 1)
            surface_height = max(surface_height, 1)

            centered_surface = pygame.Surface((surface_width, surface_height), pygame.SRCALPHA)

            offset_x = surface_width / 2
            offset_y = surface_height / 2

            for y in range(height):
                for x in range(width):
                    color = raw_segment_surface.get_at((x, y))
                    if color.a > 0:
                        blit_x = int(x - (cx - min_x) + offset_x)
                        blit_y = int(y - (cy - min_y) + offset_y)
                        if 0 <= blit_x < surface_width and 0 <= blit_y < surface_height:
                            centered_surface.set_at((blit_x, blit_y), color)

            self.chunks.append(ExactMaskChunk(centered_surface, hull_vertices, sim_params=self.sim_params, space=self.space))

    def clear_chunks(self) -> None:
        """
        Clear all chunks from the simulation.
        """
        for chunk in self.chunks:
            self.space.remove(chunk.shape, chunk.body)
        self.chunks.clear()

    def update_chunks(self) -> None:
        """
        Update and draw all chunks.
        """
        for chunk in self.chunks:
            chunk.draw(self.screen, debug_mode=self.debug_mode)

    def handle_button_click(self, mouse_pos: Tuple[int, int]) -> None:
        """
        Handle button clicks based on mouse position.
        """
        upload_rect, clear_rect, quit_rect = self.buttons
        if upload_rect.collidepoint(mouse_pos):
            self.upload_image()
        elif clear_rect.collidepoint(mouse_pos):
            self.clear_chunks()
        elif quit_rect.collidepoint(mouse_pos):
            quit_program()

    def upload_image(self) -> None:
        """
        Open a file dialog to upload an image and generate chunks.
        """
        file_path = filedialog.askopenfilename(
            title="Select an image",
            filetypes=[("Image files", ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.gif"))],
        )
        if file_path and os.path.exists(file_path):
            image = pygame.image.load(file_path).convert_alpha()
            # Scale the image to fit within the available area
            max_width, max_height = self.width, self.height - self.ui_bar_height
            image = scale_image_to_fit(image, max_width, max_height)

            numpy_image = pygame.surfarray.array3d(image).swapaxes(0, 1)

            # Convert RGBA to RGB if needed
            if numpy_image.shape[-1] == 4:
                numpy_image = rgba2rgb(numpy_image)
                numpy_image = img_as_ubyte(numpy_image)  # Convert to 8-bit
                numpy_image = numpy_image[:, :, :3]  # Drop alpha channel

            # --- Felzenszwalb Segmentation Here ---
            # Hyperparameters to adjust:
            # scale (int, default: 100), sigma (float, default: 0.8), min_size (int, default: 20)
            # Increase scale -> fewer segments; Decrease scale -> more segments
            # Increase sigma -> smoother image, fewer segments; Decrease sigma -> sharper details
            # Increase min_size -> larger minimum segments; Decrease min_size -> smaller segments
            segments = felzenszwalb(numpy_image, scale=100, sigma=0.8, min_size=20)

            self.create_chunks_from_segments(image, segments)