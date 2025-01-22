import pygame
import pymunk
import random
import sys
import os
import tkinter as tk
from tkinter import filedialog
from typing import List, Tuple
import numpy as np
from skimage.segmentation import slic
from skimage.color import rgba2rgb
from skimage.util import img_as_ubyte
from scipy.spatial import ConvexHull
import math

# Initialize Tkinter for file dialogs
root = tk.Tk()
root.withdraw()

# Pygame and simulation constants
WIDTH, HEIGHT = 1600, 1000
UI_BAR_HEIGHT = 60
FPS = 60

COLORS = {
    "BLACK": (0, 0, 0),
    "WHITE": (255, 255, 255),
    "DARK_GRAY": (70, 70, 70),
    "UI_BG": (30, 30, 30),
    "RED": (255, 0, 0),
}

# Button dimensions
BUTTON_WIDTH = 120
BUTTON_HEIGHT = 40
BUTTON_GAP = 10

# Pygame initialization
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Image Chunks with SLIC Segmentation and Scaling")
font = pygame.font.SysFont(None, 24)

# Pymunk space initialization
space = pymunk.Space()
space.gravity = (0, 0)

def scale_image_to_fit(img: pygame.Surface, max_width: int, max_height: int) -> pygame.Surface:
    """
    Scale the image to fit within the given max_width and max_height.

    Args:
        img (pygame.Surface): The source image to scale.
        max_width (int): The maximum allowed width.
        max_height (int): The maximum allowed height.

    Returns:
        pygame.Surface: The scaled image surface.
    """
    iw, ih = img.get_size()
    scale = min(max_width / iw, max_height / ih)
    if scale < 1:
        new_w = int(iw * scale)
        new_h = int(ih * scale)
        return pygame.transform.scale(img, (new_w, new_h))
    else:
        return img

class ExactMaskChunk:
    """
    A class to manage chunks with exact masks from the image.
    Each chunk is represented by a convex polygon and associated image segment.
    """

    def __init__(self, segment_surface: pygame.Surface, vertices: List[Tuple[float, float]]):
        """
        Initialize an ExactMaskChunk instance.

        Args:
            segment_surface (pygame.Surface): The image surface for this chunk's segment.
            vertices (List[Tuple[float, float]]): The polygon vertices of the chunk relative to its centroid.
        """
        self.segment_surface = segment_surface

        # Compute the bounding box of the polygon
        xs = [v[0] for v in vertices]
        ys = [v[1] for v in vertices]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        poly_width = max_x - min_x
        poly_height = max_y - min_y

        # Calculate allowed position ranges to keep the chunk entirely within boundaries
        allowed_min_x = -min_x
        allowed_max_x = WIDTH - max_x
        allowed_min_y = UI_BAR_HEIGHT - min_y
        allowed_max_y = HEIGHT - max_y

        if allowed_min_x > allowed_max_x or allowed_min_y > allowed_max_y:
            # If polygon is too large, place it at a default position
            random_x = max(0, allowed_min_x)
            random_y = max(UI_BAR_HEIGHT, allowed_min_y)
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

        space.add(self.body, self.shape)

    def draw(self, surface: pygame.Surface, debug_mode: bool = False) -> None:
        """
        Draw the chunk on the Pygame surface.

        Args:
            surface (pygame.Surface): The surface to draw on.
            debug_mode (bool): If True, also draw the polygon outline for debugging.
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
            pygame.draw.polygon(surface, COLORS["RED"], transformed_vertices, 1)

class SLICSimulation:
    """
    A class to manage the overall simulation of SLIC-based chunks.
    Handles UI, image upload, segmentation into chunks, and physics simulation.
    """

    def __init__(self):
        """
        Initialize the SLICSimulation instance.
        """
        self.chunks: List[ExactMaskChunk] = []
        self.buttons = self.create_buttons()
        self.create_boundaries()
        self.debug_mode = False

    def create_buttons(self) -> List[pygame.Rect]:
        """
        Create button rectangles for the UI.

        Returns:
            List[pygame.Rect]: List of button rects in order: upload, clear, quit.
        """
        upload_rect = pygame.Rect(10, (UI_BAR_HEIGHT - BUTTON_HEIGHT) // 2, BUTTON_WIDTH, BUTTON_HEIGHT)
        clear_rect = pygame.Rect(upload_rect.right + BUTTON_GAP, (UI_BAR_HEIGHT - BUTTON_HEIGHT) // 2, BUTTON_WIDTH, BUTTON_HEIGHT)
        quit_rect = pygame.Rect(clear_rect.right + BUTTON_GAP, (UI_BAR_HEIGHT - BUTTON_HEIGHT) // 2, BUTTON_WIDTH, BUTTON_HEIGHT)
        return [upload_rect, clear_rect, quit_rect]

    def create_boundaries(self) -> None:
        """
        Create static boundaries for the simulation area.
        Chunks will still collide with these boundaries.
        """
        top = UI_BAR_HEIGHT
        static_lines = [
            pymunk.Segment(space.static_body, (0, top), (WIDTH, top), 1),
            pymunk.Segment(space.static_body, (0, top), (0, HEIGHT), 1),
            pymunk.Segment(space.static_body, (0, HEIGHT), (WIDTH, HEIGHT), 1),
            pymunk.Segment(space.static_body, (WIDTH, top), (WIDTH, HEIGHT), 1),
        ]
        for line in static_lines:
            line.elasticity = 1.0
        space.add(*static_lines)

    def draw_ui(self) -> None:
        """
        Draw the UI bar and buttons.
        """
        ui_bar_rect = pygame.Rect(0, 0, WIDTH, UI_BAR_HEIGHT)
        pygame.draw.rect(screen, COLORS["UI_BG"], ui_bar_rect)
        button_texts = ["Upload Image", "Clear Canvas", "Quit"]
        for rect, text in zip(self.buttons, button_texts):
            self.draw_button(rect, text)

        # Display Debug mode status
        debug_text = f"Debug: {'ON' if self.debug_mode else 'OFF'} (Press D to toggle)"
        debug_surface = font.render(debug_text, True, COLORS["WHITE"])
        screen.blit(debug_surface, (WIDTH - 300, (UI_BAR_HEIGHT - BUTTON_HEIGHT)//2))

    def draw_button(self, rect: pygame.Rect, text: str) -> None:
        """
        Draw a button with the specified text.

        Args:
            rect (pygame.Rect): The button rectangle.
            text (str): The text to display.
        """
        pygame.draw.rect(screen, COLORS["DARK_GRAY"], rect, border_radius=5)
        text_surface = font.render(text, True, COLORS["WHITE"])
        text_rect = text_surface.get_rect(center=rect.center)
        screen.blit(text_surface, text_rect)

    def create_chunks_from_segments(self, img: pygame.Surface, segments: np.ndarray) -> None:
        """
        Create chunks from the SLIC segments using the exact mask.

        Args:
            img (pygame.Surface): The input image surface.
            segments (np.ndarray): The segment labels from SLIC.
        """
        for segment_label in np.unique(segments):
            # Get the mask for the current segment
            mask = (segments == segment_label)

            # Coordinates of the segment's pixels
            coords = np.argwhere(mask)  # coords: [(y, x), ...]

            if len(coords) == 0:
                continue  # Skip empty segments

            # Compute centroid of the segment in image coordinates
            centroid = coords.mean(axis=0)
            cy, cx = centroid

            # Compute bounding box of the segment
            min_y, min_x = coords.min(axis=0)
            max_y, max_x = coords.max(axis=0)
            width = max_x - min_x + 1
            height = max_y - min_y + 1

            # Create a tight surface for the segment
            raw_segment_surface = pygame.Surface((width, height), pygame.SRCALPHA)
            for y, x in coords:
                raw_segment_surface.set_at((x - min_x, y - min_y), img.get_at((x, y)))

            # Shift vertices so the polygon's centroid is at (0,0)
            vertices = [(x - cx, y - cy) for y, x in coords]

            # Compute the convex hull for a clean, convex polygon
            vertices_array = np.array(vertices)
            if len(vertices_array) < 3:
                # ConvexHull requires at least 3 points
                continue
            hull = ConvexHull(vertices_array)
            hull_vertices = [tuple(vertices_array[v]) for v in hull.vertices]

            # Calculate the bounding box of the hull
            hull_xs = [v[0] for v in hull_vertices]
            hull_ys = [v[1] for v in hull_vertices]
            hull_min_x, hull_max_x = min(hull_xs), max(hull_xs)
            hull_min_y, hull_max_y = min(hull_ys), max(hull_ys)
            hull_width = hull_max_x - hull_min_x
            hull_height = hull_max_y - hull_min_y

            # Determine the size of the new surface to center the image
            surface_width = int(math.ceil(hull_width))
            surface_height = int(math.ceil(hull_height))

            # Ensure the surface is at least as big as the hull
            surface_width = max(surface_width, 1)
            surface_height = max(surface_height, 1)

            # Create a new surface centered on the hull's centroid
            centered_surface = pygame.Surface((surface_width, surface_height), pygame.SRCALPHA)

            # Calculate the offset to center the image
            # The polygon centroid (0,0) should align with (surface_width / 2, surface_height / 2)
            offset_x = surface_width / 2
            offset_y = surface_height / 2

            # Blit the raw segment surface onto the centered surface with the calculated offset
            for y in range(height):
                for x in range(width):
                    color = raw_segment_surface.get_at((x, y))
                    if color.a > 0:
                        # Calculate the world position relative to the centroid
                        world_x = x + min_x - cx
                        world_y = y + min_y - cy

                        # Map to centered surface coordinates
                        # Since the polygon is centered at (0,0), the centroid of the polygon corresponds to the center of the surface
                        blit_x = int(x - (cx - min_x) + offset_x)
                        blit_y = int(y - (cy - min_y) + offset_y)

                        # Ensure we don't go out of bounds
                        if 0 <= blit_x < surface_width and 0 <= blit_y < surface_height:
                            centered_surface.set_at((blit_x, blit_y), color)

            # Now the polygon centroid (0,0) aligns with the surface center (surface_width / 2, surface_height / 2)

            # Create the chunk and add it to the simulation
            self.chunks.append(ExactMaskChunk(centered_surface, hull_vertices))

    def clear_chunks(self) -> None:
        """
        Clear all chunks from the simulation.
        """
        for chunk in self.chunks:
            space.remove(chunk.shape, chunk.body)
        self.chunks.clear()

    def update_chunks(self) -> None:
        """
        Update and draw all chunks.
        """
        for chunk in self.chunks:
            chunk.draw(screen, debug_mode=self.debug_mode)

    def handle_button_click(self, mouse_pos: Tuple[int, int]) -> None:
        """
        Handle button clicks based on mouse position.

        Args:
            mouse_pos (Tuple[int, int]): The position of the mouse click.
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
            max_width, max_height = WIDTH, HEIGHT - UI_BAR_HEIGHT
            image = scale_image_to_fit(image, max_width, max_height)

            numpy_image = pygame.surfarray.array3d(image).swapaxes(0, 1)

            # Convert RGBA to RGB if needed
            if numpy_image.shape[-1] == 4:
                numpy_image = rgba2rgb(numpy_image)
                numpy_image = img_as_ubyte(numpy_image)  # Convert to 8-bit
                numpy_image = numpy_image[:, :, :3]  # Drop alpha channel

            # Apply SLIC segmentation on the scaled image
            segments = slic(numpy_image, n_segments=100, compactness=10)
            self.create_chunks_from_segments(image, segments)

def quit_program() -> None:
    """
    Quit the program.
    """
    pygame.quit()
    sys.exit()

def main() -> None:
    """
    Main loop for the simulation.
    """
    clock = pygame.time.Clock()
    simulation = SLICSimulation()
    running = True

    while running:
        mouse_pos = pygame.mouse.get_pos()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                simulation.handle_button_click(mouse_pos)
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_d:
                    # Toggle debug mode
                    simulation.debug_mode = not simulation.debug_mode

        screen.fill(COLORS["BLACK"])
        simulation.draw_ui()
        space.step(1 / FPS)
        simulation.update_chunks()
        pygame.display.flip()
        clock.tick(FPS)

    quit_program()

if __name__ == "__main__":
    main()