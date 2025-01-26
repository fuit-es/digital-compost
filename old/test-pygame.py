import pygame
import pymunk
import random
import sys
import os
import tkinter as tk
from tkinter import filedialog
from typing import List, Tuple

# Initialize Tkinter for file dialogs
root = tk.Tk()
root.withdraw()

# Pygame and simulation constants
WIDTH, HEIGHT = 800, 600
UI_BAR_HEIGHT = 60
PARTICLE_RADIUS = 2
INITIAL_SCALE = 0.2
FPS = 60
COLORS = {
    "BLACK": (0, 0, 0),
    "WHITE": (255, 255, 255),
    "DARK_GRAY": (70, 70, 70),
    "UI_BG": (30, 30, 30),
}

# Button dimensions
BUTTON_WIDTH = 120
BUTTON_HEIGHT = 40
BUTTON_GAP = 10

# Pygame initialization
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Image Pixel Particles with Wall Bounds")
font = pygame.font.SysFont(None, 24)

# Pymunk space initialization
space = pymunk.Space()
space.gravity = (0, 0)


class Particle:
    """A class to manage particles in the simulation."""

    def __init__(self, x: float, y: float, color: Tuple[int, int, int]):
        self.body = pymunk.Body(1, pymunk.moment_for_circle(1, 0, PARTICLE_RADIUS))
        self.body.position = x, y
        self.body.velocity = random.uniform(-100, 100), random.uniform(-100, 100)
        self.shape = pymunk.Circle(self.body, PARTICLE_RADIUS)
        self.shape.color = color
        self.shape.elasticity = 1.0
        self.shape.friction = 0.0
        space.add(self.body, self.shape)

    def draw(self, surface: pygame.Surface) -> None:
        """Draw the particle on the Pygame surface."""
        pos = self.body.position
        if pos.y >= UI_BAR_HEIGHT:
            pygame.draw.circle(surface, self.shape.color, (int(pos.x), int(pos.y)), PARTICLE_RADIUS)


class ParticleSimulation:
    """A class to manage the overall simulation."""

    def __init__(self):
        self.particles: List[Particle] = []
        self.buttons = self.create_buttons()
        self.create_boundaries()

    def create_buttons(self) -> List[pygame.Rect]:
        """Create button rectangles for the UI."""
        upload_rect = pygame.Rect(10, (UI_BAR_HEIGHT - BUTTON_HEIGHT) // 2, BUTTON_WIDTH, BUTTON_HEIGHT)
        clear_rect = pygame.Rect(upload_rect.right + BUTTON_GAP, (UI_BAR_HEIGHT - BUTTON_HEIGHT) // 2, BUTTON_WIDTH, BUTTON_HEIGHT)
        quit_rect = pygame.Rect(clear_rect.right + BUTTON_GAP, (UI_BAR_HEIGHT - BUTTON_HEIGHT) // 2, BUTTON_WIDTH, BUTTON_HEIGHT)
        return [upload_rect, clear_rect, quit_rect]

    def create_boundaries(self) -> None:
        """Create static boundaries for the simulation area."""
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
        """Draw the UI bar and buttons."""
        ui_bar_rect = pygame.Rect(0, 0, WIDTH, UI_BAR_HEIGHT)
        pygame.draw.rect(screen, COLORS["UI_BG"], ui_bar_rect)
        button_texts = ["Upload Image", "Clear Canvas", "Quit"]
        for rect, text in zip(self.buttons, button_texts):
            self.draw_button(rect, text)

    def draw_button(self, rect: pygame.Rect, text: str) -> None:
        """Draw a button with the specified text."""
        pygame.draw.rect(screen, COLORS["DARK_GRAY"], rect, border_radius=5)
        text_surface = font.render(text, True, COLORS["WHITE"])
        text_rect = text_surface.get_rect(center=rect.center)
        screen.blit(text_surface, text_rect)

    def create_particles_from_image(self, img: pygame.Surface, start_x: int, start_y: int) -> None:
        """Create particles based on the given image."""
        w, h = img.get_size()
        px_array = pygame.surfarray.array3d(img)
        for x in range(w):
            for y in range(h):
                color = px_array[x, y]
                particle_x = start_x + x * (PARTICLE_RADIUS * 2)
                particle_y = start_y + y * (PARTICLE_RADIUS * 2)
                self.particles.append(Particle(particle_x, particle_y, color))

    def clear_particles(self) -> None:
        """Clear all particles from the simulation."""
        for particle in self.particles:
            space.remove(particle.shape, particle.body)
        self.particles.clear()

    def update_particles(self) -> None:
        """Update and draw all particles."""
        for particle in self.particles:
            particle.draw(screen)

    def handle_button_click(self, mouse_pos: Tuple[int, int]) -> None:
        """Handle button clicks based on mouse position."""
        upload_rect, clear_rect, quit_rect = self.buttons
        if upload_rect.collidepoint(mouse_pos):
            self.upload_image()
        elif clear_rect.collidepoint(mouse_pos):
            self.clear_particles()
        elif quit_rect.collidepoint(mouse_pos):
            quit_program()

    def upload_image(self) -> None:
        """Open a file dialog to upload an image and generate particles."""
        file_path = filedialog.askopenfilename(
            title="Select an image",
            filetypes=[("Image files", ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.gif"))],
        )
        if file_path and os.path.exists(file_path):
            image = pygame.image.load(file_path)
            image, _ = self.scale_image_to_fit(image, INITIAL_SCALE, WIDTH, HEIGHT - UI_BAR_HEIGHT)
            start_x = random.randint(0, max(0, WIDTH - image.get_width() * PARTICLE_RADIUS * 2))
            start_y = random.randint(UI_BAR_HEIGHT, max(UI_BAR_HEIGHT, HEIGHT - image.get_height() * PARTICLE_RADIUS * 2))
            self.create_particles_from_image(image, start_x, start_y)

    @staticmethod
    def scale_image_to_fit(img: pygame.Surface, scale: float, max_width: int, max_height: int) -> Tuple[pygame.Surface, float]:
        """Scale the image to fit within specified dimensions."""
        iw, ih = img.get_size()
        scaled_w = int(iw * scale)
        scaled_h = int(ih * scale)
        if scaled_w * PARTICLE_RADIUS * 2 > max_width or scaled_h * PARTICLE_RADIUS * 2 > max_height:
            fit_scale = min(max_width / (iw * PARTICLE_RADIUS * 2), max_height / (ih * PARTICLE_RADIUS * 2))
            scaled_w = int(iw * fit_scale)
            scaled_h = int(ih * fit_scale)
            scale = fit_scale
        return pygame.transform.scale(img, (scaled_w, scaled_h)), scale


def quit_program() -> None:
    """Quit the program."""
    pygame.quit()
    sys.exit()


def main() -> None:
    """Main loop for the simulation."""
    clock = pygame.time.Clock()
    simulation = ParticleSimulation()
    running = True

    while running:
        mouse_pos = pygame.mouse.get_pos()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                simulation.handle_button_click(mouse_pos)

        screen.fill(COLORS["BLACK"])
        simulation.draw_ui()
        space.step(1 / FPS)
        simulation.update_particles()
        pygame.display.flip()
        clock.tick(FPS)

    quit_program()


if __name__ == "__main__":
    main()