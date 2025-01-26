import pygame
import pymunk
import tkinter as tk
from simulation import Simulation
from utils import quit_program


# Pygame and simulation constants
WIDTH, HEIGHT = 800, 1000
UI_BAR_HEIGHT = 60
FPS = 60
DT = 1/FPS

# Button dimensions
BUTTON_WIDTH = 120
BUTTON_HEIGHT = 40
BUTTON_GAP = 10

COLORS = {
    "BLACK": (0, 0, 0),
    "WHITE": (255, 255, 255),
    "DARK_GRAY": (70, 70, 70),
    "UI_BG": (30, 30, 30),
    "RED": (255, 0, 0),
}

# Initialize Tkinter for file dialogs
root = tk.Tk()
root.withdraw()

# Pygame initialization
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Fuit.es Compost")
font = pygame.font.SysFont(None, 24)


def main() -> None:
    """
    Main loop for the simulation.
    """
    # Prepare simulation parameters
    sim_params = {
        "width": WIDTH,
        "height": HEIGHT,
        "ui_bar_height": UI_BAR_HEIGHT,
        "colors": COLORS,
        "button_dimensions": {
            "width": BUTTON_WIDTH,
            "height": BUTTON_HEIGHT,
            "gap": BUTTON_GAP,
        },
    }

    clock = pygame.time.Clock()
    running = True

    # Pymunk space initialization
    space = pymunk.Space()
    space.gravity = (0, 0)

    simulation = Simulation(sim_params, space, screen, font)

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

        screen.fill(COLORS["WHITE"])
        simulation.draw_ui()
        space.step(DT)
        simulation.update_chunks()
        pygame.display.flip()
        clock.tick(FPS)

    quit_program()

if __name__ == "__main__":
    main()