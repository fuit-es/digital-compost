"""Entry point for the Compost simulation.

Physics, worms and audio run here; the picture is drawn in the browser
(web/, served at http://<host>:<port>/). pygame runs headless.
"""
import os
import webbrowser

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")  # No pygame window, no Dock icon

import pygame
import pymunk
import yaml

from simulation import Simulation
from network import UploadServer, ViewerHub
from utils.system_utils import quit_program
from utils.logging_utils import get_logger

_logger = get_logger(__name__)

# World size used with "auto" until the first browser reports its viewport
_AUTO_DEFAULT_SIZE = (1280, 800)


def load_config(config_file: str = "config.yaml") -> dict:
    """
    Load configuration settings from a YAML file.

    Args:
        config_file: Path to the configuration file.

    Returns:
        Configuration parameters as a dictionary.
    """
    with open(config_file, 'r') as file:
        return yaml.safe_load(file)


def main() -> None:
    """Main loop for the simulation."""
    # Load the configuration
    config = load_config()

    # Mixer is handled by SDAudioManager (sounddevice), not pygame.
    # Quit SDL's mixer so it doesn't hold the audio device.
    pygame.init()
    pygame.mixer.quit()

    # Resolve world size: "auto" follows the browser viewport, fixed sizes are letterboxed
    win_cfg = config["simulation"]["window"]
    follow_viewer = win_cfg.get("width") == "auto" or win_cfg.get("height") == "auto"
    if win_cfg.get("width") == "auto":
        win_cfg["width"] = _AUTO_DEFAULT_SIZE[0]
    if win_cfg.get("height") == "auto":
        win_cfg["height"] = _AUTO_DEFAULT_SIZE[1]

    # Initialize Pymunk space
    space = pymunk.Space()
    space.gravity = (
        config["simulation"]["physics"]["gravity"]["x"],
        config["simulation"]["physics"]["gravity"]["y"]
    )

    # Pass the entire config to the Simulation class
    simulation = Simulation(config, space)
    viewer_hub = ViewerHub(config)

    # Start background HTTP server for uploads
    server_cfg = config.get("server", {})
    host = server_cfg.get("host", "127.0.0.1")
    port = int(server_cfg.get("port", 5055))

    upload_server = UploadServer(
        host=host,
        port=port,
        config=config,
        on_upload_callback=simulation.on_upload_received,
        on_audio_callback=simulation.submit_audio,
        viewer_hub=viewer_hub,
    )
    upload_server.start()

    viewer_url = f"http://{host}:{port}/"
    _logger.info("Viewer: %s", viewer_url)
    if server_cfg.get("open_browser", True):
        webbrowser.open(viewer_url)

    clock = pygame.time.Clock()
    running = True
    dt = 1.0 / config["simulation"]["fps"]
    fps = 0.0

    while running:
        # Check if simulation requested quit
        if simulation.should_quit:
            running = False

        # Keyboard shortcuts and pill bar buttons from the browser
        for key in viewer_hub.drain_commands():
            simulation.handle_key(key)

        if follow_viewer and viewer_hub.requested_size:
            simulation.resize(*viewer_hub.requested_size)

        # Process any pending uploads (main-thread safe)
        simulation.process_pending_uploads()

        # Receive processed chunks from background (non-blocking)
        simulation.receive_chunks()

        # Handle audio hover for sound chunks
        simulation.handle_audio_hover(viewer_hub.mouse_pos)

        space.step(dt)
        simulation.update_chunks()
        viewer_hub.publish(simulation, fps)
        clock.tick(config["simulation"]["fps"])
        fps = clock.get_fps()
        upload_server.stats["fps"] = round(fps, 1)
        upload_server.stats["chunks"] = len(simulation.chunks)
        glues = simulation.glues
        upload_server.stats["glues"] = len(glues)
        upload_server.stats["avg_flocked"] = round(
            sum(len(g.glued_chunks) for g in glues) / len(glues), 1
        ) if glues else 0

    # Clean up worker process before exiting
    simulation.cleanup()
    quit_program()


if __name__ == "__main__":
    main()
