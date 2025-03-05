import pygame
import sys
from typing import List, Tuple


def scale_image_to_fit(image: pygame.Surface, max_width: int, max_height: int) -> pygame.Surface:
    """
    Scales the image to fit within the specified maximum width and height while maintaining aspect ratio.
    
    Args:
        image (pygame.Surface): The image to scale.
        max_width (int): Maximum allowed width.
        max_height (int): Maximum allowed height.
    
    Returns:
        pygame.Surface: The scaled image.
    """
    original_width, original_height = image.get_size()
    scale_ratio = min(max_width / original_width, max_height / original_height, 1)
    new_size = (int(original_width * scale_ratio), int(original_height * scale_ratio))
    return pygame.transform.smoothscale(image, new_size)


def quit_program() -> None:
    """
    Quits the pygame program gracefully.
    """
    pygame.quit()
    sys.exit()
