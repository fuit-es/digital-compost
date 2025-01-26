import pygame
import sys


def scale_image_to_fit(img: pygame.Surface, max_width: int, max_height: int) -> pygame.Surface:
    """
    Scale the image to fit within the given max_width and max_height.
    """
    iw, ih = img.get_size()
    scale = min(max_width / iw, max_height / ih)
    if scale < 1:
        new_w = int(iw * scale)
        new_h = int(ih * scale)
        return pygame.transform.scale(img, (new_w, new_h))
    else:
        return img
    
def quit_program() -> None:
    """
    Quit the program.
    """
    pygame.quit()
    sys.exit()
