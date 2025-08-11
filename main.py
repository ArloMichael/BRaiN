"""
main.py
Entry point.  Sets up Pygame, the window/renderer, and
spins the main loop using SceneManager.
"""

import os
os.environ["SDL_WINDOWS_DPI_AWARENESS"] = "permonitorv2"   # Hi‑DPI on Windows

import pygame
from pygame._sdl2.video import Window, Renderer
import pygame.freetype

from scene_manager import Scene, SceneManager
from menu_scene import MenuScene

# ------------------------------------------------------------------ CONSTANTS
LOGICAL    = (1920, 1080)
BG_COLOR   = (30, 30, 30, 255)
FPS        = 60

# ------------------------------------------------------------------ INITIALISATION
pygame.init()
pygame.freetype.init()

win  = Window("BRaiN", size=LOGICAL, allow_high_dpi=True)
rend = Renderer(win, vsync=True)
rend.logical_size = LOGICAL

clock = pygame.time.Clock()

# ------------------------------------------------------------------ MAIN LOOP
def main() -> None:
    manager = SceneManager(MenuScene(rend))
    running = True

    while running and manager.current:
        dt = clock.tick(FPS) / 1000.0

        # ----------------------------- event pump
        # main.py  (inside the event loop)
        for e in pygame.event.get():
            # pygame-ce posts WINDOWCLOSE when the user clicks the X button.
            if e.type in (pygame.QUIT, pygame.WINDOWCLOSE):
                running = False
                break

            cmd = manager.current.handle_event(e)
            if isinstance(cmd, Scene):
                manager.change(cmd)

        # ----------------------------- draw
        rend.draw_color = BG_COLOR
        rend.clear()

        manager.current.update(dt)
        manager.current.draw(rend)

        rend.present()

    pygame.quit()


if __name__ == "__main__":
    main()