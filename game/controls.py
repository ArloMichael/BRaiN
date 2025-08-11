"""
controls.py – event handling; mutates Player and returns updated tile‑size.
The camera is *not* moved here anymore; main.py now centres it once per frame.
"""
from __future__ import annotations

import pygame
from pygame.locals import *

from .graphics import Camera
from .world import Player


def process_events(
    camera: Camera,
    player: Player,
    tile_size: int,
    zoom_step: int,
    grid_size: int,
) -> tuple[bool, int]:
    """
    Handle Pygame events.

    Returns
    -------
    running : bool
        False when the user closes the window or presses Esc.
    new_tile_size : int
        Updated tile size (may be unchanged).
    """
    running = True
    new_tile_size = tile_size

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        elif event.type == KEYDOWN:
            if event.key == K_ESCAPE:
                running = False

            # ---- zoom ---- #
            elif event.key == K_MINUS:
                new_tile_size = max(16, new_tile_size - zoom_step)
            elif event.key == K_EQUALS:
                new_tile_size += zoom_step

            # ---- player movement ---- #
            elif event.key in (K_UP, K_w):
                player.y = max(0, player.y - 1)
            elif event.key in (K_DOWN, K_s):
                player.y = min(grid_size - 1, player.y + 1)
            elif event.key in (K_LEFT, K_a):
                player.x = max(0, player.x - 1)
            elif event.key in (K_RIGHT, K_d):
                player.x = min(grid_size - 1, player.x + 1)

        elif event.type == MOUSEBUTTONDOWN:
            # mouse wheel zoom
            if event.button == 4:            # wheel up
                new_tile_size += zoom_step
            elif event.button == 5:          # wheel down
                new_tile_size = max(16, new_tile_size - zoom_step)

    # NOTE: camera centring is now performed in main.py
    return running, new_tile_size