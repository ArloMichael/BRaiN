# graphics.py
"""
graphics.py – camera, assets, and depth-sorted isometric rendering.

Key idea
--------
Anchor entities (player, collectables, shadows) to the cell’s *ground point*:
    ground_x = sx + tile_size/2
    ground_y = sy_lifted + TOP_FACE_FRAC * tile_size
For a standard 2:1 isometric cube, the top diamond height is tile_size/2,
so TOP_FACE_FRAC = 0.5. This avoids "sinking" without any magic numbers.
"""
from __future__ import annotations

import math
import os
from typing import Dict

import pygame

from .world import Player, TerrainBlock

# Fraction of the tile height where the top face reaches its bottom point
# For classic 2:1 isometric tiles, the top diamond height is tile_size/2.
TOP_FACE_FRAC = 0.2


# --------------------------------------------------------------------------- #
# CAMERA
# --------------------------------------------------------------------------- #
class Camera:
    """
    Float-coordinate camera.
    The (x, y) pair is the *world tile* that sits at the logical view origin.
    """

    def __init__(self) -> None:
        self.x: float = 0.0
        self.y: float = 0.0


# --------------------------------------------------------------------------- #
# ASSET HELPERS
# --------------------------------------------------------------------------- #
def load_images(
    asset_dir: str, terrain_types: list[str], tile_size: int
) -> Dict[str, pygame.Surface]:
    """Load & scale cube sprites once per tile_size."""
    images: Dict[str, pygame.Surface] = {}
    for terrain in terrain_types:
        path = os.path.join(asset_dir, f"cube_{terrain}.png")
        raw = pygame.image.load(path).convert_alpha()
        images[terrain] = pygame.transform.scale(raw, (tile_size, tile_size))
    return images


def load_player(asset_dir: str, tile_size: int) -> pygame.Surface:
    path = os.path.join(asset_dir, "player.png")
    raw = pygame.image.load(path).convert_alpha()
    return pygame.transform.scale(raw, (tile_size, tile_size))


def load_bot(asset_dir: str, tile_size: int) -> pygame.Surface:
    """Load the bot sprite (bot.png) scaled to tile_size."""
    path = os.path.join(asset_dir, "bot.png")
    raw = pygame.image.load(path).convert_alpha()
    return pygame.transform.scale(raw, (tile_size, tile_size))


def get_font(font_path: str, size: int) -> pygame.font.Font:
    try:
        return pygame.font.Font(font_path, size)
    except FileNotFoundError:
        return pygame.font.SysFont("tahoma", size)


# --------------------------------------------------------------------------- #
# DRAWING
# --------------------------------------------------------------------------- #
def draw_grid(
    surface: pygame.Surface,
    images: Dict[str, pygame.Surface],
    grid: Dict[str, TerrainBlock],
    camera: Camera,
    tile_size: int,
    font: pygame.font.Font,
    player: Player,
    player_img: pygame.Surface,
    fps_val: float,
    collectables=None,
    collectable_img=None,
    bots: list[Player] | None = None,
    bot_img: pygame.Surface | None = None,
    player_taken: int = 0,            # NEW: HUD breakdown
    bot_taken: int = 0,               # NEW: HUD breakdown
    timer_text=None,
) -> None:
    """
    Depth-sorted isometric rendering with principled ground anchoring.

    Sort key:
        (depth = wx + wy, layer, y)
    where layer is 0=tile, 1=shadow, 2=sprite.
    """
    win_w, win_h = surface.get_size()
    surface.fill((0, 0, 0))

    half_tile = tile_size / 2.0
    quarter_tile = tile_size / 4.0

    # Camera in iso-space
    iso_cam_x = camera.x - camera.y
    iso_cam_y = camera.x + camera.y

    # Place camera near center
    base_x = win_w / 2 - tile_size / 2
    base_y = win_h / 2 - tile_size / 4

    # Visible world bounds (with a small padding)
    world_left   = camera.x - (win_w / 2) / half_tile
    world_right  = camera.x + (win_w / 2) / half_tile
    world_top    = camera.y - (win_h / 2) / quarter_tile
    world_bottom = camera.y + (win_h / 2) / quarter_tile

    x0 = int(math.floor(world_left))  - 2
    x1 = int(math.ceil(world_right))  + 2
    y0 = int(math.floor(world_top))   - 2
    y1 = int(math.ceil(world_bottom)) + 2

    mouse_x, mouse_y = pygame.mouse.get_pos()
    hovered_key: str | None = None

    # Visible collectables for quick lookup
    if collectables:
        visible_collectables = {
            (c.x, c.y): c
            for c in collectables
            if not c.collected and world_left <= c.x <= world_right and world_top <= c.y <= world_bottom
        }
    else:
        visible_collectables = {}

    # Reusable shadow surface (ellipse)
    shadow_w = int(tile_size * 0.6)
    shadow_h = int(tile_size * 0.25)
    shadow_surf = pygame.Surface((shadow_w, shadow_h), pygame.SRCALPHA)
    pygame.draw.ellipse(shadow_surf, (0, 0, 0, 110), shadow_surf.get_rect())

    # Accumulate draw items: (depth, layer, sx, sy, surface)
    draw_list: list[tuple[int, int, int, int, pygame.Surface]] = []

    def world_to_screen(wx: int, wy: int) -> tuple[int, int]:
        sx = base_x + ((wx - wy) - iso_cam_x) * half_tile
        sy = base_y + ((wx + wy) - iso_cam_y) * quarter_tile
        return int(sx), int(sy)

    for wx in range(x0, x1 + 1):
        for wy in range(y0, y1 + 1):
            key = f"{wx}-{wy}"
            block = grid.get(key)
            if block is None:
                continue

            sx, sy = world_to_screen(wx, wy)

            # Cull off-screen quickly
            if sx + tile_size < 0 or sx > win_w or sy + tile_size < 0 or sy > win_h:
                continue

            # Mouse hover lift on the front/top faces
            lift = 0
            # if (...) set hovered_key and lift

            sy_lifted = sy - lift
            depth = wx + wy

            # --- Tile -------------------------------------------------------
            draw_list.append((depth, 0, sx, sy_lifted, images[block.terrain]))

            # Compute the ground point for this tile (bottom-center of top diamond)
            ground_x = sx + tile_size / 2
            ground_y = sy_lifted + TOP_FACE_FRAC * tile_size

            # --- Collectable on this tile -----------------------------------
            if collectable_img and (wx, wy) in visible_collectables:
                # Bobbing (unique phase per world coord)
                t_ms = pygame.time.get_ticks()
                amplitude = tile_size * 0.1
                period = 1500
                phase_offset = ((wx * 31 + wy * 17) % 360) * (2 * math.pi / 360)
                hover_offset = int(
                    amplitude * math.sin((2 * math.pi * t_ms) / period + phase_offset)
                )

                # Shadow at ground
                sh_x = int(ground_x - shadow_w / 2)
                sh_y = int(ground_y - shadow_h / 2)
                draw_list.append((depth, 1, sh_x, sh_y, shadow_surf))

                # Sprite: center on ground, then lift by ~tile_size/4 and apply bob
                c_w, c_h = collectable_img.get_size()
                draw_x = int(ground_x - c_w / 2)
                draw_y = int(ground_y - c_h - tile_size // 4 + hover_offset)
                draw_list.append((depth, 2, draw_x, draw_y, collectable_img))

            # --- Player on this tile ----------------------------------------
            if block.x == player.x and block.y == player.y:
                t_ms = pygame.time.get_ticks()
                amplitude = tile_size * 0.1
                period = 1500
                hover_offset = int(
                    amplitude * math.sin((2 * math.pi * t_ms) / period)
                )

                bottom_px = int(ground_x)
                bottom_py = int(ground_y)  # anchor: feet on the ground point

                # Shadow at ground
                sh_x = int(bottom_px - shadow_w / 2)
                sh_y = int(bottom_py - shadow_h / 2)
                draw_list.append((depth, 1, sh_x, sh_y, shadow_surf))

                # Sprite (scaled), feet stay on ground + hover bob
                scaled_size = int(tile_size * player.scale)
                sprite = (
                    player_img
                    if player.scale == 1.0
                    else pygame.transform.scale(player_img, (scaled_size, scaled_size))
                )
                draw_x = int(bottom_px - scaled_size / 2)
                draw_y = int(bottom_py - scaled_size + hover_offset)
                draw_list.append((depth, 2, draw_x, draw_y, sprite))

            # --- Bots on this tile (use bot.png) ----------------------------
            if bots:
                for bot in bots:
                    if block.x == bot.x and block.y == bot.y:
                        t_ms = pygame.time.get_ticks()
                        amplitude = tile_size * 0.1
                        period = 1500
                        # Small per-bot phase so the bot's bob isn't perfectly in sync
                        phase_offset = ((bot.x * 31 + bot.y * 17) % 360) * (2 * math.pi / 360)
                        hover_offset = int(
                            amplitude * math.sin((2 * math.pi * t_ms) / period + phase_offset)
                        )

                        bottom_px = int(ground_x)
                        bottom_py = int(ground_y)

                        # Shadow
                        sh_x = int(bottom_px - shadow_w / 2)
                        sh_y = int(bottom_py - shadow_h / 2)
                        draw_list.append((depth, 1, sh_x, sh_y, shadow_surf))

                        # Sprite (bot-specific image; respects bot.scale)
                        base_img = bot_img or player_img
                        scaled_size = int(tile_size * bot.scale)
                        sprite = (
                            base_img
                            if bot.scale == 1.0
                            else pygame.transform.scale(base_img, (scaled_size, scaled_size))
                        )
                        draw_x = int(bottom_px - scaled_size / 2)
                        draw_y = int(bottom_py - scaled_size + hover_offset)
                        draw_list.append((depth, 2, draw_x, draw_y, sprite))

    # Depth sort: depth -> layer -> y (stable tiebreaker)
    draw_list.sort(key=lambda it: (it[0], it[1], it[3]))
    for _, _, dx, dy, img in draw_list:
        surface.blit(img, (dx, dy))

    # ---------------- HUD ----------------
    cam_txt = font.render(
        f"Camera: ({int(camera.x)}, {int(camera.y)})", True, (255, 255, 255)
    )
    surface.blit(
        cam_txt, (win_w - cam_txt.get_width() - 8, cam_txt.get_height() + 8)
    )

    # Collectables counter (top-right, under Camera)
    if collectables is not None:
        taken = sum(1 for c in collectables if getattr(c, "collected", False))
        total = len(collectables)
        col_txt = font.render(f"Collectables: {taken}/{total}", True, (255, 255, 255))
        col_y = cam_txt.get_height() * 2 + 12
        surface.blit(col_txt, (win_w - col_txt.get_width() - 8, col_y))

        # NEW: breakdown line
        bd_txt = font.render(f"By P: {player_taken} | Bot: {bot_taken}", True, (255, 255, 255))
        surface.blit(
            bd_txt,
            (win_w - bd_txt.get_width() - 8, col_y + col_txt.get_height() + 4),
        )

    if hovered_key:
        hx, hy = map(int, hovered_key.split("-"))
        hov_txt = font.render(f"Block: ({hx}, {hy})", True, (255, 255, 255))
        surface.blit(hov_txt, (win_w - hov_txt.get_width() - 8, 8))

    pl_txt = font.render(
        f"Player: ({player.x}, {player.y}) {player.terrain}",
        True,
        (255, 255, 255),
    )
    surface.blit(pl_txt, (8, 8))

    fps_txt = font.render(f"FPS: {fps_val:.0f}", True, (255, 255, 0))
    surface.blit(fps_txt, (8, 8 + pl_txt.get_height()))

    if timer_text is not None:
        time_txt = font.render(f"Time: {timer_text}", True, (255, 255, 255))
        surface.blit(
            time_txt,
            (8, 8 + pl_txt.get_height() + fps_txt.get_height() + 4),
        )