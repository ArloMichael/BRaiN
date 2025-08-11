# win_scene.py
from __future__ import annotations

from typing import Optional
import pygame
import pygame.freetype as ft
from pygame._sdl2.video import Texture

from scene_manager import Scene
import audio

def _ensure_display_surface() -> None:
    if pygame.display.get_surface() is None:
        try:
            pygame.display.set_mode((1, 1), flags=pygame.HIDDEN)
        except Exception:
            pygame.display.set_mode((1, 1))

class WinScene(Scene):
    """
    Simple victory screen. Shows winner and scores.
    Keys:
      - Enter : Rematch (start a new GameScene with the provided agent)
      - Esc   : Back to main menu
    """
    def __init__(self, rend, winner: str, player_score: int, bot_score: int, total: int, agent, elapsed_ms=0):
        self._rend = rend
        self._agent = agent
        self.winner = winner
        self.player_score = player_score
        self.bot_score = bot_score
        self.total = total
        self.elapsed_ms = int(max(0, elapsed_ms))

        lw, lh = getattr(rend, "logical_size", (800, 300))
        self._logical_w, self._logical_h = int(lw), int(lh)

        _ensure_display_surface()

        # Off-screen framebuffer
        self._fb = pygame.Surface((self._logical_w, self._logical_h), pygame.SRCALPHA)
        self._fb_tex: Optional[Texture] = None

        # Fonts (match menu_scene style)
        self.title_font = ft.SysFont(None, 56)
        self.text_font  = ft.SysFont(None, 28)
        self.hint_font  = ft.SysFont(None, 22)

        self._played_sfx = False

    @staticmethod
    def _format_time(ms: int) -> str:
        total_cs = ms // 10
        minutes = total_cs // 6000
        seconds = (total_cs // 100) % 60
        centis = total_cs % 100
        return f"{minutes:02d}:{seconds:02d}.{centis:02d}"

    def on_enter(self, previous: Optional[Scene]) -> None:
        audio.stop_music(fade_ms=300)
        if not self._played_sfx:
            try:
                audio.play_sfx("fanfare")
            except Exception:
                pass
            self._played_sfx = True

    def on_exit(self, next_scene: Optional[Scene]) -> None:
        pass

    def handle_event(self, e):
        if e.type == pygame.KEYUP:
            if e.key == pygame.K_RETURN or e.key == pygame.K_KP_ENTER:
                from game_scene import GameScene
                return GameScene(self._rend, self._agent)
            if e.key == pygame.K_ESCAPE:
                from menu_scene import MenuScene
                return MenuScene(self._rend)

    def update(self, dt: float) -> None:
        pass

    def draw(self, r) -> None:
        self._fb.fill((0, 0, 0, 0))  # fully clear

        # Semi-transparent background overlay
        overlay = pygame.Surface((self._logical_w, self._logical_h), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 10))  # subtle darken
        self._fb.blit(overlay, (0, 0))

        # Text lines
        title = f"{self.winner} Wins!"
        line1 = f"Player: {self.player_score} / {self.total}"
        line2 = f"Bot: {self.bot_score} / {self.total}"
        line3 = f"Time: {self._format_time(self.elapsed_ms)}"
        hint1 = "Press Enter for a rematch"
        hint2 = "Press Esc to return to the menu"

        # Render with freetype (SysFont like menu_scene)
        t_surf, _  = self.title_font.render(title, fgcolor=(240, 240, 255))
        l1_surf, _ = self.text_font.render(line1,  fgcolor=(230, 230, 230))
        l2_surf, _ = self.text_font.render(line2,  fgcolor=(230, 230, 230))
        l3_surf, _ = self.text_font.render(line3, fgcolor=(230, 230, 230))
        h1_surf, _ = self.hint_font.render(hint1,  fgcolor=(210, 210, 210))
        h2_surf, _ = self.hint_font.render(hint2,  fgcolor=(210, 210, 210))

        spacing = 16
        total_h = (
            t_surf.get_height() + spacing +
            l1_surf.get_height() + spacing +
            l2_surf.get_height() + spacing +
            l3_surf.get_height() + 40 +
            h1_surf.get_height() + spacing +
            h2_surf.get_height()
        )
        y = (self._logical_h - total_h) // 2

        def blit_center(surf, ypos):
            self._fb.blit(surf, ((self._logical_w - surf.get_width()) // 2, ypos))
            return ypos + surf.get_height()

        y = blit_center(t_surf, y)
        y += spacing
        y = blit_center(l1_surf, y)
        y += spacing
        y = blit_center(l2_surf, y)
        y += spacing
        y = blit_center(l3_surf, y)
        y += 40
        y = blit_center(h1_surf, y)
        y += spacing
        y = blit_center(h2_surf, y)

        # Upload and draw
        self._fb_tex = Texture.from_surface(self._rend, self._fb)
        self._fb_tex.draw(dstrect=(0, 0, self._logical_w, self._logical_h))