# settings_scene.py
"""
SettingsScene: full-page settings screen with smooth fade in/out and On/Off toggles.
- Replaces the menu visually (no overlay/panel/cutout)
- PNG "Back" icon at the same spot as the menu's settings icon (top-left)
- Four toggles (Audio, Music, SFX, Ambiance) — active side is blue, other side is dark grey
- Fade in on enter; fade out on Back icon / Done / ESC; returns to previous scene after fade-out
"""

import pygame
import pygame.freetype
from pygame._sdl2.video import Texture
from typing import Optional, Dict, Tuple

from scene_manager import Scene
import audio

# Match your logical resolution
LOGICAL = (1920, 1080)

# ------------------------------ UI CONFIG
# Layout spacings for a clean full-page composition
TOP_MARGIN      = 120    # distance from top for the title
TITLE_SUB_GAP   = 12     # gap between title and subtitle
SECTION_GAP     = 40     # gap between subtitle and first toggle group
ROW_GAP         = 28     # vertical spacing between toggle rows
LABEL_GAP       = 20     # gap between label and toggle in a row
BOTTOM_MARGIN   = 80     # distance from bottom for the Done button

# Back icon (exact same placement/size as the menu's settings icon)
ICON_SIZE         = 44
ICON_PAD          = 20
BACK_ICON_PATH    = "assets/ui/back.png"   # <- place your back icon PNG here
BACK_HOVER_BRIGHT = 230 / 255.0                 # slightly darker on hover
BACK_DOWN_BRIGHT  = 200 / 255.0                 # darker on press

# Button
BTN_SIZE       = (220, 70)
BTN_RADIUS     = 18
BTN_UP         = (230, 230, 230, 255)
BTN_HOVER      = (245, 245, 245, 255)
BTN_DOWN       = (210, 210, 210, 255)
OUTLINE_RGBA   = (200, 200, 200, 255)  # For button outline

# Toggle visuals
TOGGLE_SIZE    = (220, 56)             # width, height
TOGGLE_RADIUS  = 28
TOGGLE_BLUE    = (50, 120, 255, 255)   # active side
TOGGLE_DARK    = (60, 60, 60, 255)     # inactive side
TOGGLE_TEXT_ON = "ON"
TOGGLE_TEXT_OFF= "OFF"
TOGGLE_LABEL_COL = (255, 255, 255)     # ON/OFF text color (white)

# Text colours
TITLE_COL        = (255, 255, 255)     # Settings title in white
SUB_LABEL_COL    = (70, 70, 70)        # Subtitle
BUTTON_LABEL_COL = (40, 40, 40)        # Button text
ROW_LABEL_COL    = (255, 255, 255)     # Row labels (Audio, Music, ...)

TITLE_TEXT     = "Settings"
SUBTEXT        = "Press Esc, click the back icon, or click Done."

# Fade timings (seconds)
FADE_IN_TIME   = 0.20
FADE_OUT_TIME  = 0.15

# Custom event used to signal that fade-out finished (safe close)
SETTINGS_CLOSED_EVENT = pygame.USEREVENT + 42


# ------------------------------ helpers
def _label_texture(rend, text, font, fg):
    surf, _ = font.render(text, fgcolor=fg)
    return Texture.from_surface(rend, surf), surf.get_rect()


def _button_texture(rend, font, text, fill_rgba):
    # Supersample for smoother corners/text
    ss = 3
    big_size   = (BTN_SIZE[0] * ss, BTN_SIZE[1] * ss)
    big_radius = BTN_RADIUS * ss

    big = pygame.Surface(big_size, pygame.SRCALPHA)
    pygame.draw.rect(big, fill_rgba, big.get_rect(), border_radius=big_radius)
    # Subtle outline on button
    pygame.draw.rect(big, OUTLINE_RGBA, big.get_rect(), width=ss, border_radius=big_radius)

    small = pygame.transform.smoothscale(big, BTN_SIZE)

    label_surf, _ = font.render(text, fgcolor=BUTTON_LABEL_COL)
    small.blit(label_surf, label_surf.get_rect(center=small.get_rect().center))

    return Texture.from_surface(rend, small)


def _load_png_icon_surface(path, size_px: int, brightness: float = 1.0) -> pygame.Surface:
    """Load a PNG as a SURFACE, scale to size, and apply brightness via BLEND_RGBA_MULT.
    Works with SDL2 Renderer (no convert_alpha() needed)."""
    surf = pygame.image.load(str(path))  # no convert_alpha()
    if surf.get_size() != (size_px, size_px):
        surf = pygame.transform.smoothscale(surf, (size_px, size_px))
    if brightness != 1.0:
        factor = max(0, min(255, int(255 * brightness)))
        mod = pygame.Surface(surf.get_size(), pygame.SRCALPHA)
        mod.fill((factor, factor, factor, 255))
        surf = surf.copy()
        surf.blit(mod, (0, 0), special_flags=pygame.BLEND_RGBA_MULT)
    return surf


def _make_toggle_pair_textures(
    rend,
    font: pygame.freetype.Font,
    size: Tuple[int, int] = TOGGLE_SIZE,
    radius: int = TOGGLE_RADIUS,
) -> Tuple[Texture, Texture]:
    """
    Build two textures for the toggle:
      - OFF-active (left half blue, right dark)  -> selecting OFF (False)
      - ON-active  (left half dark, right blue)  -> selecting ON  (True)
    Each shows 'OFF' on the left, 'ON' on the right; the active side is blue.
    """
    w, h = size
    ss = 3
    big = pygame.Surface((w * ss, h * ss), pygame.SRCALPHA)
    # Base pill background (we'll draw halves over it)
    pygame.draw.rect(big, (0, 0, 0, 0), big.get_rect(), border_radius=radius * ss)  # transparent base

    # Helper to render a state
    def render_state(left_col, right_col):
        s = big.copy()
        rect = s.get_rect()
        # Draw two rounded halves by overlaying rects; small overlap to avoid hairline seam
        mid_x = rect.w // 2
        overlap = 2 * ss
        left_rect  = pygame.Rect(rect.left, rect.top, mid_x + overlap, rect.h)
        right_rect = pygame.Rect(mid_x - overlap, rect.top, rect.w - (mid_x - overlap), rect.h)
        pygame.draw.rect(s, left_col, left_rect,  border_radius=radius * ss)
        pygame.draw.rect(s, right_col, right_rect, border_radius=radius * ss)

        small = pygame.transform.smoothscale(s, (w, h))

        # Add OFF/ON labels
        off_surf, _ = font.render(TOGGLE_TEXT_OFF, fgcolor=TOGGLE_LABEL_COL)
        on_surf,  _ = font.render(TOGGLE_TEXT_ON,  fgcolor=TOGGLE_LABEL_COL)

        small.blit(off_surf, off_surf.get_rect(center=(w * 0.25, h * 0.5)))
        small.blit(on_surf,  on_surf.get_rect(center=(w * 0.75, h * 0.5)))
        return Texture.from_surface(rend, small)

    # OFF active (left blue, right dark)
    tex_off = render_state(TOGGLE_BLUE, TOGGLE_DARK)
    # ON active  (left dark, right blue)
    tex_on  = render_state(TOGGLE_DARK, TOGGLE_BLUE)
    return tex_off, tex_on


class SettingsScene(Scene):
    """
    Full-page settings that returns to the previous scene after a fade-out.
    """
    def __init__(self, rend, previous: Scene):
        self._rend = rend
        self._previous = previous  # we return this instance when closing

        # Fade state
        self._fade = 0.0               # 0.0 .. 1.0
        self._closing = False
        self._posted_close = False

        # Fonts
        self._title_font = pygame.freetype.SysFont(None, 64)
        self._body_font  = pygame.freetype.SysFont(None, 24)
        self._btn_font   = pygame.freetype.SysFont(None, 32)
        self._row_font   = pygame.freetype.SysFont(None, 32)  # for "Audio", "Music", etc.
        self._toggle_font= pygame.freetype.SysFont(None, 26)  # for ON/OFF inside toggles

        # Title and subtitle
        self._title_tex, self._title_rect = _label_texture(rend, TITLE_TEXT, self._title_font, TITLE_COL)
        self._sub_tex,   self._sub_rect   = _label_texture(rend, SUBTEXT, self._body_font, SUB_LABEL_COL)

        # Done button states
        self._done_up    = _button_texture(rend, self._btn_font, "Done", BTN_UP)
        self._done_hover = _button_texture(rend, self._btn_font, "Done", BTN_HOVER)
        self._done_down  = _button_texture(rend, self._btn_font, "Done", BTN_DOWN)
        self._done_tex   = self._done_up
        self._done_rect  = self._done_tex.get_rect()
        self._done_pressed_inside = False

        # Back icon (PNG) at same spot as menu's settings icon
        self._back_up_surf    = _load_png_icon_surface(BACK_ICON_PATH, ICON_SIZE, 1.0)
        self._back_hover_surf = _load_png_icon_surface(BACK_ICON_PATH, ICON_SIZE, BACK_HOVER_BRIGHT)
        self._back_down_surf  = _load_png_icon_surface(BACK_ICON_PATH, ICON_SIZE, BACK_DOWN_BRIGHT)

        self._back_up_tex     = Texture.from_surface(rend, self._back_up_surf)
        self._back_hover_tex  = Texture.from_surface(rend, self._back_hover_surf)
        self._back_down_tex   = Texture.from_surface(rend, self._back_down_surf)

        self._back_icon_tex   = self._back_up_tex
        self._back_icon_rect  = self._back_icon_tex.get_rect(topleft=(ICON_PAD, ICON_PAD))
        self._back_pressed_inside = False
        self._back_hovering = False

        # Toggles state (defaults; you can load from disk or audio module)
        self._toggles: Dict[str, bool] = {
            "audio": True,
            "music": True,
            "sfx": True,
            "ambiance": True,
        }

        # Toggle textures (shared for all toggles)
        self._toggle_off_tex, self._toggle_on_tex = _make_toggle_pair_textures(
            rend, self._toggle_font, TOGGLE_SIZE, TOGGLE_RADIUS
        )

        # Labels for rows
        self._row_labels: Dict[str, Tuple[Texture, pygame.Rect]] = {}
        for name in ("Audio", "Music", "SFX", "Ambiance"):
            tex, rect = _label_texture(rend, name, self._row_font, ROW_LABEL_COL)
            self._row_labels[name.lower()] = (tex, rect)

        # Toggle rects (positions filled in by layout)
        self._toggle_rects: Dict[str, pygame.Rect] = {
            "audio": self._toggle_on_tex.get_rect(),
            "music": self._toggle_on_tex.get_rect(),
            "sfx": self._toggle_on_tex.get_rect(),
            "ambiance": self._toggle_on_tex.get_rect(),
        }

        # Lay out rects
        self._layout()

    # ------------------------------ internal
    def _layout(self):
        # Center title/subtitle horizontally; position vertically using margins
        self._title_rect.midtop = (LOGICAL[0] // 2, TOP_MARGIN)
        self._sub_rect.midtop   = (LOGICAL[0] // 2, self._title_rect.bottom + TITLE_SUB_GAP)

        # Rows start below subtitle
        start_y = self._sub_rect.bottom + SECTION_GAP
        center_x = LOGICAL[0] // 2

        # Place each row: label right-aligned at (center_x - LABEL_GAP),
        # toggle left-aligned at (center_x + LABEL_GAP)
        rows = ["audio", "music", "sfx", "ambiance"]
        y = start_y
        for key in rows:
            label_tex, label_rect = self._row_labels[key]
            toggle_rect = self._toggle_rects[key]

            # Position label: right aligned to (center_x - LABEL_GAP)
            label_rect.midright = (center_x - LABEL_GAP, y + toggle_rect.h // 2)

            # Position toggle: left aligned to (center_x + LABEL_GAP)
            toggle_rect.topleft = (center_x + LABEL_GAP, y)

            # Commit positions back
            self._row_labels[key] = (label_tex, label_rect)
            self._toggle_rects[key] = toggle_rect

            # Next row
            y += toggle_rect.h + ROW_GAP

        # Done button centered at bottom with margin
        self._done_rect.midbottom = (LOGICAL[0] // 2, LOGICAL[1] - BOTTOM_MARGIN)

    # ------------------------------ Scene hooks
    def on_enter(self, previous: Optional[Scene]) -> None:
        # Start fade-in
        self._fade = 0.0
        self._closing = False
        self._posted_close = False

        if audio is not None:
            try:
                audio.play_music(music_type=audio.MusicType.MENU)
            except Exception:
                pass

    def on_exit(self, next_scene: Optional[Scene]) -> None:
        # Let next scene control music
        pass

    # ------------------------------ Event handling
    def handle_event(self, e) -> Optional[Scene]:
        # During fade-out, block input; close when our event fires
        if self._closing:
            if e.type == SETTINGS_CLOSED_EVENT:
                return self._previous
            return None

        # Allow window close behavior consistent with your app
        if e.type == pygame.WINDOWCLOSE:
            pygame.event.post(pygame.event.Event(pygame.QUIT))
            return None

        # Keyboard: Esc/Backspace starts fade-out
        if e.type == pygame.KEYDOWN and e.key in (pygame.K_ESCAPE, pygame.K_BACKSPACE):
            self._begin_close()
            return None

        # Mouse interactions
        if e.type == pygame.MOUSEMOTION:
            # Done button hover
            if self._done_rect.collidepoint(e.pos):
                self._done_tex = self._done_down if self._done_pressed_inside else self._done_hover
            else:
                self._done_tex = self._done_up

            # Back icon hover
            if self._back_icon_rect.collidepoint(e.pos):
                self._back_hovering = True
                self._back_icon_tex = self._back_down_tex if self._back_pressed_inside else self._back_hover_tex
            else:
                self._back_hovering = False
                self._back_icon_tex = self._back_up_tex

        elif e.type == pygame.MOUSEBUTTONDOWN and e.button == 1:
            # Done button press
            if self._done_rect.collidepoint(e.pos):
                self._done_pressed_inside = True
                self._done_tex = self._done_down

            # Back icon press
            if self._back_icon_rect.collidepoint(e.pos):
                self._back_pressed_inside = True
                self._back_icon_tex = self._back_down_tex

        elif e.type == pygame.MOUSEBUTTONUP and e.button == 1:
            # Done button release
            if self._done_pressed_inside and self._done_rect.collidepoint(e.pos):
                self._begin_close()
            self._done_pressed_inside = False
            self._done_tex = self._done_hover if self._done_rect.collidepoint(e.pos) else self._done_up

            # Back icon release
            if self._back_pressed_inside and self._back_icon_rect.collidepoint(e.pos):
                self._begin_close()
            self._back_pressed_inside = False
            self._back_icon_tex = self._back_hover_tex if self._back_icon_rect.collidepoint(e.pos) else self._back_up_tex

            # Toggle clicks: flip based on which half was clicked
            for key, rect in self._toggle_rects.items():
                if rect.collidepoint(e.pos):
                    # Left half -> OFF (False); Right half -> ON (True)
                    on = e.pos[0] >= rect.centerx
                    self._set_toggle(key, on)
                    break

        # Swallow all other inputs
        return None

    # ------------------------------ Toggle application hooks
    def _set_toggle(self, key: str, value: bool) -> None:
        if key not in self._toggles:
            return
        self._toggles[key] = value

        if key == "audio":
            audio.set_master_enabled(value)
        elif key == "music":
            audio.set_music_enabled(value)
        elif key == "sfx":
            audio.set_sfx_enabled(value)
        elif key in ("ambience", "ambiance"):
            audio.set_ambience_enabled(value)

    def _begin_close(self):
        """Initiate fade-out; actual scene return happens after fade completes."""
        self._closing = True

    # ------------------------------ Update / Draw
    def update(self, dt: float) -> None:
        # Fade towards target (1 for open, 0 for close)
        if not self._closing and self._fade < 1.0:
            self._fade = min(1.0, self._fade + (dt / FADE_IN_TIME if FADE_IN_TIME > 0 else 1.0))
        elif self._closing and self._fade > 0.0:
            self._fade = max(0.0, self._fade - (dt / FADE_OUT_TIME if FADE_OUT_TIME > 0 else 1.0))

        # Once fade-out is complete, post a close event (so handle_event can return previous)
        if self._closing and self._fade <= 0.0 and not self._posted_close:
            pygame.event.post(pygame.event.Event(SETTINGS_CLOSED_EVENT))
            self._posted_close = True

    def draw(self, r) -> None:
        """
        Full-page draw: assumes your main loop clears the renderer each frame
        to the same background colour as the menu (renderer.draw_color).
        We then fade the UI elements (back icon, title, subtitle, toggles, button).
        """
        alpha = int(self._fade * 255)

        # Back icon (top-left, same spot as menu's settings icon)
        self._back_icon_tex.alpha = alpha
        self._back_icon_tex.draw(dstrect=self._back_icon_rect)

        # Title & subtitle
        self._title_tex.alpha = alpha
        self._title_tex.draw(dstrect=self._title_rect)

        self._sub_tex.alpha = alpha
        self._sub_tex.draw(dstrect=self._sub_rect)

        # Toggle rows
        for key in ("audio", "music", "sfx", "ambiance"):
            label_tex, label_rect = self._row_labels[key]
            label_tex.alpha = alpha
            label_tex.draw(dstrect=label_rect)

            rect = self._toggle_rects[key]
            tex = self._toggle_on_tex if self._toggles[key] else self._toggle_off_tex
            tex.alpha = alpha
            tex.draw(dstrect=rect)

        # Done button
        self._done_tex.alpha = alpha
        self._done_tex.draw(dstrect=self._done_rect)