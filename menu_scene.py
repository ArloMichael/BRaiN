# menu_scene.py  (UPDATED)
import pygame
import pygame.freetype
from pygame._sdl2.video import Texture

from scene_manager import Scene
import audio  # NEW
from settings_scene import SettingsScene

# ------------------------------------------------------------------ CONFIG
LOGICAL           = (1920, 1080)

BTN_SIZE          = (200, 70)
BTN_RADIUS        = 24
SUPERSAMPLE       = 4

BTN_COL_UP        = (230, 230, 230, 255)
BTN_COL_HOVER     = (245, 245, 245, 255)
BTN_COL_DOWN      = (210, 210, 210, 255)
OUTLINE_COL       = (200, 200, 200, 255)

LABEL_COLOR       = (40, 40, 40)

# --- Settings icon config
ICON_SIZE           = 44
ICON_PAD            = 20
ICON_PATH           = "assets/ui/gear.png"  # <- your PNG
ICON_HOVER_BRIGHT   = 230 / 255.0
ICON_DOWN_BRIGHT    = 200 / 255.0
ICON_TARGET_ANGLE   = 90.0
ICON_ROTATE_SPEED   = 600.0

def _load_png_icon_surface(path, size_px: int, brightness: float = 1.0) -> pygame.Surface:
    surf = pygame.image.load(str(path))
    if surf.get_size() != (size_px, size_px):
        surf = pygame.transform.smoothscale(surf, (size_px, size_px))
    if brightness != 1.0:
        factor = max(0, min(255, int(255 * brightness)))
        mod = pygame.Surface(surf.get_size(), pygame.SRCALPHA)
        mod.fill((factor, factor, factor, 255))
        surf = surf.copy()
        surf.blit(mod, (0, 0), special_flags=pygame.BLEND_RGBA_MULT)
    return surf

def _make_button_texture(rend, font, fill_color, label="Click me") -> Texture:
    big_size   = (BTN_SIZE[0] * SUPERSAMPLE, BTN_SIZE[1] * SUPERSAMPLE)
    big_radius = BTN_RADIUS * SUPERSAMPLE

    big = pygame.Surface(big_size, pygame.SRCALPHA)
    pygame.draw.rect(big, fill_color, big.get_rect(), border_radius=big_radius)
    pygame.draw.rect(big, OUTLINE_COL, big.get_rect(),
                     width=SUPERSAMPLE, border_radius=big_radius)

    small = pygame.transform.smoothscale(big, BTN_SIZE)

    lbl_surf, _ = font.render(label, fgcolor=LABEL_COLOR)
    small.blit(lbl_surf, lbl_surf.get_rect(center=small.get_rect().center))

    return Texture.from_surface(rend, small)

class MenuScene(Scene):
    def __init__(self, rend):
        self._rend = rend

        title_font = pygame.freetype.SysFont(None, 72)
        btn_font   = pygame.freetype.SysFont(None, 32)

        title_surf, _ = title_font.render("BRaiN", fgcolor="white")
        self._title_tex  = Texture.from_surface(rend, title_surf)
        self._title_rect = self._title_tex.get_rect(
            center=(LOGICAL[0] // 2, LOGICAL[1] // 2 - 60))

        self._btn_up    = _make_button_texture(rend, btn_font, BTN_COL_UP, "Start")
        self._btn_hover = _make_button_texture(rend, btn_font, BTN_COL_HOVER, "Start")
        self._btn_down  = _make_button_texture(rend, btn_font, BTN_COL_DOWN, "Start")
        self._btn_tex   = self._btn_up
        self._btn_rect  = self._btn_tex.get_rect(
            center=(LOGICAL[0] // 2, LOGICAL[1] - 100))

        self._mouse_down_inside = False

        # Settings icon
        self._settings_up_surf    = _load_png_icon_surface(ICON_PATH, ICON_SIZE, 1.0)
        self._settings_hover_surf = _load_png_icon_surface(ICON_PATH, ICON_SIZE, 230/255.0)
        self._settings_down_surf  = _load_png_icon_surface(ICON_PATH, ICON_SIZE, 200/255.0)

        self._settings_up    = Texture.from_surface(rend, self._settings_up_surf)
        self._settings_hover = Texture.from_surface(rend, self._settings_hover_surf)
        self._settings_down  = Texture.from_surface(rend, self._settings_down_surf)

        self._settings_tex   = self._settings_up
        self._settings_rect  = self._settings_tex.get_rect(topleft=(ICON_PAD, ICON_PAD))
        self._settings_mouse_down_inside = False

        # Rotation state
        self._settings_angle      = 0.0
        self._settings_target     = 0.0
        self._settings_hovering   = False
        self._settings_spin_speed = ICON_ROTATE_SPEED

    def on_enter(self, previous):
        audio.play_music(music_type=audio.MusicType.MENU)

    def on_exit(self, next_scene):
        audio.stop_music(fade_ms=500)

    def handle_event(self, e):
        if e.type == pygame.WINDOWCLOSE:
            pygame.event.post(pygame.event.Event(pygame.QUIT))
            return

        if e.type == pygame.MOUSEMOTION:
            if self._btn_rect.collidepoint(e.pos):
                self._btn_tex = (self._btn_down
                                 if self._mouse_down_inside else self._btn_hover)
            else:
                self._btn_tex = self._btn_up

            if self._settings_rect.collidepoint(e.pos):
                self._settings_hovering = True
                self._settings_target = ICON_TARGET_ANGLE
                self._settings_tex = (self._settings_down
                                      if self._settings_mouse_down_inside else self._settings_hover)
            else:
                self._settings_hovering = False
                self._settings_target = 0.0
                self._settings_tex = self._settings_up

        elif e.type == pygame.MOUSEBUTTONDOWN and e.button == 1:
            if self._btn_rect.collidepoint(e.pos):
                self._mouse_down_inside = True
                self._btn_tex = self._btn_down

            if self._settings_rect.collidepoint(e.pos):
                self._settings_mouse_down_inside = True
                self._settings_tex = self._settings_down

        elif e.type == pygame.MOUSEBUTTONUP and e.button == 1:
            # Start button release
            if self._mouse_down_inside and self._btn_rect.collidepoint(e.pos):
                # Switch to the loading scene (NEW)
                from loading_scene import LoadingScene
                return LoadingScene(self._rend)

            self._mouse_down_inside = False
            self._btn_tex = (self._btn_hover if self._btn_rect.collidepoint(e.pos)
                             else self._btn_up)

            # Settings icon release
            if self._settings_mouse_down_inside and self._settings_rect.collidepoint(e.pos):
                return SettingsScene(self._rend, previous=self)

            self._settings_mouse_down_inside = False
            self._settings_tex = (self._settings_hover if self._settings_rect.collidepoint(e.pos)
                                  else self._settings_up)

    def update(self, dt: float) -> None:
        if self._settings_angle != self._settings_target:
            step = self._settings_spin_speed * dt
            if self._settings_angle < self._settings_target:
                self._settings_angle = min(self._settings_angle + step, self._settings_target)
            else:
                self._settings_angle = max(self._settings_angle - step, self._settings_target)

    def draw(self, r):
        # Settings icon
        if self._settings_mouse_down_inside and self._settings_hovering:
            base_surf = self._settings_down_surf
            base_tex  = self._settings_down
        elif self._settings_hovering:
            base_surf = self._settings_hover_surf
            base_tex  = self._settings_hover
        else:
            base_surf = self._settings_up_surf
            base_tex  = self._settings_up

        if self._settings_angle != 0.0:
            rotated = pygame.transform.rotate(base_surf, self._settings_angle)
            rot_tex = Texture.from_surface(self._rend, rotated)
            rect = rot_tex.get_rect(center=self._settings_rect.center)
            rot_tex.draw(dstrect=rect)
        else:
            base_tex.draw(dstrect=self._settings_rect)

        # Rest of UI
        self._title_tex.draw(dstrect=self._title_rect)
        self._btn_tex.draw(dstrect=self._btn_rect)