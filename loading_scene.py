# loading_scene.py  (UPDATED)
import pygame
import pygame.freetype
from pygame._sdl2.video import Texture

from scene_manager import Scene

AI_MODEL_PATH = "v10.pt"

class LoadingScene(Scene):
    """
    Shows step-by-step loading messages while importing & loading the AI.
    After success, posts a custom event that handle_event uses to switch scenes.
    """
    def __init__(self, rend):
        self._rend = rend
        lw, lh = getattr(rend, "logical_size", (1280, 720))
        self._logical_w, self._logical_h = int(lw), int(lh)

        # Font & first line (same look as before; just different text)
        self._font = pygame.freetype.SysFont(None, 48)
        self._txt = None
        self._txt_rect = None
        self._set_text("Loading the AI...")  # first log line

        # State machine so each line is visible before heavy work
        self._phase = "pre_import"  # -> do_import -> pre_load_model -> do_load_model -> done
        self._has_drawn_once = False
        self._failed: str | None = None

        # Custom events to signal completion/error
        self._loaded_evt = pygame.event.custom_type()
        self._error_evt  = pygame.event.custom_type()

    # ------------------------------- helpers
    def _set_text(self, msg: str) -> None:
        # Render one line of text, same font/size as before
        surf, _ = self._font.render(msg, fgcolor="white")
        self._txt = Texture.from_surface(self._rend, surf)
        self._txt_rect = self._txt.get_rect(center=(self._logical_w // 2, self._logical_h // 2))

    # ------------------------------- Scene API
    def handle_event(self, e):
        if e.type == pygame.WINDOWCLOSE:
            pygame.event.post(pygame.event.Event(pygame.QUIT))
            return

        if e.type == self._loaded_evt:
            from game_scene import GameScene
            return GameScene(self._rend, agent=e.agent)

        if e.type == self._error_evt:
            self._failed = e.message

    def update(self, dt: float):
        # Ensure the loading text has drawn at least once before doing work
        if not self._has_drawn_once or self._failed:
            return

        # Advance the small state machine one step per frame so messages appear
        if self._phase == "pre_import":
            # Show: "Loading AI lib..."
            print("Loading PyTorch & Custom library...")
            self._set_text("Loading PyTorch & Custom AI library...")
            self._phase = "do_import"
            return

        if self._phase == "do_import":
            try:
                from ai import Agent
                print("PyTorch and Custom Library loaded successfully!")
                self._set_text("AI lib loaded successfully.")
                # Next frame will show "Loading AI model..."
                self._AgentCls = Agent
                self._phase = "pre_load_model"
            except Exception as exc:
                pygame.event.post(pygame.event.Event(self._error_evt, message=str(exc)))
            return

        if self._phase == "pre_load_model":
            print("Loading Custom AI models...")
            self._set_text("Loading Custom AI models...")
            self._phase = "do_load_model"
            return

        if self._phase == "do_load_model":
            try:
                agent = self._AgentCls.load(AI_MODEL_PATH)
                print("Starting Game...")
                self._set_text("Starting Game...")
                # Hand off to GameScene via event so the manager can switch scenes
                pygame.event.post(pygame.event.Event(self._loaded_evt, agent=agent))
                self._phase = "done"
            except Exception as exc:
                pygame.event.post(pygame.event.Event(self._error_evt, message=str(exc)))
            return

        # "done" -> nothing more to do here

    def draw(self, r):
        r.draw_color = (0, 0, 0, 255)
        r.clear()

        if self._failed:
            # Show the last status line plus an error
            if self._txt:
                self._txt.draw(dstrect=self._txt_rect)
            msg = f"Failed to load AI: {self._failed}"
            err_surf, _ = pygame.freetype.SysFont(None, 28).render(msg, fgcolor=(255, 80, 80))
            err_tex = Texture.from_surface(self._rend, err_surf)
            err_rect = err_tex.get_rect(center=(self._logical_w // 2, self._logical_h // 2 + 60))
            err_tex.draw(dstrect=err_rect)
        else:
            if self._txt:
                self._txt.draw(dstrect=self._txt_rect)

        self._has_drawn_once = True