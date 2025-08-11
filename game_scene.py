# game_scene.py
"""
GameScene – integrates the isometric world/camera/sound stack
(world.py, graphics.py, controls.py, sound.py) into the Scene
framework that uses pygame._sdl2.video.Renderer.

We render onto an off-screen pygame.Surface using graphics.draw_grid(),
then upload that Surface to a Texture each frame for the SDL2 Renderer.
"""
from __future__ import annotations

from typing import Optional
import os

import pygame
from pygame._sdl2.video import Texture

from scene_manager import Scene

# Reuse your existing modules
from game import world
from game import graphics
import audio

# ------------------------------- CONFIG
ASSET_DIR       = "images"
FONT_PATH       = "tahoma.ttf"
TERRAIN_TYPES   = ["dirt","grass","rocky","urban1","urban2","water1","water2","water3"]
GRID_SIZE       = 128
TILE_SIZE_START = 64
ZOOM_STEP       = 8
LAG_RATE        = 4.0
SFX_PAN_STEP    = 0.15
BOT_TICKS_BETWEEN_MOVES = 7  # move every update tick (raise to slow down)

def _ensure_display_surface() -> None:
    if pygame.display.get_surface() is None:
        try:
            pygame.display.set_mode((1, 1), flags=pygame.HIDDEN)
        except Exception:
            pygame.display.set_mode((1, 1))

class GameScene(Scene):
    def __init__(self, rend, agent):
        self._rend = rend
        self.agent = agent  # ← provided by LoadingScene
        lw, lh = getattr(rend, "logical_size", (800, 300))
        self._logical_w, self._logical_h = int(lw), int(lh)

        _ensure_display_surface()

        # ---------- world & players ----------
        self.grid = world.generate_grid(GRID_SIZE, TERRAIN_TYPES)
        self.player = world.Player(x=GRID_SIZE // 2, y=GRID_SIZE // 2, scale=1.0)
        self.player.update_terrain(self.grid)

        self.bot = world.Player(
            x=min(GRID_SIZE - 1, self.player.x + 1),
            y=self.player.y,
            scale=1.0
        )
        self.bot.update_terrain(self.grid)
        self.bots = [self.bot]

        # ---------- assets ----------
        self.tile_size = TILE_SIZE_START
        self.images = graphics.load_images(ASSET_DIR, TERRAIN_TYPES, self.tile_size)
        self.player_img = graphics.load_player(ASSET_DIR, self.tile_size)
        self.bot_img = graphics.load_bot(ASSET_DIR, self.tile_size)
        self.font = graphics.get_font(FONT_PATH, 20)
        self.camera = graphics.Camera()

        # ---------- collectables ----------
        self.collectable_img = pygame.image.load(os.path.join(ASSET_DIR, "collectable.png")).convert_alpha()
        self.collectable_img = pygame.transform.scale(self.collectable_img, (self.tile_size, self.tile_size))
        self.collectables = world.generate_collectables(
            GRID_SIZE, count=50,
            forbidden=[(self.player.x, self.player.y), (self.bot.x, self.bot.y)]
        )

        # Win condition / state
        self._win_target = max(1, len(self.collectables) // 2 + 1)  # "more than half"
        self._ended = False
        self._winner: Optional[str] = None

        # HUD counters
        self.player_collected = 0
        self.bot_collected = 0

        # Sound
        self.snd = audio.audio_manager

        # Off-screen framebuffer
        self._fb = pygame.Surface((self._logical_w, self._logical_h), pygame.SRCALPHA)
        self._fb_tex: Optional[Texture] = None

        # FPS estimate
        self._fps = 0.0

        # Timer tracking (milliseconds)
        self._elapsed_ms = 0.0
        self._final_time_ms = None

        # Movement SFX tracking
        self._last_pos = (self.player.x, self.player.y)

        # Assets dirty flag (zoom)
        self._assets_dirty = False

        # Bot tick counter
        self._bot_tick = BOT_TICKS_BETWEEN_MOVES - 1

        # Bot start position (for AI)
        self._bot_prev = (self.bot.x, self.bot.y)

        # Deferred scene switch holder
        self._pending_scene: Optional[Scene] = None

    def _format_time(self, ms: float) -> str:
        total_cs = int(ms // 10)  # centiseconds
        minutes = total_cs // 6000
        seconds = (total_cs // 100) % 60
        centis = total_cs % 100
        return f"{minutes:02d}:{seconds:02d}.{centis:02d}"

    def on_enter(self, previous: Optional[Scene]) -> None:
        audio.play_music(music_type=audio.MusicType.GAME)

    def on_exit(self, next_scene: Optional[Scene]) -> None:
        audio.stop_music(fade_ms=500)

    def handle_event(self, e):
        # Process any deferred scene change first (so SceneManager.change() runs via handle_event)
        if self._pending_scene is not None:
            next_scene, self._pending_scene = self._pending_scene, None
            return next_scene

        if self._ended and e.type == pygame.KEYUP:
            if e.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
                # Optional fallback: restart directly if WinScene didn't engage yet
                from game_scene import GameScene
                return GameScene(self._rend, self.agent)
            if e.key == pygame.K_ESCAPE:
                from menu_scene import MenuScene
                return MenuScene(self._rend)

        if e.type == pygame.KEYUP and e.key == pygame.K_ESCAPE:
            from menu_scene import MenuScene
            return MenuScene(self._rend)

        if e.type == pygame.KEYDOWN:
            if e.key == pygame.K_MINUS:
                self.tile_size = max(16, self.tile_size - ZOOM_STEP)
                self._assets_dirty = True
            elif e.key == pygame.K_EQUALS:
                self.tile_size += ZOOM_STEP
                self._assets_dirty = True
            elif not self._ended and e.key in (pygame.K_UP, pygame.K_w):
                self.player.y = max(0, self.player.y - 1)
            elif not self._ended and e.key in (pygame.K_DOWN, pygame.K_s):
                self.player.y = min(GRID_SIZE - 1, self.player.y + 1)
            elif not self._ended and e.key in (pygame.K_LEFT, pygame.K_a):
                self.player.x = max(0, self.player.x - 1)
            elif not self._ended and e.key in (pygame.K_RIGHT, pygame.K_d):
                self.player.x = min(GRID_SIZE - 1, self.player.x + 1)

        elif e.type == pygame.MOUSEBUTTONDOWN:
            if e.button == 4:
                self.tile_size += ZOOM_STEP
                self._assets_dirty = True
            elif e.button == 5:
                self.tile_size = max(16, self.tile_size - ZOOM_STEP)
                self._assets_dirty = True

    def _try_finish(self):
        """If someone crossed the threshold, schedule a WinScene via handle_event()."""
        if self._ended:
            return

        if self.player_collected >= self._win_target:
            self._ended = True
            self._winner = "Player"
        elif self.bot_collected >= self._win_target:
            self._ended = True
            self._winner = "Bot"

        if self._ended and self._pending_scene is None:
            # Audio cues
            try:
                audio.stop_music(fade_ms=200)
                audio.play_sfx("fanfare")
            except Exception:
                pass

            # Prepare WinScene and nudge event loop so handle_event() returns it
            self._final_time_ms = self._elapsed_ms
            from win_scene import WinScene
            self._pending_scene = WinScene(
                self._rend,
                winner=self._winner,
                player_score=self.player_collected,
                bot_score=self.bot_collected,
                total=len(self.collectables),
                agent=self.agent,
                elapsed_ms=int(self._final_time_ms or 0),
            )
            pygame.event.post(pygame.event.Event(pygame.USEREVENT, {"kind": "scene_change"}))

    def update(self, dt: float) -> None:
        # If assets were resized
        if self._assets_dirty:
            self.images = graphics.load_images(ASSET_DIR, TERRAIN_TYPES, self.tile_size)
            self.player_img = graphics.load_player(ASSET_DIR, self.tile_size)
            self.bot_img = graphics.load_bot(ASSET_DIR, self.tile_size)
            self.collectable_img = pygame.image.load(os.path.join(ASSET_DIR, "collectable.png")).convert_alpha()
            self.collectable_img = pygame.transform.scale(self.collectable_img, (self.tile_size, self.tile_size))
            self._assets_dirty = False

        # If ended, keep drawing but skip gameplay updates; scene swap will happen via handle_event()
        if self._ended:
            return

        # Footstep SFX (player)
        cur_pos = (self.player.x, self.player.y)
        if cur_pos != self._last_pos:
            dx = self.player.x - self._last_pos[0]
            pan = (-SFX_PAN_STEP if dx < 0 else (SFX_PAN_STEP if dx > 0 else 0.0))
            audio.play_sfx("footstep", pan=pan)
        self._last_pos = cur_pos

        # Player & bot collectables
        for c in self.collectables:
            if not c.collected:
                if c.x == self.player.x and c.y == self.player.y:
                    c.collected = True
                    self.player_collected += 1
                    self.player.scale += 0.1  # <-- grow player a bit
                    self._try_finish()
                else:
                    for b in self.bots:
                        if c.x == b.x and c.y == b.y:
                            c.collected = True
                            self.bot_collected += 1
                            b.scale += 0.1  # <-- grow bot a bit
                            self._try_finish()
                            break

        if not hasattr(self, '_move_acc'):
            self._move_acc = [0.0, 0.0]

        # Player movement (continuous → tile steps)
        keys = pygame.key.get_pressed()
        speed = 5.0
        move_x = float(keys[pygame.K_RIGHT] or keys[pygame.K_d]) - float(keys[pygame.K_LEFT] or keys[pygame.K_a])
        move_y = float(keys[pygame.K_DOWN]  or keys[pygame.K_s]) - float(keys[pygame.K_UP]   or keys[pygame.K_w])
        self._move_acc[0] += move_x * speed * dt
        self._move_acc[1] += move_y * speed * dt

        while self._move_acc[0] >= 1.0:
            self.player.x += 1; self._move_acc[0] -= 1.0
        while self._move_acc[0] <= -1.0:
            self.player.x -= 1; self._move_acc[0] += 1.0
        while self._move_acc[1] >= 1.0:
            self.player.y += 1; self._move_acc[1] -= 1.0
        while self._move_acc[1] <= -1.0:
            self.player.y -= 1; self._move_acc[1] += 1.0

        # AI-driven bot move
        self._bot_tick += 1
        if self._bot_tick % BOT_TICKS_BETWEEN_MOVES == 0 and self.agent is not None:
            coins_list = [(c.x, c.y) for c in self.collectables if not c.collected]
            new_x, new_y = self.agent.next(
                coins=coins_list,
                current=(self.bot.x, self.bot.y),
                prev=self._bot_prev,
            )
            self._bot_prev = (self.bot.x, self.bot.y)
            self.bot.x = max(0, min(GRID_SIZE - 1, int(new_x)))
            self.bot.y = max(0, min(GRID_SIZE - 1, int(new_y)))
            self.bot.update_terrain(self.grid)

        # Terrain + camera follow
        self.player.update_terrain(self.grid)
        target_x, target_y = self.player.x, self.player.y
        alpha = min(1.0, LAG_RATE * max(0.0, dt))
        self.camera.x += (target_x - self.camera.x) * alpha
        self.camera.y += (target_y - self.camera.y) * alpha

        # FPS EMA
        if dt > 0:
            inst = 1.0 / dt
            self._fps = inst if self._fps <= 0 else (0.9 * self._fps + 0.1 * inst)

        # Advance timer only if game not ended
        self._elapsed_ms += dt * 1000.0

    def draw(self, r) -> None:
        graphics.draw_grid(
            surface=self._fb,
            images=self.images,
            grid=self.grid,
            camera=self.camera,
            tile_size=self.tile_size,
            font=self.font,
            player=self.player,
            player_img=self.player_img,
            fps_val=self._fps,
            collectables=self.collectables,
            collectable_img=self.collectable_img,
            bots=self.bots,
            bot_img=self.bot_img,
            player_taken=self.player_collected,
            bot_taken=self.bot_collected,
            timer_text=self._format_time(self._elapsed_ms),
        )

        # If ended but SceneManager hasn't swapped yet (e.g., frame timing),
        # draw a quick overlay to show the state.
        if self._ended:
            overlay = pygame.Surface((self._logical_w, self._logical_h), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 140))
            self._fb.blit(overlay, (0, 0))
            big = graphics.get_font(FONT_PATH, 48)
            small = graphics.get_font(FONT_PATH, 22)
            title = f"{self._winner} Wins!"
            t = big.render(title, True, (245, 245, 255))
            s1 = small.render("Press Enter for rematch • Esc for menu", True, (225, 225, 225))
            self._fb.blit(t, ((self._logical_w - t.get_width()) // 2, (self._logical_h // 2) - 40))
            self._fb.blit(s1, ((self._logical_w - s1.get_width()) // 2, (self._logical_h // 2) + 12))

        self._fb_tex = Texture.from_surface(self._rend, self._fb)
        self._fb_tex.draw(dstrect=(0, 0, self._logical_w, self._logical_h))