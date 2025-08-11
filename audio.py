from __future__ import annotations

import os
import random
import math
import pygame
import threading
import time
from dataclasses import dataclass
from typing import Optional, Dict
from enum import Enum, auto

# Optional dependency for loudness analysis
try:
    import numpy as np
    _HAS_NUMPY = True
except ImportError:
    _HAS_NUMPY = False

# Supported audio file extensions
_AUDIO_EXTS = {".wav", ".ogg", ".mp3", ".flac"}


class MusicType(Enum):
    """Specifies the category of music to play."""
    MENU = auto()
    GAME = auto()


def _is_audio_file(path: str) -> bool:
    return os.path.splitext(path)[1].lower() in _AUDIO_EXTS


def _base_name(filename: str) -> str:
    # Group variants by base (e.g., footstep, footstep2 -> "footstep")
    stem = os.path.splitext(os.path.basename(filename))[0]
    i = 0
    while i < len(stem) and stem[i].isalpha():
        i += 1
    return (stem[:i] or stem).lower()


@dataclass
class SfxEntry:
    sound: pygame.mixer.Sound
    gain: float = 1.0


class AudioManager:
    """
    Global audio manager: handles SFX groups + single-track music for different contexts (e.g., menu, game).
    Public control surface is designed to be driven by a Settings screen with On/Off toggles.
    """

    def __init__(
        self,
        sfx_dir: str = "audio/sfx",
        menu_music_dir: str = "audio/music/menu",
        game_music_dir: str = "audio/music/game",
        channels: int = 16,
        sfx_volume: float = 0.8,
        music_volume: float = 0.5,
        normalize: bool = True,
        target_rms: float = 0.2,
        max_gain: float = 3.0,
        startup_sfx_name: Optional[str] = "startup",
    ) -> None:
        self.sfx_dir = sfx_dir
        self.menu_music_dir = menu_music_dir
        self.game_music_dir = game_music_dir

        # Base volumes (0..1). Use setters to apply live.
        self._sfx_volume = float(max(0.0, min(1.0, sfx_volume)))
        self._music_volume = float(max(0.0, min(1.0, music_volume)))

        # Feature flags
        self._normalize = normalize
        self._target_rms = target_rms
        self._max_gain = max_gain

        # User-facing enable/disable states
        self._muted = False                # master mute
        self._sfx_enabled = True
        self._music_enabled = True
        self._ambience_enabled = True      # placeholder; exposed for Settings toggle

        # SFX bookkeeping
        self._sfx: Dict[str, list[SfxEntry]] = {}
        self._round_robin = 0

        # Startup SFX
        self._startup_sfx_name = startup_sfx_name
        self._startup_sound_played = False

        # Init mixer
        if not pygame.mixer.get_init():
            pygame.mixer.init()
        pygame.mixer.set_num_channels(channels)
        pygame.mixer.music.set_volume(self._effective_music_volume())

        # --- Simple playlist + worker for music cycling ---
        self._playlists: Dict[MusicType, list[str]] = {
            MusicType.MENU: [],
            MusicType.GAME: [],
        }
        self._playlist_idx: Dict[MusicType, int] = {
            MusicType.MENU: 0,
            MusicType.GAME: 0,
        }
        self._current_music_type: Optional[MusicType] = None
        self._music_loop: bool = True
        self._music_fade_ms: int = 500
        self._music_thread: Optional[threading.Thread] = None
        self._stop_thread: bool = False
        self._start_music_worker()

        # Load SFX
        self._rescan_sfx()

    # ---------- Loudness & normalization ----------
    def _analyze_loudness(self, snd: pygame.mixer.Sound) -> tuple[float, float]:
        if not _HAS_NUMPY:
            return 1.0, 1.0
        try:
            arr = pygame.sndarray.array(snd)
        except Exception:
            return 1.0, 1.0
        # Convert array to float [-1,1]
        if arr.dtype == np.int16:
            x = arr.astype(np.float32) / 32768.0
        elif arr.dtype == np.int8:
            x = arr.astype(np.float32) / 128.0
        elif arr.dtype == np.int32:
            x = arr.astype(np.float32) / 2147483648.0
        else:
            x = arr.astype(np.float32)
        x = x.flatten()
        rms = float(np.sqrt((x * x).mean())) if _HAS_NUMPY else 1.0
        peak = float(max(abs(x))) if x.size else 1.0
        return max(rms, 1e-6), max(peak, 1e-6)

    def _compute_gain(self, snd: pygame.mixer.Sound) -> float:
        if not self._normalize:
            return 1.0
        rms, peak = self._analyze_loudness(snd)
        g = self._target_rms / rms
        g = min(g, self._max_gain)
        g = min(g, 0.98 / peak)
        return max(0.0, g)

    # ---------- SFX loading ----------
    def _rescan_sfx(self) -> None:
        self._sfx.clear()
        if not os.path.isdir(self.sfx_dir):
            return
        for fn in os.listdir(self.sfx_dir):
            path = os.path.join(self.sfx_dir, fn)
            if os.path.isfile(path) and _is_audio_file(path):
                try:
                    snd = pygame.mixer.Sound(path)
                    gain = self._compute_gain(snd)
                    # Apply effective volume (considers enabled + master mute)
                    snd.set_volume(self._effective_sfx_volume())
                    base = _base_name(fn)
                    self._sfx.setdefault(base, []).append(SfxEntry(snd, gain))
                except pygame.error:
                    continue
        for group in self._sfx.values():
            random.shuffle(group)

    # ---------- Music playlist helpers ----------
    def _scan_music_files(self, directory: str) -> list[str]:
        """Return a sorted list of absolute paths to audio files in a directory."""
        try:
            files = []
            for f in os.listdir(directory):
                p = os.path.join(directory, f)
                if os.path.isfile(p) and _is_audio_file(p):
                    files.append(os.path.abspath(p))
            files.sort()
            return files
        except Exception:
            return []

    def _play_next_track(self) -> bool:
        """Load and play the next track for the current music type. Returns True on success."""
        mt = self._current_music_type
        if mt is None:
            return False
        plist = self._playlists.get(mt, [])
        if not plist:
            return False
        idx = self._playlist_idx[mt] % len(plist)
        path = plist[idx]
        self._playlist_idx[mt] = (idx + 1) % len(plist)
        try:
            pygame.mixer.music.load(path)
            pygame.mixer.music.set_volume(self._effective_music_volume())
            # Play a single pass; worker will advance the playlist.
            pygame.mixer.music.play(0, fade_ms=self._music_fade_ms)
            return True
        except pygame.error as e:
            print(f"Error playing music file {path}: {e}")
            return False

    def _start_music_worker(self) -> None:
        if self._music_thread is not None:
            return

        def worker():
            while not self._stop_thread:
                # Only advance when looping is enabled and music should be audible
                if (
                    self._music_loop
                    and self._current_music_type is not None
                    and not self._muted
                    and self._music_enabled
                ):
                    if not pygame.mixer.music.get_busy():
                        # Try to advance; if playlist empty, just wait a bit
                        advanced = self._play_next_track()
                        if not advanced:
                            time.sleep(0.3)
                    else:
                        time.sleep(0.1)
                else:
                    time.sleep(0.2)

        self._music_thread = threading.Thread(target=worker, daemon=True)
        self._music_thread.start()

    # ---------- Effective volumes considering toggles/mute ----------
    def _effective_sfx_volume(self) -> float:
        if self._muted or not self._sfx_enabled:
            return 0.0
        return self._sfx_volume

    def _effective_music_volume(self) -> float:
        if self._muted or not self._music_enabled:
            return 0.0
        return self._music_volume

    # ---------- Public playback ----------
    def has_sfx(self, name: str) -> bool:
        return name.lower() in self._sfx

    def play_sfx(self, name: str, pan: float = 0.0, volume: Optional[float] = None) -> None:
        # Respect master mute and SFX toggle
        if self._muted or not self._sfx_enabled:
            return
        group = self._sfx.get(name.lower())
        if not group:
            return
        entry = random.choice(group)
        snd = entry.sound
        ch = pygame.mixer.find_channel()
        if ch is None:
            idx = self._round_robin % pygame.mixer.get_num_channels()
            ch = pygame.mixer.Channel(idx)
            self._round_robin += 1
        # Apply volumes
        base = self._effective_sfx_volume() if volume is None else max(0.0, min(1.0, volume))
        base *= entry.gain
        base = min(base, 1.0)
        # Stereo panning
        theta = (pan + 1) * (math.pi / 4)
        left = base * math.cos(theta)
        right = base * math.sin(theta)
        ch.set_volume(left, right)
        ch.play(snd)

    def play_music(
        self,
        path: Optional[str] = None,
        music_type: MusicType = MusicType.GAME,
        loop: bool = True,
        fade_ms: int = 500,
    ) -> None:
        """
        Plays a music track from either the menu or game music directory.
        Respects startup SFX, master mute, and music enabled toggle.

        Behavior:
        - loop=True (default): cycles through all tracks in the target directory.
        - loop=False: plays a single track (no auto-advance).
        - If 'path' refers to a file inside the directory, starts there and continues.
        - If 'path' is outside, plays only that file once (no cycling).
        """
        # --- One-time startup sound (when entering menu music the first time) ---
        if (
            music_type == MusicType.MENU
            and not self._startup_sound_played
            and self._startup_sfx_name
        ):
            if self.has_sfx(self._startup_sfx_name):
                self.play_sfx(self._startup_sfx_name)
            self._startup_sound_played = True

        target_dir = self.menu_music_dir if music_type == MusicType.MENU else self.game_music_dir
        self._music_fade_ms = int(fade_ms)
        self._music_loop = bool(loop)  # loop now means "loop the playlist"

        if not os.path.isdir(target_dir):
            print(f"Warning: Music directory not found: {target_dir}")
            return

        # Build/refresh playlist for the target type
        playlist = self._scan_music_files(target_dir)
        if not playlist:
            print(f"Warning: No music files found in {target_dir}")
            return
        self._playlists[music_type] = playlist

        # Determine starting index
        start_idx = 0
        abs_path: Optional[str] = None
        if path:
            abs_path = os.path.abspath(path if os.path.isabs(path) else os.path.join(target_dir, path))
            if abs_path in playlist:
                try:
                    start_idx = playlist.index(abs_path)
                except ValueError:
                    start_idx = 0
            elif not os.path.isfile(abs_path):
                print(f"Warning: Music file not found: {path}")
                return

        # If they gave a file outside the directory, play just that once and return
        if abs_path and abs_path not in playlist:
            try:
                pygame.mixer.music.load(abs_path)
                pygame.mixer.music.set_volume(self._effective_music_volume())
                pygame.mixer.music.play(0, fade_ms=self._music_fade_ms)
            except pygame.error as e:
                print(f"Error playing music file {abs_path}: {e}")
            # Do not auto-advance when playing an external file
            self._current_music_type = None
            return

        self._playlist_idx[music_type] = start_idx
        self._current_music_type = music_type

        # Start playing from the selected index; the worker will advance if loop=True
        pygame.mixer.music.stop()
        started = self._play_next_track()
        if not started:
            print("Warning: Could not start music playback.")

    def stop_music(self, fade_ms: int = 300) -> None:
        pygame.mixer.music.fadeout(fade_ms)
        # Also stop auto-advancing until play_music is called again
        self._current_music_type = None
        self._music_loop = False

    # ---------- Public control surface for SettingsScene ----------
    def set_master_enabled(self, enabled: bool) -> None:
        """Enable/disable all audio (master)."""
        self._muted = not bool(enabled)
        # Apply to existing channels
        for group in self._sfx.values():
            for entry in group:
                entry.sound.set_volume(self._effective_sfx_volume())
        pygame.mixer.music.set_volume(self._effective_music_volume())

    def is_master_enabled(self) -> bool:
        return not self._muted

    def set_music_enabled(self, enabled: bool) -> None:
        self._music_enabled = bool(enabled)
        pygame.mixer.music.set_volume(self._effective_music_volume())

    def is_music_enabled(self) -> bool:
        return self._music_enabled

    def set_sfx_enabled(self, enabled: bool) -> None:
        self._sfx_enabled = bool(enabled)
        for group in self._sfx.values():
            for entry in group:
                entry.sound.set_volume(self._effective_sfx_volume())

    def is_sfx_enabled(self) -> bool:
        return self._sfx_enabled

    # "Ambience/Ambiance" are synonyms; expose both.
    def set_ambience_enabled(self, enabled: bool) -> None:
        self._ambience_enabled = bool(enabled)
        # Placeholder: if you later add a looped ambience channel, apply volume here.

    def set_ambiance_enabled(self, enabled: bool) -> None:
        self.set_ambience_enabled(enabled)

    def is_ambience_enabled(self) -> bool:
        return self._ambience_enabled

    # Volumes (0..1). These DO NOT flip the enabled flags; they set the base.
    def set_music_volume(self, vol: float) -> None:
        self._music_volume = float(max(0.0, min(1.0, vol)))
        pygame.mixer.music.set_volume(self._effective_music_volume())

    def set_sfx_volume(self, vol: float) -> None:
        self._sfx_volume = float(max(0.0, min(1.0, vol)))
        for group in self._sfx.values():
            for entry in group:
                entry.sound.set_volume(self._effective_sfx_volume())

    # Legacy toggle (kept for compatibility with existing code)
    def toggle_mute(self) -> None:
        self.set_master_enabled(not self.is_master_enabled())

    # ---------- Introspection helpers ----------
    def state_dict(self) -> Dict[str, bool]:
        """Convenience snapshot for UI (Audio/Music/SFX/Ambience)."""
        return {
            "audio": self.is_master_enabled(),
            "music": self.is_music_enabled(),
            "sfx": self.is_sfx_enabled(),
            "ambience": self.is_ambience_enabled(),
        }


# Global instance
audio_manager = AudioManager()


# ---------------- Module-level convenience wrappers ----------------

def play_music(
    path: Optional[str] = None,
    music_type: MusicType = MusicType.GAME,
    loop: bool = True,
    fade_ms: int = 500
) -> None:
    """Convenience function to play music."""
    audio_manager.play_music(path=path, music_type=music_type, loop=loop, fade_ms=fade_ms)


def stop_music(fade_ms: int = 300) -> None:
    """Convenience function to stop music."""
    audio_manager.stop_music(fade_ms=fade_ms)


def toggle_mute() -> None:
    """Convenience function to toggle master mute."""
    audio_manager.toggle_mute()


def has_sfx(name: str) -> bool:
    """Convenience function to check if a sound effect exists."""
    return audio_manager.has_sfx(name=name)


def play_sfx(name: str, pan: float = 0.0, volume: Optional[float] = None) -> None:
    """Convenience function to play a sound effect."""
    audio_manager.play_sfx(name=name, pan=pan, volume=volume)


# New convenience functions for SettingsScene
def set_master_enabled(enabled: bool) -> None:
    audio_manager.set_master_enabled(enabled)

def set_music_enabled(enabled: bool) -> None:
    audio_manager.set_music_enabled(enabled)

def set_sfx_enabled(enabled: bool) -> None:
    audio_manager.set_sfx_enabled(enabled)

def set_ambience_enabled(enabled: bool) -> None:
    audio_manager.set_ambience_enabled(enabled)

# Allow alternate spelling
def set_ambiance_enabled(enabled: bool) -> None:
    audio_manager.set_ambience_enabled(enabled)
