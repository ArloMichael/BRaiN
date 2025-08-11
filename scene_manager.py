"""
scene_manager.py
Core scene interfaces and a very small stack‑based manager.
"""

from typing import Optional, List


class Scene:
    """Base‑class every scene inherits from."""
    def handle_event(self, event) -> Optional["Scene"]:
        return None           # Return a *new* Scene to trigger a change.

    def update(self, dt: float) -> None:
        pass

    def draw(self, renderer) -> None:
        pass

    # Optional enter/exit hooks
    def on_enter(self, previous: Optional["Scene"]) -> None:
        pass

    def on_exit(self, next_scene: Optional["Scene"]) -> None:
        pass


class SceneManager:
    """
    A tiny stack‑based manager.  
    Only `change()` is implemented because the demo never pushes
    overlaid scenes, but you could add push/pop later.
    """
    def __init__(self, start_scene: Scene):
        self._stack: List[Scene] = [start_scene]
        start_scene.on_enter(None)

    # ---------------------------------------------------------------- public
    @property
    def current(self) -> Optional[Scene]:
        return self._stack[-1] if self._stack else None

    def change(self, new_scene: Scene) -> None:
        cur = self.current
        if cur:
            cur.on_exit(new_scene)
            self._stack.pop()

        self._stack.append(new_scene)
        new_scene.on_enter(cur)