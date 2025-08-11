"""
world.py – game‑world data (no pygame dependency)
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List

@dataclass
class Collectable:
    x: int
    y: int
    collected: bool = False

@dataclass
class TerrainBlock:
    x: int
    y: int
    terrain: str
    height: int = 0


@dataclass
class Player:
    x: int
    y: int
    scale: float = 1.0          # ← NEW (default = normal size)
    terrain: str = "unknown"

    def update_terrain(self, grid: Dict[str, TerrainBlock]) -> None:
        key = f"{self.x}-{self.y}"          # same fix as before
        block = grid.get(key)
        self.terrain = block.terrain if block else "void"


def generate_grid(grid_size: int, terrain_types: List[str]) -> Dict[str, TerrainBlock]:
    """Return a dictionary keyed by 'x-y' → TerrainBlock."""
    grid: Dict[str, TerrainBlock] = {}
    for x in range(grid_size):
        for y in range(grid_size):
            key = f"{x}-{y}"
            grid[key] = TerrainBlock(
                x=x,
                y=y,
                terrain=random.choice(terrain_types),
                height=random.randint(0, 100),
            )
    return grid

def generate_collectables(grid_size: int, count: int, forbidden: Optional[List[Tuple[int, int]]] = None) -> List[Collectable]:
    """Randomly scatter collectables, avoiding forbidden positions."""
    forbidden = set(forbidden or [])
    positions = set()
    while len(positions) < count:
        x = random.randint(0, grid_size - 1)
        y = random.randint(0, grid_size - 1)
        if (x, y) not in positions and (x, y) not in forbidden:
            positions.add((x, y))
    return [Collectable(x, y) for x, y in positions]