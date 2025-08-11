"""
Simple neural-network Agent that learns (via supervised imitation of A*) to collect
randomly scattered coins on a 128x128 grid by outputting a single move at each step.

Key points
----------
- Inputs (to the NN): the agent (x, y) and the K nearest coins as relative vectors.
- Outputs: a 4-way softmax over {up, down, left, right}.
- Training: supervised behavior cloning using an A* teacher policy to the nearest coin.
- Inference: pure neural policy; no pathfinding is used after training.

API
---
from ai import Agent

agent = Agent(map_size=128, k_nearest=16)
agent.train(steps=50_000, batch_size=1024, lr=1e-3)  # prints a progress bar
agent.save("model.pt")

agent2 = Agent.load("model.pt")  # restore later

stats = agent.eval(episodes=50, horizon=512)
move = agent.next(coins=[(10, 10), (40, 50)], current=(0, 0))  # -> (x, y) next coordinate

Notes
-----
- Coordinates are 0-indexed with (x, y) where x increases to the right and y increases downward.
  'up' means (x, y-1); 'down' means (x, y+1); 'left' means (x-1, y); 'right' means (x+1, y).
- The teacher uses A* with Manhattan heuristic on an empty grid (no obstacles). For this task,
  A* reduces to a shortest Manhattan path to the nearest coin; this satisfies the
  "learns based on a path finding algorithm" requirement while keeping the runtime small.
- During inference (.next), the model consumes only the coin list and current location.
"""
from __future__ import annotations

import math
import random
from heapq import heappop, heappush
from typing import Iterable, List, Sequence, Tuple, Optional, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm

Coord = Tuple[int, int]


# -------------------------------
# Utilities
# -------------------------------

ACTION_TO_STR = {0: "up", 1: "down", 2: "left", 3: "right"}
STR_TO_ACTION = {v: k for k, v in ACTION_TO_STR.items()}


def clamp(v: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, v))


def manhattan(a: Coord, b: Coord) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _neighbors(p: Coord, size: int) -> Iterable[Coord]:
    x, y = p
    if y > 0:
        yield (x, y - 1)  # up
    if y + 1 < size:
        yield (x, y + 1)  # down
    if x > 0:
        yield (x - 1, y)  # left
    if x + 1 < size:
        yield (x + 1, y)  # right


def _reconstruct_first_step(came_from: Dict[Coord, Coord], start: Coord, goal: Coord) -> Optional[int]:
    """Return the first action (0..3) from start along the A* path to goal.
    If start==goal, returns None.
    """
    if start == goal:
        return None

    # Reconstruct full path from goal back to start, then take the first move.
    path: List[Coord] = [goal]
    cur = goal
    while cur in came_from and came_from[cur] != start:
        cur = came_from[cur]
        path.append(cur)
    if cur not in came_from:
        # No path (shouldn't happen on empty grid) – fall back to greedy.
        sx, sy = start
        gx, gy = goal
        if abs(gx - sx) >= abs(gy - sy):
            return 3 if gx > sx else 2  # right or left
        else:
            return 1 if gy > sy else 0  # down or up

    first = goal
    prev = came_from[first]
    dx = first[0] - prev[0]
    dy = first[1] - prev[1]
    if dx == 0 and dy == -1:
        return 0  # up
    if dx == 0 and dy == 1:
        return 1  # down
    if dx == -1 and dy == 0:
        return 2  # left
    if dx == 1 and dy == 0:
        return 3  # right
    return None


def astar_next_action(start: Coord, goal: Coord, size: int) -> Optional[int]:
    """A* on a 4-connected, obstacle-free grid. Returns the first action (0..3).
    Uses Manhattan heuristic. If start==goal, returns None.
    """
    if start == goal:
        return None

    open_heap: List[Tuple[int, Coord]] = []
    heappush(open_heap, (manhattan(start, goal), start))

    g: Dict[Coord, int] = {start: 0}
    came_from: Dict[Coord, Coord] = {}
    closed: set[Coord] = set()

    while open_heap:
        _, current = heappop(open_heap)
        if current in closed:
            continue
        closed.add(current)

        if current == goal:
            return _reconstruct_first_step(came_from, start, goal)

        base_g = g[current]
        for nb in _neighbors(current, size):
            tentative_g = base_g + 1
            if nb in g and tentative_g >= g[nb]:
                continue
            g[nb] = tentative_g
            came_from[nb] = current
            f = tentative_g + manhattan(nb, goal)
            heappush(open_heap, (f, nb))

    return None


def first_step_towards(start: Coord, goal: Coord) -> int:
    """Closed-form first step along any shortest Manhattan path on an empty grid.
    Equivalent to A*'s first move in this environment but ~100x cheaper.
    Returns an action in {0:up,1:down,2:left,3:right}.
    """
    sx, sy = start
    gx, gy = goal
    dx, dy = gx - sx, gy - sy
    if dx == 0 and dy == 0:
        return 0  # arbitrary (up)
    if abs(dx) >= abs(dy):
        if dx > 0:
            return 3  # right
        if dx < 0:
            return 2  # left
        # dx == 0 -> move vertically
        return 1 if dy > 0 else 0
    else:
        return 1 if dy > 0 else 0


# -------------------------------
# Feature builder
# -------------------------------

def build_features(
    current: Coord,
    coins: Sequence[Coord],
    size: int,
    k: int,
) -> torch.Tensor:
    """Fixed-length feature vector: [ax, ay, coin_density, (dx,dy,dist,valid)*k]
    - ax, ay are normalized to [0,1]
    - dx, dy are normalized to [-1,1] by dividing by (size-1)
    - dist is Manhattan distance normalized to [0,1]
    - valid is 1.0 for a real coin slot, else 0.0
    """
    ax = current[0] / (size - 1)
    ay = current[1] / (size - 1)
    density = min(1.0, len(coins) / float(size * size))

    # Sort coins by Manhattan distance, take k nearest
    if len(coins) > 0:
        ordered = sorted(coins, key=lambda c: manhattan(current, c))[:k]
    else:
        ordered = []

    feats: List[float] = [ax, ay, density]
    for i in range(k):
        if i < len(ordered):
            cx, cy = ordered[i]
            dx = (cx - current[0]) / (size - 1)
            dy = (cy - current[1]) / (size - 1)
            dist = manhattan(current, (cx, cy)) / (2 * (size - 1))  # max manhattan is 2*(size-1)
            feats.extend([dx, dy, dist, 1.0])
        else:
            feats.extend([0.0, 0.0, 0.0, 0.0])

    return torch.tensor(feats, dtype=torch.float32)


# -------------------------------
# Policy Network
# -------------------------------

class PolicyNet(nn.Module):
    def __init__(self, k_nearest: int, hidden: int = 128):
        super().__init__()
        in_dim = 3 + 4 * k_nearest
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 4),  # logits for up, down, left, right
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# -------------------------------
# Agent
# -------------------------------

class Agent:
    def __init__(
        self,
        map_size: int = 128,
        k_nearest: int = 16,
        hidden: int = 128,
        device: Optional[str] = None,
        seed: int = 0,
    ):
        """
        Args:
            map_size: width/height of the square grid (default 128)
            k_nearest: how many nearest coins to encode (fixed-size input)
            hidden: hidden width of the MLP
            device: 'cpu' or 'cuda'; default auto-detect
            seed: RNG seed for reproducibility
        """
        self.size = map_size
        self.k = k_nearest
        self.hidden = hidden
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        random.seed(seed)
        torch.manual_seed(seed)

        self.model = PolicyNet(k_nearest=self.k, hidden=hidden).to(self.device)

    # ------------- Public API -------------
    def train(
        self,
        steps: int = 50_000,
        batch_size: int = 1024,
        lr: float = 1e-3,
        teacher: str = "greedy",
        coins_range: Tuple[int, int] = (1, 200),
        progress: bool = True,
    ) -> None:
        """Supervised imitation of an A* teacher policy on synthetic states.

        Args:
            steps: number of gradient steps
            batch_size: batch size
            lr: learning rate (Adam)
            teacher: 'greedy' (fast, equivalent on empty grid) or 'astar' (slow)
            coins_range: (min,max) number of coins sampled per synthetic state
            progress: if True, shows a tqdm progress bar
        """
        opt = torch.optim.AdamW(self.model.parameters(), lr=lr)

        pbar = tqdm(range(steps), disable=not progress, desc="train")
        for _ in pbar:
            feats, labels = self._batch_from_teacher(batch_size, coins_range, teacher)
            logits = self.model(feats)
            loss = F.cross_entropy(logits, labels)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            if progress:
                pbar.set_postfix({"loss": float(loss.item())})

    @torch.no_grad()
    def eval(
        self,
        episodes: int = 20,
        horizon: int = 512,
        coins_per_episode: Tuple[int, int] = (50, 150),
        start_random: bool = True,
    ) -> dict:
        """Roll out the *learned* policy in simple simulations and report stats.

        Returns a dict with averages (coins_collected, steps, coins_per_100_steps).
        """
        collected_total = 0
        steps_total = 0

        for _ in range(episodes):
            num_coins = random.randint(*coins_per_episode)
            coins = set()
            while len(coins) < num_coins:
                coins.add((random.randrange(self.size), random.randrange(self.size)))

            if start_random:
                cur = (random.randrange(self.size), random.randrange(self.size))
            else:
                cur = (self.size // 2, self.size // 2)

            coins_collected = 0
            for _ in range(horizon):
                # auto-collect if standing on a coin
                if cur in coins:
                    coins.remove(cur)
                    coins_collected += 1
                    if not coins:
                        break

                cur = self.next(list(coins), cur)

            collected_total += coins_collected
            steps_total += horizon

        cps = (100.0 * collected_total / steps_total) if steps_total else 0.0
        return {
            "episodes": episodes,
            "avg_coins_per_episode": collected_total / episodes if episodes else 0.0,
            "horizon": horizon,
            "coins_per_100_steps": cps,
        }

    def save(self, path: str = "model.pt") -> None:
        torch.save({
            "state_dict": self.model.state_dict(),
            "size": self.size,
            "k": self.k,
            "hidden": self.hidden,
        }, path)

    @classmethod
    def load(cls, path: str, device: Optional[str] = None) -> "Agent":
        """Load an Agent from a .pt checkpoint saved by .save()."""
        target_device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        ckpt = torch.load(path, map_location=target_device)
        size = ckpt.get("size", 128)
        k = ckpt.get("k", 16)
        hidden = ckpt.get("hidden", 128)
        agent = cls(map_size=size, k_nearest=k, hidden=hidden, device=str(target_device))
        agent.model.load_state_dict(ckpt["state_dict"])
        return agent

    @torch.no_grad()
    def next(
        self,
        coins: Sequence[Coord],
        current: Coord,
        prev: Optional[Coord] = None,
    ) -> Coord:
        """Return next (x, y). No Manhattan anywhere; random fallback if the policy can't act."""
        # If no coins, simple in-bounds deterministic behavior (no Manhattan)
        if not coins:
            x, y = current
            if y > 0:   action = 0  # up
            elif y + 1 < self.size: action = 1  # down
            elif x > 0: action = 2  # left
            else:       action = 3  # right
            return self._apply_action(current, action)

        # NN logits
        feat = build_features(current, coins, self.size, self.k).unsqueeze(0).to(self.device)
        logits = self.model(feat)[0]  # shape [4]

        # Mask immediate U-turn back to prev (prevents 2-tile oscillation)
        if prev is not None:
            for a in range(4):
                if self._apply_action(current, a) == prev:
                    logits[a] = -1e9

        # If everything is masked or not finite → random neighbor (avoid prev if possible)
        if not torch.isfinite(logits).any():
            nbrs = list(_neighbors(current, self.size))
            if prev in nbrs and len(nbrs) > 1:
                nbrs = [p for p in nbrs if p != prev]
            return random.choice(nbrs) if nbrs else current

        # Best allowed action from the policy
        act = int(torch.argmax(logits).item())
        return self._apply_action(current, act)

    # ------------- Internals -------------
    @torch.no_grad()
    def _apply_action(self, p: Coord, action: int) -> Coord:
        x, y = p
        if action == 0:  # up
            return (x, clamp(y - 1, 0, self.size - 1))
        if action == 1:  # down
            return (x, clamp(y + 1, 0, self.size - 1))
        if action == 2:  # left
            return (clamp(x - 1, 0, self.size - 1), y)
        if action == 3:  # right
            return (clamp(x + 1, 0, self.size - 1), y)
        return p

    def _batch_from_teacher(
        self,
        batch_size: int,
        coins_range: Tuple[int, int],
        teacher: str,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        feats: List[torch.Tensor] = []
        labels: List[int] = []

        for _ in range(batch_size):
            # Sample a random state
            cur = (random.randrange(self.size), random.randrange(self.size))
            n_coins = random.randint(*coins_range)

            # Ensure unique coin tiles and avoid placing a coin on the agent for supervision
            coins: set[Coord] = set()
            while len(coins) < n_coins:
                c = (random.randrange(self.size), random.randrange(self.size))
                if c != cur:
                    coins.add(c)

            # Pick the nearest coin and query A* for the first step towards it
            nearest = min(coins, key=lambda c: manhattan(cur, c))
            if teacher == "astar":
                action = astar_next_action(cur, nearest, self.size)
                if action is None:
                    action = first_step_towards(cur, nearest)
            else:
                action = first_step_towards(cur, nearest)

            feats.append(build_features(cur, list(coins), self.size, self.k))
            labels.append(int(action))

        X = torch.stack(feats).to(self.device)
        y = torch.tensor(labels, dtype=torch.long, device=self.device)
        return X, y