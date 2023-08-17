"""Baselines for environments."""
from __future__ import annotations

import abc
from random import choice

import numpy as np
from jaxtyping import Int, Bool


import environments


class Baseline(abc.ABC):
    """Baseline class."""

    @staticmethod
    def find_path(start: tuple[int, int],
                  end: tuple[int, int],
                  obstacles: Bool[np.ndarray, "width height"]) -> list[tuple[int, int]]:
        """
        Find the shortest path from start to end while avoiding obstacles.

        If no path is found, return an empty list.
        """

        w, h = obstacles.shape

        def get_neighbors(pos):
            """Get neighbors of pos."""
            x, y = pos
            neighbors = []
            if x > 0:
                neighbors.append((x - 1, y))
            if x < w - 1:
                neighbors.append((x + 1, y))
            if y > 0:
                neighbors.append((x, y - 1))
            if y < h - 1:
                neighbors.append((x, y + 1))
            return neighbors

        def get_path(parents, pos):
            """Get path from start to pos."""
            path = [pos]
            while path[-1] != start:
                path.append(parents[path[-1]])
            return path[::-1]

        # Initialize
        queue = [start]
        visited = np.zeros_like(obstacles)
        parents = {}

        # Search
        while queue:
            pos = queue.pop(0)
            if pos == end:
                return get_path(parents, pos)
            visited[pos] = 1
            for neighbor in get_neighbors(pos):
                if not visited[neighbor] and not obstacles[neighbor]:
                    queue.append(neighbor)
                    parents[neighbor] = pos

        # No path found
        return []

    @staticmethod
    def find(grid: Int[np.ndarray, "width height channels"], obj: list[int]) -> tuple[int, int]:
        """Find the position of obj in grid, by value."""
        match = np.all(grid == obj, axis=-1)
        matching_pos = np.argwhere(match)
        return tuple(matching_pos[0])

    @classmethod
    def random_action(cls) -> int:
        """Get a random action."""
        print("Random!")
        return choice(list(environments.GridEnv.Actions))

    @classmethod
    def direction_to(cls, start: tuple[int, int], end: tuple[int, int]) -> int:
        """Get direction from start to end purely based on the coordinates."""

        dx = end[0] - start[0]
        dy = end[1] - start[1]

        if dx == 0 and dy == 0:
            return cls.random_action()
        elif abs(dx) == abs(dy):
            vertical = choice([True, False])
        else:
            vertical = abs(dx) < abs(dy)

        if vertical:
            if dy > 0:
                return environments.GridEnv.Actions.DOWN
            else:
                return environments.GridEnv.Actions.UP
        else:
            if dx > 0:
                return environments.GridEnv.Actions.RIGHT
            else:
                return environments.GridEnv.Actions.LEFT

    @abc.abstractmethod
    def _predict(self, obs):
        """Predict baseline."""

    def predict(self, obs, deterministic=False):
        return self._predict(obs), None

