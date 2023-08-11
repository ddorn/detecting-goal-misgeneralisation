from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from random import choice, sample
from typing import SupportsFloat, Any, Literal

import gymnasium as gym
import numpy as np
import plotly.express as px
import pygame
import pygame.gfxdraw
from gymnasium.core import ActType, ObsType, RenderFrame, Wrapper
from pygame import Color

from utils import Distribution, Pos, sample_distribution, uniform_distribution
import wrappers as my_wrappers

__all__ = [
    "Cell",
    "GridEnv",
    "ThreeGoalsEnv",
    "RandomGoalEnv",
]


@dataclass
class Cell:
    label: str
    color: str | tuple[int, int, int] | None
    terminates: bool = False
    reward: float = 0
    can_overlap: bool = True
    manual: bool = False


class GridEnv(gym.Env[gym.spaces.MultiDiscrete, gym.spaces.Discrete]):
    class Actions(IntEnum):
        RIGHT = 0
        DOWN = 1
        LEFT = 2
        UP = 3

    DIR_TO_VEC = [
        # Pointing right (positive X)
        np.array((1, 0)),
        # Down (positive Y)
        np.array((0, 1)),
        # Pointing left (negative X)
        np.array((-1, 0)),
        # Up (negative Y)
        np.array((0, -1)),
    ]

    EMPTY_CELL = Cell(".", None)
    # AGENT_CELL = Cell("A", "#E91E63")
    AGENT_CELL = Cell("A", "#FFFFFF")
    WALL_CELL = Cell("#", "#607D8B", can_overlap=False)
    GOAL_CELL = Cell("G", "#8BC34A", True, 1)
    LAVA_CELL = Cell("L", "#FF5722", True, -1)

    BASE_CELLS = [EMPTY_CELL, AGENT_CELL]
    ALL_CELLS = BASE_CELLS

    def __init__(
            self,
            agent_start: Distribution[Pos] | None,
            width: int,
            height: int,
            max_steps: int | None = None,
    ):
        self.width = width
        self.height = height
        self.agent_start = agent_start
        self.max_steps = max_steps if max_steps is not None else width * height

        self.steps = 0
        self.agent_pos: tuple[int, int] = (-1, -1)
        assert self.ALL_CELLS[0] is self.EMPTY_CELL, self.ALL_CELLS
        assert self.ALL_CELLS[1] is self.AGENT_CELL, self.ALL_CELLS
        self.grid = np.zeros((self.width, self.height), dtype="int8")
        self.make_grid()

        self.action_space = gym.spaces.Discrete(len(self.Actions))
        self.observation_space = gym.spaces.MultiDiscrete([[len(self.ALL_CELLS)] * width] * height)
        self.reward_range = -1, 1

        self.last_reward = None  # Used for rendering

        self.step_reward = -1 / self.max_steps

    def __repr__(self):
        out = "\n".join("".join(self[x, y].label for x in range(self.width))
                        for y in range(self.height))
        return f"<{self.__class__.__name__}:\n{out}>"

    __str__ = __repr__

    def __getitem__(self, item: tuple[int, int]):
        obj = self.grid[item[0], item[1]]
        return self.ALL_CELLS[obj]

    def __setitem__(self, item: tuple[int, int], value: Cell):
        self.grid[item[0], item[1]] = self.ALL_CELLS.index(value)

    def make_grid(self):
        self.grid.fill(0)
        self.place_agent(self.agent_start)

    def place_agent(self, pos_distribution: Distribution[Pos] | None = None):
        # Remove the agent from the grid, if previously present
        if self.grid[self.agent_pos] == 1:
            self.grid[self.agent_pos] = 0  # empty cell

        self.agent_pos = self.place_obj(self.AGENT_CELL, pos_distribution)

    def place_obj(self, obj: Cell, pos_distribution: Distribution[Pos] | None = None) -> tuple[int, int]:
        """Place an object on an empty cell according to the given distribution."""
        if isinstance(pos_distribution, tuple):
            pos = pos_distribution

        # Sample a random position that is empty
        elif pos_distribution is None:
            while True:
                pos = self.np_random.choice(self.width), self.np_random.choice(self.height)
                if self.grid[pos] == 0:
                    break
        else:
            pos = sample_distribution({p: w for p, w in pos_distribution.items() if self[p] is self.EMPTY_CELL})

        self[pos] = obj
        return pos

    def reset(
            self,
            *,
            seed: int | None = None,
            options: dict[str, Any] | None = None,
    ) -> tuple[ObsType, dict[str, Any]]:
        super().reset(seed=seed, options=options)
        self.steps = 0
        self.agent_pos = -1, -1
        self.make_grid()
        self.last_reward = None
        return self.grid, {}

    def step(self, action: ActType) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        new_pos = tuple(self.agent_pos + self.DIR_TO_VEC[action])

        reward = self.step_reward
        terminated = False
        truncated = False
        can_move = True

        # If out of bounds, don't move
        if not (0 <= new_pos[0] < self.width and 0 <= new_pos[1] < self.height):
            can_move = False

        # Handle empty cells
        elif self.grid[new_pos] == 0:
            pass

        # Handle objects
        else:
            obj = self[new_pos]
            if obj.manual:
                can_move, reward, terminated = self.handle_object(obj)
            else:
                can_move = obj.can_overlap
                reward = obj.reward
                terminated = obj.terminates

        if can_move:
            self[self.agent_pos] = self.EMPTY_CELL
            self.agent_pos = new_pos
            self[self.agent_pos] = self.AGENT_CELL

        self.steps += 1
        if self.steps >= self.max_steps:
            truncated = True

        self.last_reward = reward
        return self.grid, reward, terminated, truncated, {}

    def handle_object(self, obj: Cell) -> tuple[bool, float, bool]:
        """Returns (can_move, reward, terminated)"""

    def render(self, resolution: int = 32, plot: bool = False) -> RenderFrame:
        # Draw borders
        full_img = pygame.Surface(((self.width + 1) * resolution, (self.height + 1) * resolution))
        full_img.fill(self.frame_color())

        img = full_img.subsurface(
            resolution // 2,
            resolution // 2,
            self.width * resolution,
            self.height * resolution,
            )

        # Draw each cell
        for x in range(self.width):
            for y in range(self.height):
                rect = (x * resolution, y * resolution, resolution, resolution)
                # Draw checkered background
                color = "#EBE7E5" if (x + y) % 2 else "#D6D2CF"
                img.fill(color, rect)

                # Draw the object
                if self[x, y] is self.AGENT_CELL:
                    cx = int((x + 0.5) * resolution) + 1
                    cy = int((y + 0.5) * resolution) + 1
                    radius = int(resolution / 3)
                    points = [
                        (cx + radius, cy),
                        (cx, cy + radius),
                        (cx - radius, cy),
                        (cx, cy - radius),
                    ]
                    # pygame.draw.polygon(img, color, points)
                    pygame.draw.circle(img, 'black', (cx, cy), radius)

                elif self[x, y].color is not None:
                    img.fill(self[x, y].color, rect)

        self.render_extra(img, resolution)
        txt = self.render_caption()
        if txt:
            # noinspection PyTypeChecker
            s = pygame.font.SysFont(None, int(resolution / 1.5)).render(txt, True, "#FFFFFF")
            full_img.blit(s, s.get_rect(bottomleft=full_img.get_rect().bottomleft))

        # Convert the image to numpy array
        array = np.array(pygame.surfarray.array3d(full_img))
        # Swap x and y-axis (numpy uses a different coordinate system)
        array = np.transpose(array, (1, 0, 2))

        if plot:
            px.imshow(array).show()

        return array

    def render_caption(self) -> str:
        if self.last_reward is not None:
            return f"{self.last_reward:.2f}"

    def render_extra(self, img: pygame.Surface, resolution: int) -> None:
        pass

    def frame_color(self) -> Color:
        if self.last_reward is None:
            reward = 0
        else:
            reward = self.last_reward

        grey = Color("#37474F")
        red = Color("#FF5722")
        green = Color("#4CAF50")

        if reward < 0:
            assert self.reward_range[0] < 0
            return grey.lerp(red, reward / self.reward_range[0])
        else:
            assert self.reward_range[1] > 0
            return grey.lerp(green, reward / self.reward_range[1])


class RandomGoalEnv(GridEnv):
    ALL_CELLS = GridEnv.ALL_CELLS + [GridEnv.GOAL_CELL]

    def __init__(self, size: int = 5, br_freq: float | None = None):
        self.br_freq = br_freq

        self.goal_distribution = uniform_distribution((size, size))
        if br_freq is not None:
            # There are (env_size)**2-1 other positions
            self.goal_distribution[size - 1, size - 1] = (br_freq / (1 - br_freq) * (size ** 2 - 1))

        super().__init__(
            agent_start=None,
            width=size,
            height=size,
        )

    def make_grid(self):
        super().make_grid()
        self.goal_pos = self.place_obj(self.GOAL_CELL, self.goal_distribution)



class ThreeGoalsEnv(GridEnv):
    GOAL_RED = Cell("r", "#FF0000", manual=True)
    GOAL_GREEN = Cell("g", "#00FF00", manual=True)
    GOAL_BLUE = Cell("b", "#0000FF", manual=True)

    # GOAL_RED = Cell("r", "#F44336", manual=True)
    # GOAL_BLUE = Cell("b", "#2196F3", manual=True)
    # GOAL_GREEN = Cell("g", "#4CAF50", manual=True)

    GOAL_CELLS = [GOAL_RED, GOAL_GREEN, GOAL_BLUE]
    ALL_CELLS = GridEnv.ALL_CELLS + GOAL_CELLS

    def __init__(self,
                 size: int = 4,
                 *,
                 true_goal: Literal['red', 'green', 'blue'] | None = None,
                 agent_pos: Distribution[Pos] | None = None,
                 red_pos: Distribution[Pos] | None = None,
                 green_pos: Distribution[Pos] | None = None,
                 blue_pos: Distribution[Pos] | None = None,
                 ):
        self.red_pos_dist = red_pos
        self.green_pos_dist = green_pos
        self.blue_pos_dist = blue_pos

        self.true_goal_init = true_goal
        self.true_goal_idx = -42
        self.goal_positions = [(-1, -1)] * 3

        super().__init__(agent_pos, size, size)

    @property
    def true_goal(self) -> Cell:
        return self.GOAL_CELLS[self.true_goal_idx]

    def new_goal(self) -> int:
        if self.true_goal_init is None:
            return self.np_random.choice(len(self.GOAL_CELLS))
        elif self.true_goal_init == "red":
            return 0
        elif self.true_goal_init == "green":
            return 1
        elif self.true_goal_init == "blue":
            return 2
        else:
            raise ValueError(f"Invalid true_goal_init: {self.true_goal_init}")

    def make_grid(self):
        super().make_grid()
        goal_distributions = [self.red_pos_dist, self.green_pos_dist, self.blue_pos_dist]
        self.goal_positions = [
            self.place_obj(goal, dist)
            for goal, dist in zip(self.GOAL_CELLS, goal_distributions)
        ]
        self.true_goal_idx = self.new_goal()

    def handle_object(self, obj: Cell) -> tuple[bool, float, bool]:
        if obj is self.true_goal:
            return True, 1, True
        else:
            return True, 0, True

    def render_extra(self, img: pygame.Surface, resolution: int):
        # Add a star on the true goal
        x, y = self.goal_positions[self.true_goal_idx]
        cx = int((x + 0.5) * resolution) - 1  # just for prettiness
        cy = int((y + 0.5) * resolution) + 1
        radius = int(resolution / 3)
        # 5 points star
        angle = np.pi * 4 / 5
        shift = -np.pi / 2  # rotate 90 degrees, so that the star is pointing up
        points = [(cx + radius * np.cos(angle * i + shift), cy + radius * np.sin(angle * i + shift))
                  for i in range(5)]

        pygame.gfxdraw.aapolygon(img, points, (255, 255, 255))
        pygame.gfxdraw.filled_polygon(img, points, (255, 255, 255))

    @classmethod
    def constant(cls, size=4, true_goal: Distribution[str] = None):
        """Return an environment that is always the same, even after reset."""
        true = sample_distribution(true_goal, choice(["red", "green", "blue"]))
        positions = [(x, y) for x in range(size) for y in range(size)]
        agent, red, green, blue = sample(positions, k=4)
        # noinspection PyTypeChecker
        return cls(size, true_goal=true, agent_pos=agent, red_pos=red, green_pos=green, blue_pos=blue)

    @classmethod
    def interesting(cls, size: int = 4, n_random: int = 3, wrappers: list[Wrapper] | None = None) -> list[ThreeGoalsEnv]:
        agent_pos = (0, 0)
        red_green_blue = [
            [(0, 1), (1, 0), (1, 1)],
            [(0, 1), (1, 1), (0, 2)],
            [(0, 2), (1, 2), (1, 3)],
            [(0, size - 1), (size - 1, 0), (size - 1, size - 1)],
            [(0, 1), (0, 2), (0, 3)],
        ]
        envs = [
                   cls(size, true_goal="blue", agent_pos=agent_pos, red_pos=red_pos, green_pos=green_pos,
                       blue_pos=blue_pos)
                   for red_pos, green_pos, blue_pos in red_green_blue
               ] + [
                   cls(size) for _ in range(n_random)
               ]
        if wrappers is None:
            wrappers = [my_wrappers.FlatOneHotWrapper, my_wrappers.AddTrueGoalWrapper]
        for wrapper in wrappers:
            envs = [wrapper(env) for env in envs]
        return envs

