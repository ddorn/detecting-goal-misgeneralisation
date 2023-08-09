from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import TypeVar, Callable, Literal, Union, SupportsFloat, Any

import einops
import gymnasium as gym
import numpy as np
import plotly.express as px
import pygame
import pygame.gfxdraw
import wandb
from gymnasium import ObservationWrapper
from gymnasium.core import WrapperObsType, ActType, ObsType, RenderFrame
from pygame import Color
from stable_baselines3 import PPO

Pos = tuple[int, int]
T = TypeVar("T")
Distribution = Union[dict[T, float], T]

pygame.font.init()


def uniform_distribution(
        bottom_right: tuple[int, int], top_left: tuple[int, int] = (0, 0)) -> dict[Pos, float]:
    """Returns a uniform distribution over the given rectangle. Bottom and right bounds are not inclusive."""
    return {
        (x, y): 1
        for x in range(top_left[0], bottom_right[0])
        for y in range(top_left[1], bottom_right[1])
    }


_sentinel = object()


def sample_distribution(distribution: Distribution[T] | None, default: T = _sentinel) -> T:
    """Sample a value from the given distribution."""
    if isinstance(distribution, dict):
        options = list(distribution.keys())
        probability_sum = sum(distribution.values())
        probabilities = np.array(list(distribution.values())) / probability_sum
        index = np.random.choice(len(options), p=probabilities)
        return options[index]
    elif distribution is None:
        if default is _sentinel:
            raise ValueError("Distribution is None and no default value was given")
        return default
    else:
        return distribution


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
    AGENT_CELL = Cell("A", "#E91E63")
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
        self.grid: np.ndarray  # Defined in make_grid
        self.make_grid()

        self.action_space = gym.spaces.Discrete(len(self.Actions))
        # self.observation_space = gym.spaces.MultiBinary((width, height, len(self.CellTypes)))
        self.observation_space = gym.spaces.MultiDiscrete([[len(self.ALL_CELLS)] * width] * height)
        self.reward_range = -1, 1

        self.render_mode = "rgb_array"
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
        self.grid = np.zeros((self.width, self.height), dtype="int8")
        self.place_agent(self.agent_start)

    def place_agent(self, pos_distribution: Distribution[Pos] | None = None):
        # Remove the agent from the grid
        if self.agent_pos[0] >= 0:
            self[self.agent_pos] = self.EMPTY_CELL

        self.agent_pos = self.place_obj(self.AGENT_CELL, pos_distribution)

    def place_obj(self, obj: Cell, pos_distribution: Distribution[Pos] | None = None):
        """Place an object on an empty cell according to the given distribution."""
        if isinstance(pos_distribution, tuple):
            pos = pos_distribution

        # Sample a random position that is empty
        else:
            if pos_distribution is None:
                pos_distribution = uniform_distribution((self.width, self.height))
            pos = sample_distribution({p: w for p, w in pos_distribution.items() if self[p] is self.EMPTY_CELL})

        assert self[pos] is self.EMPTY_CELL, f"Position {pos} is not empty: {self[pos]}"
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
                    pygame.draw.circle(img, self[x, y].color, (cx, cy), radius)

                elif self[x, y].color is not None:
                    img.fill(self[x, y].color, rect)

        self.render_extra(img, resolution)
        txt = self.render_caption()
        if txt:
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


class FlatOneHotWrapper(ObservationWrapper):

    def __init__(self, env: GridEnv):
        super().__init__(env)
        self.n_cells = len(self.ALL_CELLS)
        obs = env.observation_space
        assert isinstance(obs, gym.spaces.MultiDiscrete), f"Expected MultiDiscrete, got {obs}"
        self.observation_space = gym.spaces.MultiBinary((self.width * self.height * self.n_cells,))

    def observation(self, obs: WrapperObsType) -> ObsType:
        w, h = obs.shape
        one_hot = np.zeros((w, h, self.n_cells), dtype=bool)
        for x in range(w):
            for y in range(h):
                one_hot[x, y, obs[x, y]] = True
        return one_hot.flatten()


def random_goal_env(size: int = 5, br_freq: float | None = None) -> gym.Env:
    """An environment with a random goal position and random agent position."""
    return FlatOneHotWrapper(RandomGoalEnv(size, br_freq))


@dataclass
class Trajectory:
    images: np.ndarray  # (time, height, width, channels)
    reward: float
    ended: Literal["truncated", "terminated", "condition"]

    def __len__(self):
        return self.images.shape[0]

    def __eq__(self, other):
        return np.allclose(self.images, other.images) and self.reward == other.reward and self.ended == other.ended

    def image(self, pad_to: int | None = None) -> np.ndarray:
        """Return all images concatenated along the time axis."""

        empty = np.zeros_like(self.images[0])
        if pad_to is None:
            pad_to = len(self.images)

        image = einops.rearrange(
            self.images + [empty] * (pad_to - len(self.images)),
            "time h w c -> h (time w) c",
        )
        return image

    @classmethod
    def from_policy(
            cls,
            policy: PPO,
            env: gym.Env,
            max_len: int = 10,
            end_condition: Callable[[dict], bool] | None = None,
            no_images: bool = False,
    ) -> Trajectory:
        """Run the policy in the environment and return the trajectory.

        Args:
            policy: The policy to use.
            env: The environment to run the policy in.
            max_len: The maximum number of steps to run the policy for.
            end_condition: A function that takes the locals() dict and returns True if the trajectory should end.
            no_images: If true, only the initial is returned.
        """
        assert env.render_mode == "rgb_array"

        def mk_output(ended: Literal["truncated", "terminated", "condition"]) -> Trajectory:
            return cls(np.stack(images), total_reward, ended)

        obs, _info = env.reset()
        images = [env.render()]
        total_reward = 0
        for step in range(max_len):
            action, _states = policy.predict(obs, deterministic=True)

            try:
                obs, reward, terminated, truncated, info = env.step(action)
            except AssertionError:
                print(env, env.observation_space, env.action_space)
                raise

            total_reward += reward
            if not no_images:
                images.append(env.render())
            if end_condition is not None and end_condition(locals()):
                return mk_output("condition")
            if terminated:
                return mk_output("terminated")
            if truncated:
                return mk_output("truncated")

        return mk_output("truncated")


class BottomRightAgent:
    """Baseline agent that always goes to the bottom right corner."""

    def __init__(self, can_turn: bool = True):
        self.can_turn = can_turn

    def predict(self, obs, deterministic=True):
        assert len(obs.shape) == 3
        size = obs.shape[0]

        map = np.argmax(obs, axis=-1)  # (size, size)

        # Find where the agent is
        for i, j in np.ndindex(size, size):
            # Those numbers refer to OneHotFullObsWrapper.MAPPING
            # and OneHotFullObsWrapper.MAPPING_NO_TURN
            if map[i, j] >= 3:
                agent_pos = (i, j)
                break
        else:
            raise ValueError("Could not find the agent position")

        if self.can_turn:
            direction = map[agent_pos] - 3  # between 0 and 3: right, down, left, up

            LEFT = 0, None
            RIGHT = 1, None
            FORWARD = 2, None

            if direction == 0:
                if agent_pos[0] == size - 1:
                    return RIGHT
                return FORWARD
            elif direction == 1:
                if agent_pos[1] == size - 1:
                    return LEFT
                return FORWARD
            elif direction == 2:
                return LEFT
            elif direction == 3:
                return RIGHT
            else:
                raise ValueError(f"Invalid direction: {direction}")

        else:
            # Numbers from minigrid.DIR_TO_VEC
            if agent_pos[0] == size - 1:  # on right wall
                return 1, None  # go down
            return 0, None  # go right


@dataclass
class Perfs:
    br_env: float
    general_env: float
    general_br_freq: float
    env_size: int
    info: dict[str, float] = field(default_factory=dict)

    @classmethod
    def from_agent(
            cls,
            policy: PPO,
            episodes: int = 100,
            env_size: int = 5,
            **info,
    ):
        # Should be enough to reach the goal
        # Higher values make the evaluation much slower,
        # because of trajectories where the agent is stuck.
        max_len = env_size * 4
        br_success_rate = eval_agent(policy,
                                     random_goal_env(env_size, 1.0),
                                     episodes,
                                     episode_len=max_len)
        success_rate = eval_agent(policy, random_goal_env(env_size), episodes, episode_len=max_len)
        br_freq = eval_agent(
            policy,
            random_goal_env(env_size, 1.0),
            episodes,
            episode_len=max_len,
            end_condition=lambda locals_: locals_["env"].agent_pos == (env_size - 1, env_size - 1),
        )
        return cls(br_success_rate, success_rate, br_freq, env_size, info)


def show_behavior(
        policy: PPO,
        env: gym.Env | list[gym.Env] | list[Trajectory],
        n_trajectories: int = 10,
        max_len: int = 10,
        add_to_wandb: bool = False,
        plot: bool = True,
        **plotly_kwargs,
):
    if isinstance(env, list) and isinstance(env[0], Trajectory):
        trajectories = [
            traj.images for traj in env
        ]
    elif isinstance(env, list):
        trajectories = [
            Trajectory.from_policy(policy, env_, max_len=max_len).images for env_ in env
        ]
    else:
        trajectories = [
            Trajectory.from_policy(policy, env, max_len=max_len).images for _ in range(n_trajectories)
        ]

    actual_max_len = max(len(traj) for traj in trajectories)
    for i, traj in enumerate(trajectories):
        trajectories[i] = np.pad(traj, ((0, actual_max_len - len(traj)), (0, 0), (0, 0), (0, 0)))

    trajectories = np.stack(trajectories)

    imgs = einops.rearrange(trajectories, "traj step h w c -> (traj h) (step w) c")

    if add_to_wandb:
        wandb.log({f"behavior": wandb.Image(imgs)})

    if plot:
        plotly_kwargs.setdefault("height", imgs.shape[0] // 2)
        plotly_kwargs.setdefault("width", imgs.shape[1] // 2)
        px.imshow(imgs, **plotly_kwargs).show()


def eval_agent(
        policy: PPO,
        env: gym.Env = random_goal_env(),
        episodes: int = 100,
        episode_len: int = 100,
        plot: bool = False,
        end_condition: Callable[[dict], bool] | None = None,
) -> float:
    nb_success = 0
    fails = []
    success_imgs = []
    for _ in range(episodes):
        trajectory = Trajectory.from_policy(
            policy,
            env,
            max_len=episode_len,
            end_condition=end_condition,
            no_images=True,
        )

        if end_condition is None:
            success = trajectory.ended == "terminated"
        else:
            success = trajectory.ended == "condition"

        if success:
            nb_success += 1
            add_to = success_imgs
        else:
            add_to = fails

        # If the initial position is not in the list, add it
        if all(np.any(img != trajectory.images[0]) for img in add_to):
            add_to.append(trajectory.images[0])

    # show fails
    if plot and fails and success_imgs:
        print(f"Success rate: {nb_success / episodes:.2%}")
        for imgs, title in [(fails, "failed"), (success_imgs, "succeeded")]:
            prop = len(imgs) / (len(fails) + len(success_imgs))
            # Max 100 images, and pad to multiple of 10
            imgs = imgs[:100]
            imgs += [np.zeros_like(imgs[0])] * (10 - len(imgs) % 10)

            img = einops.rearrange(imgs, "(row col) h w c -> (row h) (col w) c", row=10)
            px.imshow(
                img,
                title=f"Positions where the agent {title} to reach the goal. {prop:.2%}",
            ).show()

    return nb_success / episodes
