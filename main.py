# %%
from __future__ import annotations

from time import sleep
from typing import TypeVar

import einops
import gymnasium as gym
import numpy as np
import plotly.express as px
from gymnasium import ActionWrapper, ObservationWrapper
from gymnasium.core import WrapperObsType
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Goal
from minigrid.minigrid_env import MiniGridEnv
from minigrid.wrappers import FullyObsWrapper
from rich import print as pprint
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

# %%
Pos = tuple[int, int]
T = TypeVar("T")
Distribution = dict[T, float] | T


def uniform_distribution(bottom_right: tuple[int, int], top_left: tuple[int, int] = (1, 1)) -> dict[Pos, float]:
    """Returns a uniform distribution over the given rectangle. Bottom and right bounds are not inclusive."""
    return {(x, y): 1 for x in range(top_left[0], bottom_right[0]) for y in range(top_left[1], bottom_right[1])}


def sample_distribution(distribution: Distribution[T]) -> T:
    """Sample a distribution.

    If the distribution is a dictionary, the keys are the possible values and the values are the weights
    of each option. Otherwise, the distribution is assumed to be a single value with probability 1.
    """
    if isinstance(distribution, dict):
        options = list(distribution.keys())
        probability_sum = sum(distribution.values())
        probabilities = np.array(list(distribution.values())) / probability_sum
        index = np.random.choice(len(options), p=probabilities)
        return options[index]
    else:
        return distribution


class SimpleEnv(MiniGridEnv):

    def __init__(self,
                 size=5,
                 goal_pos: Distribution[Pos] | None = (-2, -2),
                 agent_start_pos: Distribution[Pos] | None = (1, 1),
                 agent_start_dir: Distribution[int] = 0,
                 max_steps: int | None = None, **kwargs):
        """
        A simple square environment with a goal square and an agent. The agent can turn left, turn right, or move forward.

        Args:
            size: The size of the grid. The outer walls are included in this size.
            goal_pos: The position of the goal square. If a dictionary is given, the keys are the possible positions
                and the values are the weights of each option. If None is given, the goal square is placed at a random
                position with equal probability. Otherwise, the goal square is placed at the given position.
            agent_start_pos: The initial position of the agent. The same rules apply as for goal_pos.
            agent_start_dir: The initial direction of the agent (0 = right, 1 = down, 2 = left, 3 = up). The same rules
                apply as for goal_pos.
            max_steps: The maximum number of steps the agent can take before the episode is terminated. If None is
                given, the maximum number of steps is 4 * (size - 2) ** 2.
            **kwargs: Passed to MiniGridEnv.__init__.
        """

        self.agent_start_pos = agent_start_pos
        self.agent_start_dir = agent_start_dir
        self.goal_pos = goal_pos

        if max_steps is None:
            max_steps = 4 * (size - 2) ** 2

        super().__init__(
            MissionSpace(mission_func=lambda: "get to the green goal square"),
            grid_size=size,
            see_through_walls=True,
            max_steps=max_steps,
            **kwargs
        )

    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Place a goal square
        if self.goal_pos is None:
            self.place_obj(Goal())
        else:
            x, y = sample_distribution(self.goal_pos)
            self.put_obj(Goal(), x % width, y % height)

        # Place the agent
        if self.agent_start_pos is not None:
            self.agent_pos = sample_distribution(self.agent_start_pos)
            self.agent_dir = sample_distribution(self.agent_start_dir)
        else:
            self.place_agent()


class ActionSubsetWrapper(ActionWrapper):
    def __init__(self, env: gym.Env, action_subset: list[int]):
        """
        Action wrapper that restricts the action space to a subset of the original discrete action space.
        """
        assert isinstance(env.action_space, gym.spaces.Discrete)
        assert isinstance(action_subset, list)
        assert all(isinstance(action, int) for action in action_subset)
        assert isinstance(env.action_space, gym.spaces.Discrete)
        assert all(0 <= action < env.action_space.n for action in action_subset)

        super().__init__(env)
        self.action_subset = action_subset
        self.action_space = gym.spaces.Discrete(len(action_subset))

    def action(self, action: int) -> int:
        return self.action_subset[action]

    def reverse_action(self, action: int) -> int:
        return self.action_subset.index(action)


class OneHotFullObsWrapper(ObservationWrapper):
    """Converts observations from SimpleEnv to one-hot vectors."""

    # All the possible objects in the grid
    MAPPING = [
        (2, 5, 0),  # wall
        (10, 0, 0),  # player facing the four directions
        (10, 0, 1),
        (10, 0, 2),
        (10, 0, 3),
        (1, 0, 0),  # empty
        (8, 1, 0),  # goal
    ]

    def __init__(self, env: gym.Env, remove_border_walls: bool = True):
        super().__init__(env)
        self.dim = len(self.MAPPING)
        self.remove_border_walls = remove_border_walls
        w, h = env.observation_space['image'].shape[:2]
        if remove_border_walls:
            w -= 2
            h -= 2
        self.observation_space = gym.spaces.MultiBinary((w, h, self.dim))

    def observation(self, observation: WrapperObsType):
        img = observation['image']
        out = np.zeros((img.shape[0], img.shape[1], self.dim))

        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                try:
                    out[i, j, self.MAPPING.index(tuple(img[i, j]))] = 1
                except ValueError:
                    print(f"Unknown object: {img[i, j]}")
                    raise

        if self.remove_border_walls:
            out = out[1:-1, 1:-1, :]
        return out


def wrap_env(env: gym.Env):
    """Returns a wrapped environment with the following wrappers:
    - Restrict the action space to the first 3 actions (turn left, turn right, move forward)
    - Convert the observation to a fully observable representation
    - Observations are returned as images
    """

    env = ActionSubsetWrapper(env, [0, 1, 2])
    env = FullyObsWrapper(env)
    # env = OneHotPartialObsWrapper(env)
    env = OneHotFullObsWrapper(env)
    # env = ImgObsWrapper(env)
    return env


def make_env(
        size=5,
        goal_pos: Distribution[Pos] | None = (-2, -2),
        agent_start_pos: Distribution[Pos] | None = (1, 1),
        agent_start_dir: Distribution[int] = 0,
        max_steps: int | None = None, **kwargs) -> gym.Env:
    """Utility function to create and wrap a SimpleEnv."""
    env = SimpleEnv(size, goal_pos, agent_start_pos, agent_start_dir, max_steps, **kwargs)
    return wrap_env(env)


def get_trajectory(
        policy: PPO,
        agent_start: tuple[int, int],
        goal: tuple[int, int],
        env_size: int = 5,
        max_len: int = 10,
):
    env = wrap_env(SimpleEnv(env_size, goal, agent_start, render_mode='rgb_array'))
    obs, _info = env.reset()
    images = [env.render()]
    for i in range(max_len):
        action, _states = policy.predict(obs, deterministic=True)

        obs, reward, terminated, truncated, info = env.step(action)
        images.append(env.render())
        if terminated or truncated:
            break

    # images += [np.zeros_like(images[0])] * (max_len - len(images))
    return np.stack(images)

def get_trajectories(
        policy: PPO,
        starts_goals: list[tuple[tuple[int, int], tuple[int, int]]],
        max_len: int = 10,
        plot: bool = True, env_size: int = 5):
    trajectories = []
    for start, goal in starts_goals:
        trajectories.append(get_trajectory(policy, start, goal, env_size=env_size, max_len=max_len))

    actual_max_len = max(len(traj) for traj in trajectories)
    for i, traj in enumerate(trajectories):
        trajectories[i] = np.pad(traj, ((0, actual_max_len - len(traj)), (0, 0), (0, 0), (0, 0)))

    trajectories = np.stack(trajectories)
    if plot:
        imgs = einops.rearrange(trajectories, 'traj step h w c -> (traj h) (step w) c')
        px.imshow(imgs).show()

    return trajectories

def eval_agent(
        policy: PPO,
        env: gym.Env,
        episodes: int = 100,
        episode_len: int = 10,
        plot: bool = True,
) -> float:
    sucesses = 0
    fails = []
    success_imgs = []
    for _ in range(episodes):
        obs, _info = env.reset()
        initial_pos = env.render()
        for _ in range(episode_len):
            action, _states = policy.predict(obs, deterministic=False)
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated:
                sucesses += 1
                if not any(np.all(success_img == initial_pos) for success_img in success_imgs):
                    success_imgs.append(initial_pos)
                break
        else:
            for fail_initial in fails:
                if np.all(fail_initial == initial_pos):
                    break
            else:
                fails.append(initial_pos)

    # show fails
    if fails and success_imgs and plot:
        print(f"Success rate: {sucesses/episodes:.2%}")
        for imgs, title in [(fails, "failed"), (success_imgs, "succeeded")]:
            prop = len(imgs) / (len(fails) + len(success_imgs))
            if len(imgs) > 100:
                imgs = imgs[:100]
            if len(imgs) % 10 != 0:
                imgs += [np.zeros_like(imgs[0])] * (10 - len(imgs) % 10)

            img = einops.rearrange(imgs, "(row col) h w c -> (row h) (col w) c", row=10)
            px.imshow(img, title=f"Positions where the agent {title} to reach the goal. {prop:.2%}").show()

    return sucesses / episodes

