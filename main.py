from __future__ import annotations

from dataclasses import dataclass, field
from typing import TypeVar, Callable, Literal, Union

import einops
import gymnasium as gym
import numpy as np
import plotly.express as px
from gymnasium import ActionWrapper, ObservationWrapper
from gymnasium.core import WrapperObsType
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Goal, Lava
from minigrid.minigrid_env import MiniGridEnv
from minigrid.wrappers import FullyObsWrapper
from stable_baselines3 import PPO

Pos = tuple[int, int]
T = TypeVar("T")
Distribution = Union[dict[T, float], T]


def uniform_distribution(
    bottom_right: tuple[int, int], top_left: tuple[int, int] = (1, 1)) -> dict[Pos, float]:
    """Returns a uniform distribution over the given rectangle. Bottom and right bounds are not inclusive."""
    return {
        (x, y): 1
        for x in range(top_left[0], bottom_right[0])
        for y in range(top_left[1], bottom_right[1])
    }


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

    def __init__(
        self,
        size=5,
        goal_pos: Distribution[Pos] | None = (-2, -2),
        agent_start_pos: Distribution[Pos] | None = (1, 1),
        agent_start_dir: Distribution[int] | None = None,
        max_steps: int | None = None,
        **kwargs,
    ):
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
            max_steps = 4 * (size - 2)**2

        super().__init__(
            MissionSpace(mission_func=lambda: "get to the green goal square"),
            grid_size=size,
            see_through_walls=True,
            max_steps=max_steps,
            **kwargs,
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
        else:
            self.place_agent(rand_dir=False)
        if self.agent_start_dir is None:
            self.agent_dir = self._rand_int(0, 4)
        else:
            self.agent_dir = sample_distribution(self.agent_start_dir)

        self.add_walls()

    def add_walls(self):
        pass


class SimpleEnvWithLava(SimpleEnv):

    def add_walls(self):
        self.place_obj(Lava())


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
        # noinspection PyUnresolvedReferences
        w, h = env.observation_space["image"].shape[:2]
        if remove_border_walls:
            w -= 2
            h -= 2
        self.observation_space = gym.spaces.MultiBinary((w, h, self.dim))

    def observation(self, observation: WrapperObsType):
        img = observation["image"]
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
    max_steps: int | None = None,
    **kwargs,
) -> gym.Env:
    """Utility function to create and wrap a SimpleEnv."""
    env = SimpleEnv(size, goal_pos, agent_start_pos, agent_start_dir, max_steps, **kwargs)
    return wrap_env(env)


def random_goal_env(size: int = 5):
    """An SimpleEnv with a random goal position and random agent position."""
    return wrap_env(
        SimpleEnv(size=size, goal_pos=None, agent_start_pos=None, render_mode="rgb_array"))


def bottom_right_env(size: int = 5):
    """An SimpleEnv with the goal position in the bottom right corner and random agent position."""
    return wrap_env(
        SimpleEnv(
            size=size,
            goal_pos=(size - 2, size - 2),
            agent_start_pos=None,
            render_mode="rgb_array",
        ))


@dataclass
class Trajectory:
    images: np.ndarray  # (time, height, width, channels)
    reward: float
    ended: Literal["truncated", "terminated", "condition"]

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
    ) -> Trajectory:
        """Run the policy in the environment and return the trajectory."""
        assert env.render_mode == "rgb_array"

        obs, _info = env.reset()
        images = [env.render()]
        total_reward = 0
        for step in range(max_len):
            action, _states = policy.predict(obs, deterministic=True)

            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            images.append(env.render())
            if end_condition is not None and end_condition(locals()):
                return cls(np.stack(images), total_reward, "condition")
            if terminated:
                return cls(np.stack(images), total_reward, "terminated")
            if truncated:
                return cls(np.stack(images), total_reward, "truncated")

        return cls(np.stack(images), total_reward, "truncated")


class BottomRightAgent:
    """Baseline agent that always goes to the bottom right corner."""

    def predict(self, obs, deterministic=True):
        assert len(obs.shape) == 3
        size = obs.shape[0]

        map = np.argmax(obs, axis=-1)  # (size, size)

        # Find where the agent is
        for i, j in np.ndindex(size, size):
            if map[i, j] in (1, 2, 3, 4):
                agent_pos = (i, j)
                break

        # noinspection PyUnboundLocalVariable
        direction = map[agent_pos] - 1  # between 0 and 3: right, down, left, up

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


def get_trajectory(
    policy: PPO,
    env: gym.Env,
    max_len: int = 10,
    end_condition: Callable[[dict], bool] | None = None,
) -> Trajectory:
    assert env.render_mode == "rgb_array"

    obs, _info = env.reset()
    images = [env.render()]
    total_reward = 0
    for step in range(max_len):
        action, _states = policy.predict(obs, deterministic=True)

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        images.append(env.render())
        if end_condition is not None and end_condition(locals()):
            return Trajectory(np.stack(images), total_reward, "condition")
        if terminated:
            return Trajectory(np.stack(images), total_reward, "terminated")
        if truncated:
            return Trajectory(np.stack(images), total_reward, "truncated")

    return Trajectory(np.stack(images), total_reward, "truncated")


RANDOM_GOAL_ENV = wrap_env(
    SimpleEnv(size=5, goal_pos=None, agent_start_pos=None, render_mode="rgb_array"))

BR_GOAL_ENV = wrap_env(
    SimpleEnv(size=5, goal_pos=(-2, -2), agent_start_pos=None, render_mode="rgb_array"))


@dataclass
class Perfs:
    br_env: float
    general_env: float
    general_br_freq: float
    info: dict[str, float] = field(default_factory=dict)

    @classmethod
    def from_agent(cls, policy: PPO, episodes: int = 100, env_size: int = 5, **info):
        br_goal_env = wrap_env(
            SimpleEnv(
                size=env_size,
                goal_pos=(-2, -2),
                agent_start_pos=None,
                render_mode="rgb_array",
            ))
        random_goal_env = wrap_env(
            SimpleEnv(
                size=env_size,
                goal_pos=None,
                agent_start_pos=None,
                render_mode="rgb_array",
            ))

        br_success_rate = eval_agent(policy, br_goal_env, episodes)
        success_rate = eval_agent(policy, random_goal_env, episodes)
        br_freq = eval_agent(
            policy,
            random_goal_env,
            episodes,
            end_condition=lambda locals_: locals_["env"].agent_pos == (env_size - 2, env_size - 2),
        )
        return cls(br_success_rate, success_rate, br_freq, info)


def show_behavior(
    policy: PPO,
    env: gym.Env,
    n_trajectories: int = 10,
    max_len: int = 10,
    **plotly_kwargs,
):
    trajectories = [
        Trajectory.from_policy(policy, env, max_len=max_len).images for _ in range(n_trajectories)
    ]

    actual_max_len = max(len(traj) for traj in trajectories)
    for i, traj in enumerate(trajectories):
        trajectories[i] = np.pad(traj, ((0, actual_max_len - len(traj)), (0, 0), (0, 0), (0, 0)))

    trajectories = np.stack(trajectories)

    imgs = einops.rearrange(trajectories, "traj step h w c -> (traj h) (step w) c")
    plotly_kwargs.setdefault("height", imgs.shape[0] // 2)
    plotly_kwargs.setdefault("width", imgs.shape[1] // 2)
    px.imshow(imgs, **plotly_kwargs).show()


def eval_agent(
    policy: PPO,
    env: gym.Env = RANDOM_GOAL_ENV,
    episodes: int = 100,
    episode_len: int = 10,
    plot: bool = False,
    end_condition: Callable[[dict], bool] | None = None,
) -> float:
    nb_success = 0
    fails = []
    success_imgs = []
    for _ in range(episodes):
        trajectory = Trajectory.from_policy(policy,
                                            env,
                                            max_len=episode_len,
                                            end_condition=end_condition)

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
