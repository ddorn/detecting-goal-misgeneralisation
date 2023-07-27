from __future__ import annotations

from dataclasses import dataclass, field
from typing import TypeVar, Callable, Literal, Union, SupportsFloat, Any

import einops
import gymnasium as gym
import numpy as np
import plotly.express as px
from gymnasium import ActionWrapper, ObservationWrapper
from gymnasium.core import WrapperObsType, ActType, ObsType
from minigrid.core.constants import DIR_TO_VEC
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
        can_turn: bool = True,
        **kwargs,
    ):
        """
        A simple square environment with a goal square and an agent. The agent can turn left, turn right, or move forward.

        Args:
            size: The size of the grid. The outer walls are included in this size.
            goal_pos: The position of the goal square. If a dictionary is given, the keys are the possible positions
                and the values are the weights of each option. If None is given, the goal square is placed at a random
                position with equal probability. Otherwise, the goal square is at the given position.
            agent_start_pos: The initial position of the agent. The same rules apply as for goal_pos.
            agent_start_dir: The initial direction of the agent (0 = right, 1 = down, 2 = left, 3 = up). The same rules
                apply as for goal_pos.
            max_steps: The maximum number of steps the agent can take before the episode is terminated. If None is
                given, the maximum number of steps is 4 * (size - 2) ** 2.
            can_turn: If true, controls are left, right, forward. If false, controls are up, right, down, left.
            **kwargs: Passed to MiniGridEnv.__init__.
        """

        self.agent_start_pos = agent_start_pos
        self.agent_start_dir = agent_start_dir
        self.goal_pos_distribution = goal_pos
        self.can_turn = can_turn

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
        if self.goal_pos_distribution is None:
            self.goal_pos = self.place_obj(Goal())
        else:
            self.goal_pos = x, y = sample_distribution(self.goal_pos_distribution)
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

    def step(self, action: ActType) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        last_dist_to_goal = self.dist_to_goal()

        if self.can_turn:
            assert action in (0, 1, 2)
            obs, reward, terminated, truncated, info = super().step(action)
        else:
            assert action in (0, 1, 2, 3)
            # We start by orienting the agent in the direction it wants to go
            self.agent_dir = action
            # Then we always move forward
            obs, reward, terminated, truncated, info = super().step(self.actions.forward)

        # Reward when the agent gets closer to the goal
        dist_to_goal = self.dist_to_goal()
        # reward += last_dist_to_goal - dist_to_goal

        return obs, reward, terminated, truncated, info

    def dist_to_goal(self):
        """Manhattan distance from the agent to the goal."""
        return abs(self.agent_pos[0] - self.goal_pos[0]) + abs(self.agent_pos[1] - self.goal_pos[1])
    #
    # def gen_obs(self):
    #     # This is an optimisation that saved 20% of the runtime
    #     # Indeed the observation generated by MiniGridEnv is never used
    #     # As we always wrap it into FullyObsWrapper which creates its own 'image'
    #     # + we discard the 'direction' and 'mission'.
    #     return {}

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
    MAPPING_NO_TURN = [2, 10, 1, 8]  # wall, player, empty, goal

    def __init__(self, env: gym.Env, remove_border_walls: bool = True, can_turn: bool = True):
        super().__init__(env)
        self.can_turn = can_turn
        self.mapping = self.MAPPING if can_turn else self.MAPPING_NO_TURN
        self.dim = len(self.mapping)
        self.remove_border_walls = remove_border_walls
        # noinspection PyUnresolvedReferences
        w, h = env.observation_space["image"].shape[:2]
        if remove_border_walls:
            w -= 2
            h -= 2
        self.observation_space = gym.spaces.MultiBinary((w, h, self.dim))

    def observation(self, observation: WrapperObsType):
        img = observation["image"]
        out = np.zeros((img.shape[0], img.shape[1], self.dim), dtype=bool)

        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                if self.can_turn:
                    # Take (type, color, extra)
                    obs = tuple(img[i, j])
                else:
                    # We only care about the type of object, not the direction (nor color)
                    obs = img[i, j, 0]

                try:
                    out[i, j, self.mapping.index(obs)] = 1
                except ValueError:
                    print(f"Unknown object: {img[i, j]}")
                    raise

        if self.remove_border_walls:
            out = out[1:-1, 1:-1, :]
        return out


def wrap_env(env: gym.Env, can_turn: bool = False):
    """Returns a wrapped environment with the following wrappers:
    - Restrict the action space to the first 3 actions (turn left, turn right, move forward)
    - Convert the observation to a fully observable representation
    - Observations are returned as one-hot vectors (env_size, env_size, dim)
    """

    if can_turn:
        actions = [0, 1, 2]
    else:
        actions = [0, 1, 2, 3]

    env = ActionSubsetWrapper(env, actions)
    env = FullyObsWrapper(env)
    # env = OneHotPartialObsWrapper(env)
    env = OneHotFullObsWrapper(env, can_turn=can_turn)
    # env = ImgObsWrapper(env)
    return env


def random_goal_env(size: int = 5, can_turn: bool = True):
    """An SimpleEnv with a random goal position and random agent position."""
    return wrap_env(
        SimpleEnv(
            size=size,
            goal_pos=None,
            agent_start_pos=None,
            render_mode="rgb_array",
            can_turn=can_turn,
        ),
        can_turn=can_turn,
    )


def bottom_right_env(size: int = 5, can_turn: bool = True):
    """An SimpleEnv with the goal position in the bottom right corner and random agent position."""
    return wrap_env(
        SimpleEnv(
            size=size,
            goal_pos=(size - 2, size - 2),
            agent_start_pos=None,
            render_mode="rgb_array",
            can_turn=can_turn,
        ),
        can_turn=can_turn,
    )


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
            if self.can_turn:
                if map[i, j] in (
                        1,
                        2,
                        3,
                        4,
                ):  # Those numbers refer to OneHotFullObsWrapper.MAPPING
                    agent_pos = (i, j)
                    break
            else:
                if (map[i, j] == 1):  # This number refers to OneHotFullObsWrapper.MAPPING_NO_TURN
                    agent_pos = (i, j)
                    break
        else:
            raise ValueError("Could not find the agent position")

        if self.can_turn:
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
        can_turn: bool = True,
        **info,
    ):
        # Should be enough to reach the goal
        # Higher values make the evaluation much slower,
        # because of trajectories where the agent is stuck.
        max_len = env_size * 4
        br_success_rate = eval_agent(policy,
                                     bottom_right_env(env_size, can_turn),
                                     episodes,
                                     episode_len=max_len)
        success_rate = eval_agent(policy,
                                  random_goal_env(env_size, can_turn),
                                  episodes,
                                  episode_len=max_len)
        br_freq = eval_agent(
            policy,
            random_goal_env(env_size, can_turn),
            episodes,
            episode_len=max_len,
            end_condition=lambda locals_: locals_["env"].agent_pos == (env_size - 2, env_size - 2),
        )
        return cls(br_success_rate, success_rate, br_freq, env_size, info)


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
