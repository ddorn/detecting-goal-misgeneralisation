"""
Wrappers for grid environments.
"""

from __future__ import annotations

import math
from typing import Callable

import gymnasium as gym
import numpy as np
import pygame
from gymnasium import ObservationWrapper
from gymnasium.core import WrapperObsType, ObsType
from gymnasium.spaces import MultiBinary, MultiDiscrete

__all__ = [
    "AddSwitch",
    "ColorBlindWrapper",
    "OneHotColorBlindWrapper",
    "AddTrueGoalToObsFlat",
]


class AddSwitch(ObservationWrapper):
    """
    A wrapper that adds a switch to the observation.

    Input: in=Any
    Output: Dict(obs=in, switch=Discrete(n_switches))

    If in is already a dict, it will be instead be updated with the switch.
    """

    def __init__(self, env: gym.Env,
                 n_switches: int,
                 switch_function: Callable[[gym.Env], int],
                 ):
        super().__init__(env)
        self.n_switches = n_switches
        self.switch_function = switch_function

        in_space = env.observation_space
        if isinstance(in_space, gym.spaces.Dict):
            assert "switch" not in in_space.spaces, in_space
            self.observation_space = gym.spaces.Dict({
                **in_space,
                "switch": gym.spaces.Discrete(n_switches),
            })
        else:
            self.observation_space = gym.spaces.Dict({
                "obs": in_space,
                "switch": gym.spaces.Discrete(n_switches),
            })

    def observation(self, obs: ObsType) -> WrapperObsType:
        if isinstance(obs, dict):
            output = {**obs}
        else:
            output = {"obs": obs}
        output["switch"] = self.switch_function(self.env)
        return output


class ColorBlindWrapper(ObservationWrapper):
    """
    This wrapper takes a gridworld observation and converts it to a color-blind observation.

    Input: MultiDiscrete((width, height))
    Output: Box((width, height, 3), [0, 1])
    """

    def __init__(self, env,
                 merged_channels: tuple[int, ...] = (0, 1),
                 reduction: str = "mean",
                 disabled: bool = False):
        self.disabled = disabled
        self.merge_channels = list(merged_channels)
        assert reduction in ("mean", "max")
        self.reduction = reduction

        super().__init__(env)
        in_space = env.observation_space
        assert isinstance(in_space, MultiDiscrete), f"{self.__class__.__name__} expected MultiDiscrete, got {in_space}"
        assert len(in_space.shape) == 2, in_space
        self.observation_space = gym.spaces.Box(
            low=0,
            high=1,
            shape=(*in_space.shape, 3),
        )

        # noinspection PyUnresolvedReferences
        self.color_map = np.array([
            pygame.Color(cell.color)[:3]
            for cell in env.ALL_CELLS
        ], dtype=np.float) / 255.0
        print(self.color_map)

    def observation(self, obs: np.ndarray):
        image = self.color_map[obs]
        if not self.disabled:
            merged = image[..., self.merge_channels]
            reduction_funct = {
                "mean": np.mean,
                "max": np.max,
            }[self.reduction]
            merged[...] = reduction_funct(merged, axis=-1, keepdims=True)
        return image


class OneHotColorBlindWrapper(ObservationWrapper):
    """
    A wrapper that takes a gridworld observation and converts it to a "one-hot" color-blind observation.

    Here "one-hot" as quotes because each cell has a one at all the merged channels if one of them was active.

    Input: MultiDiscrete((width, height))
    Output: MultiBinary((width, height, n_cells))
    """

    def __init__(self, env: gym.Env,
                 merged_channels: tuple[int, ...] = (3, 4),
                 disabled: bool = False):
        # noinspection PyUnresolvedReferences
        self.n_goals = len(env.GOAL_CELLS)
        self.disabled = disabled
        self.merge_channels = list(merged_channels)

        super().__init__(env)
        in_space = env.observation_space
        assert isinstance(in_space, MultiDiscrete), f"{self.__class__.__name__} expected MultiDiscrete, got {in_space}"
        assert len(in_space.shape) == 2, in_space

        self.observation_space = gym.spaces.MultiBinary(n=(*in_space.shape, self.n_goals))

    def observation(self, observation: np.ndarray) -> WrapperObsType:
        # Convert to one-hot
        w, h = observation.shape
        one_hot = np.zeros((w, h, self.n_cells), dtype=bool)
        # for x in range(w):
        #     for y in range(h):
        #         one_hot[x, y, observation[x, y]] = True
        # One liner:
        one_hot[np.arange(w)[:, None], np.arange(h), observation] = True

        # Make indistinguishable
        if not self.disabled:
            one_hot[..., self.merge_channels] = one_hot[..., self.merge_channels].any(axis=-1, keepdims=True)

        return one_hot


class AddTrueGoalToObsFlat(ObservationWrapper):
    """
    Add the goal to the observation and flatten it.

    Input: MultiDiscrete(w, h)
    Output: MultiDiscrete(w * h + 1)

    Input: MultiBinary(w, h, n)
    Output: MultiBinary(w * h * n + n_goals)
    """

    def __init__(self, env):
        super().__init__(env)
        in_space = env.observation_space

        # noinspection PyUnresolvedReferences
        self.n_goals = len(env.GOAL_CELLS)

        if isinstance(in_space, MultiBinary):
            self.is_binary = True
            self.observation_space = MultiBinary(math.prod(in_space.shape) + self.n_goals)
        elif isinstance(in_space, MultiDiscrete):
            self.is_binary = False
            self.observation_space = MultiDiscrete(math.prod(in_space.shape) + 1)
        else:
            raise ValueError(f"Unknown observation space in {self.__class__.__name__}: {in_space}")

    def observation(self, obs: np.ndarray) -> ObsType:
        flat = obs.flatten()

        if self.is_binary:
            to_add = np.zeros((self.n_goals,), dtype=bool)
            to_add[self.true_goal_idx] = 1
        else:
            to_add = np.array([self.true_goal_idx], dtype=int)

        return np.concatenate([flat, to_add])
