from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import TypeVar, Callable, Literal, Union

import einops
import gymnasium as gym
import numpy as np
import plotly.express as px
import pygame.gfxdraw
import torch
import wandb
from stable_baselines3 import PPO
from torch import nn, Tensor

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


class Cache(dict[str, Tensor]):

    def __str__(self):
        return "Cached activations:\n" + "\n".join(
            f"- {name}: {tuple(activation.shape)}" for name, activation in self.items()
        )

    def __getitem__(self, item: str) -> Tensor:
        # Find the key that matches and make sure it's unique.
        if item in self:
            return super().__getitem__(item)

        keys = [key for key in self.keys() if item in key]
        if len(keys) == 0:
            raise KeyError(item)
        elif len(keys) > 1:
            raise KeyError(f"Multiple keys match {item}: {keys}")
        return super().__getitem__(keys[0])


@contextmanager
def record_activations(module: nn.Module) -> Cache:
    """Context manager to record activations from a module and its submodules.

    Args:
        module (nn.Module): Module to record activations from.

    Yields:
        dist[str, Tensor]: Dictionary of activations, that will be populated once the
            context manager is exited.
    """

    activations = Cache()
    hooks = []

    skipped = set()
    module_to_name = {m: f"{n} {m.__class__.__name__}" for n, m in module.named_modules()}

    def hook(m: nn.Module, input: Tensor, output: Tensor):
        name = module_to_name[m]
        if not isinstance(output, Tensor):
            skipped.add(name)
        elif name not in activations:
            activations[name] = output.detach()
        else:
            activations[name] = torch.cat([activations[name], output.detach()], dim=0)

    for module in module.modules():
        hooks.append(module.register_forward_hook(hook))

    try:
        yield activations
    finally:
        for hook in hooks:
            hook.remove()

    print("Skipped:")
    for name in skipped:
        print("-", name)
