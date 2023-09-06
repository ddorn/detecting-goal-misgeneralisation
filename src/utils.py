from __future__ import annotations

from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from itertools import groupby
from typing import TypeVar, Callable, Literal, Union, TYPE_CHECKING

import einops
import gymnasium as gym
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import pygame.gfxdraw
import sklearn
import torch
import wandb
from jaxtyping import Float
from sklearn.model_selection import train_test_split
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from torch import nn, Tensor
from tqdm.autonotebook import tqdm
from wandb.integration.sb3 import WandbCallback

import architectures
import environments

if TYPE_CHECKING:
    from environments import ThreeGoalsEnv

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
        policy,
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
        wandb.log({f"behavior": wandb.Image(imgs)}, commit=False)

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

    def remove_batch_dim(self):
        """Remove the batch dimension from all activations."""
        if any(activation.shape[0] != 1 for activation in self.values()):
            raise ValueError("Not all activations have batch dimension 1.")

        for name, activation in self.items():
            self[name] = activation.squeeze(0)

    def apply(self, func: Callable[[Tensor], Tensor]):
        """Apply a function to all activations."""
        for name, activation in self.items():
            self[name] = func(activation)


@contextmanager
def record_activations(module: nn.Module) -> Cache:
    """Context manager to record activations from a module and its submodules.

    Args:
        module (nn.Module): Module to record activations from.

    Yields:
        dist[str, Tensor]: Dictionary of activations, that will be populated once the
            context manager is exited.
    """

    cache = Cache()
    activations: dict[str, list[Tensor]] = {}
    hooks = []

    skipped = set()
    module_to_name = {m: f"{n} {m.__class__.__name__}" for n, m in module.named_modules()}

    def hook(m: nn.Module, input: Tensor, output: Tensor):
        name = module_to_name[m]
        if not isinstance(output, Tensor):
            skipped.add(name)
        elif name not in activations:
            activations[name] = [output.detach()]
        else:
            activations[name].append(output.detach())

    for module in module.modules():
        hooks.append(module.register_forward_hook(hook))

    try:
        yield cache
    finally:
        for hook in hooks:
            hook.remove()

    for name, activation in activations.items():
        if len(activation) == 1:
            cache[name] = activation[0]
        else:
            cache[name] = torch.stack(activation)

    if skipped:
        print("Skipped:")
        for name in skipped:
            print("-", name)


# noinspection PyDefaultArgument
def unique(x, *, __previous=set()):
    """Return the argument, if it was never seen before, otherwise raise ValueError"""
    if x in __previous:
        raise ValueError(f"Duplicate value {x}")
    __previous.add(x)
    return x


class WandbWithBehaviorCallback(WandbCallback):
    def __init__(self, env: gym.Env, show_every=10, **kwargs):
        self.env = env
        self.show_every = show_every
        self.time = 0
        super().__init__(**kwargs)

    def _on_rollout_start(self) -> None:
        super()._on_rollout_start()
        # Show every 10 rollouts
        self.time += 1
        if self.time % self.show_every == 0:
            show_behavior(self.model, self.env, max_len=20, add_to_wandb=True, plot=False)


class ProgressBarCallback(BaseCallback):
    """
    Display a progress bar when training SB3 agent
    using tqdm (and not rich).
    """

    def __init__(self) -> None:
        super().__init__()
        self.pbar = None

    def _on_training_start(self) -> None:
        # Initialize progress bar
        # Remove timesteps that were done in previous training sessions
        self.pbar = tqdm(total=self.locals["total_timesteps"] - self.model.num_timesteps)

    def _on_step(self) -> bool:
        # Update progress bar, we do num_envs steps per call to `env.step()`
        self.pbar.update(self.training_env.num_envs)
        return True

    def _on_training_end(self) -> None:
        # Flush and close progress bar
        self.pbar.refresh()
        self.pbar.close()


class WeightDecayCallback(BaseCallback):
    """Callback to schedule L1 weight decay."""

    def __init__(self, schedule: Callable[[float], float]):
        super().__init__()
        self.schedule = schedule
        self.modules_to_decay: list[architectures.WeightDecay] = []

    def _init_callback(self) -> None:
        self.modules_to_decay = [
            module for module in self.model.policy.modules()
            # We would do isinstance(module, architectures.WeightDecay), but that would
            # not work when the code is reloaded in a notebook.
            if any("WeightDecay" in c.__name__ for c in module.__class__.__mro__)
        ]
        assert len(self.modules_to_decay) > 0, "No modules to decay"

    def _on_rollout_end(self) -> None:
        # noinspection PyProtectedMember
        prop = self.model._current_progress_remaining
        wd = self.schedule(prop)
        self.logger.record("train/weight_decay", wd)
        for module in self.modules_to_decay:
            module.weight_decay = wd

    def _on_step(self) -> bool:
        return True


class LogChannelNormsCallback(BaseCallback):
    def _on_step(self) -> bool:
        conv1 = next(m for m in self.model.policy.modules() if isinstance(m, nn.Conv2d))
        by_in_channel = conv1.weight.norm(2, dim=(0, 2, 3))
        for i, norm in enumerate(by_in_channel):
            self.logger.record(f"train/channel_norm/{i}", norm.item())
        return True


def sample_trajectories(
        *trajectories_groups: list[Trajectory],
        n_trajectories: int = 30,
) -> list[Trajectory]:
    """
    Sample from multiple groups of trajectories, with sample equilibrated across the length of the trajectory.
    """

    to_show = []
    traj_by_kind = n_trajectories // len(trajectories_groups)
    for trajectories in trajectories_groups:
        n = len(trajectories)
        if n < traj_by_kind:
            to_show += trajectories
            continue

        by_len = [list(g) for _, g in groupby(trajectories, len)]
        to_show_here = []
        while len(to_show_here) < traj_by_kind:
            for traj in by_len:
                if traj:
                    to_show_here.append(traj.pop())
                if len(to_show_here) == traj_by_kind:
                    break

        to_show_here.sort(key=len)
        to_show += to_show_here

    return to_show


def evaluate(policy_, env_: ThreeGoalsEnv,
             n_episodes=1000, max_len=20, show_n=30,
             add_to_wandb=False, plot=True, **plotly_kwargs):
    """Return the proportion of episodes where the agent reached the true goal."""
    found = 0
    terminated = 0

    samples = defaultdict(list)
    for _ in tqdm(range(n_episodes)):
        trajectory = Trajectory.from_policy(
            policy_, env_, max_len=max_len,
            # end_condition=lambda locals_: env_.agent_pos == env_.goal_positions[env_.true_goal_idx]
            end_condition=lambda locals_: env_.last_reward == 1
        )

        if trajectory.ended == "condition":
            found += 1
        elif trajectory.ended == "terminated":
            terminated += 1

        if show_n:
            samples[trajectory.ended].append(trajectory)

    got_reward = found / n_episodes
    wrong_goal = terminated / n_episodes
    no_goal = (n_episodes - found - terminated) / n_episodes

    if show_n:
        to_show = sample_trajectories(*samples.values(), n_trajectories=show_n)
        title = f"Got reward: {got_reward:.1%} | Truncated: {no_goal:.1%} | Wrong goal: {wrong_goal:.1%}"
        show_behavior(policy_, to_show,
                      add_to_wandb=add_to_wandb, title=title, plot=plot,
                      **plotly_kwargs)

    return {
        "Got reward": got_reward,
        "Terminated": no_goal,
        "Wrong goal": wrong_goal,
    }


def destination_stats(policy, env: gym.Env, n_episodes=100,
                      ) -> tuple[
    Float[Tensor, "step agent_pos=2"],
    Float[Tensor, "step goal_type=3 goal_position=2"],
    Float[Tensor, "step end_type=1"],
    Float[Tensor, "step true_goal=1"],
]:
    """
    Returns:
        - the position of the agent
        - the position of the goals (red, green, blue)
        - the type of end (end on red, green, blue or did not find goal)
        - the true goal
    """
    env: environments.ThreeGoalsEnv
    assert isinstance(env.unwrapped, environments.ThreeGoalsEnv)

    agent_pos = []
    goal_positions = []
    end_types = []
    true_goals = []

    for _ in tqdm(range(n_episodes)):
        obs, _ = env.reset()
        done = terminated = False
        episode_len = 0
        while not (done or terminated):
            action, _ = policy.predict(obs)
            obs, _, done, terminated, _ = env.step(action)

            episode_len += 1
            agent_pos.append(env.agent_pos)

        try:
            end_type = env.goal_positions.index(env.agent_pos)
        except ValueError:
            end_type = 3

        end_types += [end_type] * episode_len
        goal_positions += [env.goal_positions] * episode_len
        true_goals += [env.true_goal_idx] * episode_len

    return (
        torch.tensor(agent_pos),
        torch.tensor(goal_positions),
        torch.tensor(end_types),
        torch.tensor(true_goals),
    )


def make_stats(policy, env: environments.ThreeGoalsEnv, n_episodes=100, subtitle: str = "",
               wandb_name: str = None, plot: bool = True) -> Float[Tensor, "true_goal=3 end_pos=4"]:
    """
    Returns stats of where the policy ended, given the true goal.
    """

    stats = np.zeros((3, 4))
    for _ in tqdm(range(n_episodes)):
        obs, _ = env.reset()
        true_goal = env.true_goal_idx
        done = terminated = False
        while not (done or terminated):
            action, _ = policy.predict(obs)
            obs, _, done, terminated, _ = env.step(action)

        try:
            end_goal = env.goal_positions.index(env.agent_pos)
        except ValueError:
            end_goal = 3
        stats[true_goal, end_goal] += 1

    stats = stats / stats.sum(-1, keepdims=True)

    # Plot with plotly
    fig = go.Figure(data=go.Heatmap(
        z=stats,
        x=["red", "green", "blue", "no goal"],
        y=["red", "green", "blue"],
        hoverongaps=False,
        hovertemplate="True goal: %{y}<br>End goal: %{x}<br>Proportion: %{z:.1%}<extra></extra>",
        zmin=0,
        zmax=1,
        colorscale="Blues",
    ))
    # Write the % inside each cell
    for i in range(3):
        for j in range(4):
            fig.add_annotation(
                x=j,
                y=i,
                text=f"{stats[i, j]:.1%}",
                showarrow=False,
                font_size=20,
            )

    if subtitle:
        subtitle += "<br>"

    fig.update_layout(
        title=subtitle + f"Proportion of trajectories ending at each goal (n={n_episodes})",
        xaxis_title="End goal",
        yaxis_title="True goal",
        width=500,
        height=500,
    )

    if wandb_name:
        wandb.log({wandb_name: fig}, commit=False)
    if plot:
        fig.show()

    return stats


def add_line(fig, equation: str):
    minx, maxx = fig.data[0].x.min(), fig.data[0].x.max()

    x = np.linspace(minx, maxx, 100)

    # Parse equation
    left, _, right = equation.partition("=")
    if left.strip() == "y":
        y = eval(right)
    elif right.strip() == "y":
        y = eval(left)
    else:
        raise ValueError(f"Equation {equation} should contain y on one side")

    fig.add_trace(go.Scatter(
        x=x,
        y=y,
        mode="lines",
        line=dict(color="red", width=2),
        # Add to legend
        name=equation,
    ))


def show_fit(reg, x, y, title: str, xaxis: str, yaxis: str, classification: bool = False):
    """Fit and show the fit of a regression model on a train and test set."""
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    reg.fit(x_train, y_train)

    if classification:
        matrix = sklearn.metrics.confusion_matrix(y_test, reg.predict(x_test))
        matrix = matrix / matrix.sum()

        fig = go.Figure(data=go.Heatmap(
            z=matrix,
            x=["False", "True"],
            y=["False", "True"],
            hoverongaps=False,
            hovertemplate="True: %{y}<br>Predicted: %{x}<br>Count: %{z}<extra></extra>",
            zmin=0,
            zmax=1,
            colorscale="Blues",
        ))
        # Write the % inside each cell
        for i in range(2):
            for j in range(2):
                fig.add_annotation(
                    x=j,
                    y=i,
                    text=f"{matrix[i, j]:.1%}",
                    showarrow=False,
                    font_size=20,
                )

        fig.update_layout(
            title=title + f"<br>Accuracy: {reg.score(x_test, y_test):.3f} | Train size: {len(x_train)} | Test size: {len(x_test)} | Nvars: {len(x_train[0])}",
            xaxis_title=xaxis,
            yaxis_title=yaxis,
            width=600,
            height=600,
        )
        fig.show()

        return reg

    fig = go.Figure()
    # Add the predictions on the train set, with a semi-transparent color
    fig.add_scatter(x=y_train, y=reg.predict(x_train), mode="markers", marker=dict(color="rgba(255, 165, 0, 0.2)"),
                    name="Train set")
    # Add the predictions on the test set, with a blue color
    fig.add_scatter(x=y_test, y=reg.predict(x_test), mode="markers", marker=dict(color="rgba(0, 0, 255, 1)"),
                    name="Test set")
    # Add the line y=x
    add_line(fig, "y=x")
    fig.update_layout(
        title=title + f"<br>R2 score: {reg.score(x_test, y_test):.3f} | Train size: {len(x_train)} | Test size: {len(x_test)} | Nvars: {len(x_train[0])}",
        xaxis_title=xaxis,
        yaxis_title=yaxis,
        width=1000, height=1000)
    fig.show()

    return reg
