#!/usr/bin/env python3.11

"""
Train agents on different environments and setups.
"""

import dataclasses
import json
import random
import re
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from pprint import pprint
from typing import Callable, Generator

import click
import gymnasium as gym
import torchinfo
import wandb
from joblib import Parallel, delayed
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from torch import nn

try:
    import src
except ModuleNotFoundError:
    import __init__ as src

HERE = Path(__file__).parent
ROOT = HERE.parent
MODELS_DIR = ROOT / "models"

# Remove UserWarning in LazyModules about it being an experimental feature
warnings.filterwarnings("ignore", module="torch.nn.modules.lazy")


def find_filename(directory: Path, prefix: str = "", ext="zip") -> Path:
    """Find the first available filename with the given prefix"""
    idx = 0
    while (directory / f"{prefix}{idx}.{ext}").exists():
        idx += 1
    return directory / f"{prefix}{idx}.{ext}"


def apply_all(decorators):
    """Apply all the decorators to a function"""

    def _decorator(f):
        for opt in reversed(list(decorators)):
            f = opt(f)
        return f

    return _decorator


@dataclass(kw_only=True)
class Experiment(ABC):
    """
    Abstract class for an experiment
    """

    total_timesteps: int = field(
        default=400_000,
        metadata=dict(help="Number of steps to train the agent for")
    )
    n_envs: int = field(
        default=4,
        metadata=dict(help="Number of environments to train on"),
    )
    n_evals: int = field(
        default=10_000,
        metadata=dict(help="Number of episodes to evaluate the agent on"),
    )
    initial_lr: float = field(
        default=1e-3,
        metadata=dict(help="Learning rate"),
    )
    final_wd: float = field(
        default=8e-4,
        metadata=dict(help="Weight decay"),
    )
    use_wandb: bool = field(
        default=True,
        metadata=dict(help="Disable wandb"),
    )
    env_size: int = field(
        default=4,
        metadata=dict(help="Size of the environment"),
    )
    seed: int = field(
        default=None,
        metadata=dict(help="Seed to use"),
    )

    @classmethod
    def name(cls) -> str:
        """Return the name of the experiment"""
        assert cls is not Experiment, "The base class Experiment should not be used directly."
        # Convert CamelCase to snake_case
        return re.sub(r'(?<!^)(?=[A-Z])', '_', cls.__name__).lower()

    def run(self):
        """Run one instance of the experiment"""

        if self.seed is None:
            seed = random.randint(0, 2 ** 32 - 1)
            args = {**dataclasses.asdict(self), "seed": seed}
        else:
            seed = self.seed
            args = dataclasses.asdict(self)

        # Define the policy network
        policy = PPO(
            src.CustomActorCriticPolicy,
            make_vec_env(self.get_train_env, n_envs=self.n_envs),
            policy_kwargs=dict(arch=self.get_arch(), **self.policy_kwargs()),
            n_steps=2_048 // self.n_envs,
            tensorboard_log=str(ROOT / "run_logs"),
            learning_rate=lambda f: f * self.initial_lr,
            seed=seed,
            device='cpu',
        )

        # Start wandb and define callbacks
        callbacks = [src.ProgressBarCallback(), *self.get_callbacks()]
        if self.use_wandb:
            wandb.init(
                sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
                save_code=True,
                config=args,
                project=self.name(),
            )
            callbacks.append(src.WandbWithBehaviorCallback(self.get_eval_env()))

        # Train the agent
        policy.learn(
            total_timesteps=self.total_timesteps,
            callback=callbacks,
        )

        evaluation = self.evaluate(policy)

        filename = self.save(policy, dict(
            eval=evaluation,
            args=args,
        ))

        # Log the evaluation stats, filename, and exit
        if self.use_wandb:
            wandb.config.filename = filename
            wandb.log(evaluation, commit=False)
            wandb.finish()

    def policy_kwargs(self) -> dict[str, object]:
        """Return the keyword arguments to pass to the policy"""
        return {}

    @abstractmethod
    def get_arch(self) -> nn.Module:
        """Return the architecture of the policy network"""
        raise NotImplementedError()

    @abstractmethod
    def get_train_env(self) -> gym.Env:
        """Return the training environment"""
        raise NotImplementedError()

    @abstractmethod
    def get_eval_env(self) -> gym.Env:
        """Return the evaluation environment"""
        raise NotImplementedError()

    @abstractmethod
    def evaluate(self, policy) -> dict[str, object]:
        """Evaluate the agent. Return a json serializable dictionary of evaluation stats."""
        raise NotImplementedError()

    def get_callbacks(self) -> list[BaseCallback]:
        """Return a list of SB3 callbacks to use during training."""
        return [
            src.LogChannelNormsCallback(),
            src.WeightDecayCallback(lambda f: (1 - f) * self.final_wd),
        ]

    def save(self, policy, metadata):
        """Save the model and metadata"""
        filename = find_filename(MODELS_DIR / self.name())
        policy.save(filename)

        json.dump(metadata, filename.with_suffix(".json").open("w"))

        print(f"Saved model to {filename}")
        return filename

    @classmethod
    def load(cls, idx: int) -> tuple[PPO, dict]:
        """Load the model and metadata"""
        filename = MODELS_DIR / cls.name() / f"{idx}.zip"
        metadata = json.load(filename.with_suffix(".json").open("r"))
        policy = PPO.load(filename)
        return policy, metadata

    @classmethod
    def all_experiments(cls) -> Generator["Experiment", None, None]:
        """Return all experiments classes"""

        if cls is not Experiment:
            yield cls

        for subclass in cls.__subclasses__():
            yield from subclass.all_experiments()

    @classmethod
    def make_command(cls):
        """Return a click command for this experiment"""

        assert cls.__doc__ is not None, f"Docstring must be defined for {cls.__name__}"

        @cli.command(name=cls.name(), help=cls.__doc__)
        @apply_all(
            click.option("--" + arg.name.replace("_", "-"),
                         type=arg.type,
                         default=arg.default,
                         show_default=True,
                         **arg.metadata)
            for arg in dataclasses.fields(cls)
        )
        # Meta options
        @click.option("--n-agents", default=1, help="Number of agents to train")
        @click.option("--jobs", default=1, help="Number of jobs to run in parallel")
        @click.option("--dry-run", is_flag=True, help="Don't actually run the experiment")
        def _cmd(jobs, n_agents, dry_run, **kwargs):

            experiment = cls(**kwargs)

            if dry_run:
                click.secho(f"Dry run {cls.name()} with args:", fg="yellow")
                pprint(dataclasses.asdict(experiment))
                click.secho("Environment:", fg="yellow")
                env = experiment.get_train_env()
                print(env)
                obs, _ = env.reset()
                print(f"Observation shape: {obs.shape}")
                print("Architecture:")
                torchinfo.summary(experiment.get_arch(), input_size=obs.shape, depth=99)
                return

            if n_agents != 1:
                Parallel(n_jobs=jobs)(delayed(experiment.run)() for _ in range(n_agents))
            else:
                experiment.run()

        return _cmd


@dataclass
class BlindThreeGoalsOneHot(Experiment):
    """
    Blind one-hot version of ThreeGoalsEnv

    The agent is trained on a color-blind version of the environment, where the agent
    cannot distinguish between red and green, that is, whenever a cell contains a red
    or green goal, the agent sees a 1 in both the red and green channels.
    """

    def get_arch(self):
        arch = nn.Sequential(
            src.Split(
                -3,
                left=nn.Sequential(
                    src.Rearrange("... (h w c) -> ... c h w", h=self.env_size, w=self.env_size),
                    nn.LazyConv2d(8, 3, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(8, 8, 3, padding=1),
                    nn.ReLU(),
                    nn.Flatten(-3),
                ),
                right=nn.Identity(),
            ),
            nn.LazyLinear(32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
        )
        arch = src.L1WeightDecay(arch, 0)
        return arch

    def policy_kwargs(self) -> dict[str, object]:
        # We use L1 weight decay, not L2 here
        return dict(
            optimizer_kwargs=dict(weight_decay=0),
        )

    def get_env(self, full_color: bool) -> Callable[[], gym.Env]:
        """Return a function that returns the environment"""
        return src.wrap(
            lambda: src.ThreeGoalsEnv(self.env_size, step_reward=0.0),
            lambda env: src.OneHotColorBlindWrapper(env, reward_indistinguishable_goals=True, disabled=full_color),
            lambda env: src.AddTrueGoalToObsFlat(env),
        )

    def get_train_env(self) -> gym.Env:
        return self.get_env(full_color=False)()

    def get_eval_env(self) -> gym.Env:
        return self.get_env(full_color=True)()

    def _eval_envs(self) -> dict[str, gym.Env]:
        return {
            "blind": self.get_train_env(),
            "full_color": self.get_eval_env(),
        }

    def evaluate(self, policy) -> dict[str, object]:
        # Evaluate the agent
        stats = {
            name: src.make_stats(policy, env, n_episodes=self.n_evals,
                                 wandb_name=name if self.use_wandb else None, plot=False)
            for name, env in self._eval_envs().items()
        }

        return {
            type_: {
                f"true_goal_{true_goal}": {
                    f"end_type_{end_type}": stat[tg, et]
                    for et, end_type in enumerate(["red", "green", "blue", "no goal"])
                }
                for tg, true_goal in enumerate(["red", "green", "blue"])
            }
            for type_, stat in stats.items()
        }


@dataclass
class BlindThreeGoalsRgbChannelReg(BlindThreeGoalsOneHot):
    """
    Blind RGB version of ThreeGoalsEnv with channel regularization

    The agent is trained on a color-blind version of the environment, where the agent
    cannot distinguish between red and green channels of the image, but only the max
    of the two.
    The first convolutional layer has the channel with the lowest norm regularized
    strongly.
    """

    final_wd: float = field(
        default=0.01,
        metadata=dict(help="Weight decay"),
    )

    def __init__(self, *, initial_lr: float = 0.1, **kwargs):
        super().__init__(initial_lr=initial_lr, **kwargs)

    def get_arch(self) -> nn.Module:
        # Define the architecture
        return nn.Sequential(
            src.Split(
                -3,
                left=nn.Sequential(
                    src.Rearrange("... (h w c) -> ... c h w", h=self.env_size, w=self.env_size),
                    src.PerChannelL1WeightDecay(
                        nn.LazyConv2d(8, 3, padding=1),
                        0,
                        name_filter="weight",
                    ),
                    nn.ReLU(),
                    nn.Conv2d(8, 8, 3, padding=1),
                    nn.ReLU(),
                    nn.Flatten(-3),
                ),
                right=nn.Identity(),
            ),
            nn.LazyLinear(32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
        )

    def get_env(self, full_color: bool) -> Callable[[], gym.Env]:
        return src.wrap(
            lambda: src.ThreeGoalsEnv(self.env_size, step_reward=0.0),
            lambda env: src.ColorBlindWrapper(env, reduction='max',
                                              reward_indistinguishable_goals=True, disabled=full_color),
            lambda env: src.AddTrueGoalToObsFlat(env),
        )


@dataclass
class BlindThreeGoalWeightedChannel(BlindThreeGoalsOneHot):
    """
    Blind RGB version of ThreeGoalsEnv with weighted channels

    The agent is trained on a color-blind version of the environment, where the agent
    cannot distinguish between red and green channels of the image, but only the max
    of the two. However, the green channel is then multiplied by a weight.
    """

    green_weight: float = field(
        default=0.5,
        metadata=dict(help="Weight of the green channel"),
    )

    def get_arch(self) -> nn.Module:
        arch = super().get_arch()
        # We use the same architecture, but without the L1WeightDecay wrapping it
        arch.remove()
        return arch.module

    def get_callbacks(self) -> list[BaseCallback]:
        # We don't want to use the L1WeightDecay callback
        # But still log the norm of Conv1
        return [
            src.LogChannelNormsCallback(),
        ]

    def get_env(self, full_color: bool, weighted: bool = True) -> Callable[[], gym.Env]:
        weights = [1.0, self.green_weight, 1.0]
        return src.wrap(
            lambda: src.ThreeGoalsEnv(self.env_size, step_reward=0.0),
            lambda env: src.ColorBlindWrapper(env, reduction='max',
                                              reward_indistinguishable_goals=True, disabled=full_color),
            lambda env: src.WeightedChannelWrapper(env, weights, disabled=not weighted),
            lambda env: src.AddTrueGoalToObsFlat(env),
        )

    def _eval_envs(self) -> dict[str, gym.Env]:
        return {
            "blind_weighted": self.get_env(full_color=False)(),
            "full_color_weighted": self.get_env(full_color=True)(),
            "blind_non_weighted": self.get_env(full_color=False, weighted=False)(),
            "full_color_non_weighted": self.get_env(full_color=True, weighted=False)(),
        }


@click.group(context_settings=dict(max_content_width=200))
def cli():
    """Train agents on different environments and setups."""


for e in Experiment.all_experiments():
    cli.add_command(e.make_command())

if __name__ == "__main__":
    cli()
