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
from typing import Callable, Generator, Optional

import click
import gymnasium as gym
import rich
import rich.table
import rich.console
import torchinfo
import wandb
from joblib import Parallel, delayed
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback, EveryNTimesteps
from stable_baselines3.common.env_util import make_vec_env
from torch import nn
from tqdm.autonotebook import tqdm

try:
    import src
except ModuleNotFoundError:
    import __init__ as src

HERE = Path(__file__).parent
ROOT = HERE.parent
MODELS_DIR = ROOT / "models"

# Remove UserWarning in LazyModules about it being an experimental feature
warnings.filterwarnings("ignore", module="torch.nn.modules.lazy")


def find_filename(directory: Path, prefix: str = "", ext=".zip") -> Path:
    """Find the first available filename with the given prefix"""

    # Cleaner:
    while True:
        idx = random.randrange(0, 1_000_000)
        filename = directory / f"{prefix}{idx}{ext}"
        if not filename.exists():
            return filename.resolve()

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
    nb_checkpoints: int = field(
        default=0,
        metadata=dict(help="Number of checkpoints to save"),
    )
    seed: int = field(
        default=None,
        metadata=dict(help="Seed to use"),
    )

    def __post_init__(self):
        self.save_dir = find_filename(MODELS_DIR / self.name(), ext="")
        self.save_dir.mkdir(parents=True, exist_ok=False)

    @classmethod
    def name(cls) -> str:
        """Return the name of the experiment"""
        assert cls is not Experiment, "The base class Experiment should not be used directly."
        # Convert CamelCase to snake_case
        return re.sub(r'(?<!^)(?=[A-Z])', '_', cls.__name__).lower()


    def run(self):
        """Run one instance of the experiment"""

        args = dataclasses.asdict(self)
        args['save_dir'] = str(self.save_dir)
        if self.seed is None:
            seed = random.randint(0, 2 ** 32 - 1)
            args["seed"] = seed
        else:
            seed = self.seed

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
        if self.nb_checkpoints:
            steps_per_checkpoint = int(self.total_timesteps / self.nb_checkpoints)

            class CheckpointEvalCallback(EveryNTimesteps):
                def __init__(self, n_steps, experiment: Experiment):
                    super().__init__(n_steps, None)
                    self.experiment = experiment

                def _on_event(self):
                    checkpoint_evaluation = self.experiment.evaluate(self.model)
                    if self.experiment.use_wandb:
                        wandb.log(checkpoint_evaluation, commit=False)
                    self.experiment.save(policy,
                                         dict(timesteps=self.num_timesteps, eval=checkpoint_evaluation),
                                         self.num_timesteps)

            callbacks.append(CheckpointEvalCallback(steps_per_checkpoint, self))

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

        self.save(policy, dict(
            eval=evaluation,
            args=args,
            id=wandb.run.id if self.use_wandb else None,
        ))

        # Log the evaluation stats, and exit
        if self.use_wandb:
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

    def save(self, policy, metadata, num_timesteps: int = None):
        """Save the model and metadata"""
        if num_timesteps is None:
            num_timesteps = ""
        else:
            num_timesteps = f"_{num_timesteps}"

        model_file = self.save_dir / f"model{num_timesteps}.zip"
        assert not model_file.exists(), f"Model file {model_file} already exists"
        metadata_file = self.save_dir / f"metadata{num_timesteps}.json"
        assert not metadata_file.exists(), f"Metadata file {metadata_file} already exists"

        policy.save(model_file)
        metadata_file.write_text(json.dumps(metadata, indent=2))
        print(f"Saved model to {model_file}")

    @classmethod
    def load(cls, idx: int, checkpoint: Optional[int] = None, n_envs: int = None) -> tuple[PPO, dict]:
        """Load the model and metadata"""
        if checkpoint is None:
            checkpoint = ""
        else:
            checkpoint = f"_{checkpoint}"

        directory = MODELS_DIR / cls.name() / str(idx)
        metadata = json.loads((directory / f"metadata{checkpoint}.json").read_text())
        filename = directory / f"model{checkpoint}.zip"
        if n_envs is not None:
            policy = PPO.load(filename, env=make_vec_env(cls().get_eval_env, n_envs=n_envs))
        else:
            policy = PPO.load(filename)
        return policy, metadata

    @classmethod
    def load_all(cls, idx: Optional[int] = None) -> tuple[list[PPO], list[dict]]:
        """Load all the runs for this experiment. If an index is passed, load all checkpoint for this run instead."""

        folder = MODELS_DIR / cls.name()
        if idx is None:
            # Models are of the form {idx}/model.zip
            to_load = [(int(f.parent.stem), None) for f in folder.glob("*/model.zip")]
        else:
            folder = folder / str(idx)
            # Checkpoints are of the form model_{checkpoint}.zip
            to_load = [(idx, int(f.stem.split('_')[1])) for f in folder.glob("model_*.zip")]

        print("Loading", len(to_load), "models from", folder)
        models_and_stats = [cls.load(idx, checkpoint) for idx, checkpoint in tqdm(sorted(to_load))]
        models, stats = zip(*models_and_stats)
        return models, stats

    @classmethod
    def load_all_checkpoints_stats(cls, idx: Optional[int] = None) -> list[list[dict]]:
        """Load all the stats for this experiment. If an index is passed, load all checkpoint for this run instead."""

        folder = MODELS_DIR / cls.name()

        if idx is None:
            all_runs = [int(f.parent.stem) for f in folder.glob("*/model.zip")]
        else:
            all_runs = [idx]

        # Models are of the form {idx}/metadata_{steps}.json
        to_load = [
            [
                (int(f.parent.stem), int(f.stem.split('_')[1]))
                for f in folder.glob(f"{idx}/metadata_*.json")
            ]
            for idx in all_runs
        ]

        if len(set(map(len, to_load))) != 1:
            warnings.warn("Not all runs have the same number of checkpoints")

        print("Loading", sum(map(len, to_load)), "stats from", folder)
        stats = [
            [
                json.loads((folder / str(idx) / f"metadata_{steps}.json").read_text())
                for idx, steps in sorted(checkpoints)
            ]
            for checkpoints in tqdm(to_load)
        ]
        return stats


    @classmethod
    def load_v1(cls, idx: int, n_envs: int = None) -> tuple[PPO, dict]:
        """Load the model and metadata. Use with models trained before 2023-09-11."""
        filename = MODELS_DIR / cls.name() / f"{idx}.zip"
        metadata = json.load(filename.with_suffix(".json").open("r"))
        if n_envs is not None:
            policy = PPO.load(filename, env=make_vec_env(cls().get_eval_env, n_envs=n_envs))
        else:
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

    @classmethod
    def show_all_experiments(cls):
        """Show all the experiments that have been run with summary statistics."""

        table = rich.table.Table("Folder", "Nb runs", "Size",
                                 "Experiment class",
                                 title="Experiments",
                                 caption=f"Experiments found in {MODELS_DIR}",
                                 width=100,
                                 )

        experiments_by_name = {exp.name(): exp.__name__ for exp in cls.all_experiments()}

        for exp in MODELS_DIR.iterdir():
            n_runs = len(list(exp.glob("*.zip")))
            if not n_runs:
                continue

            total_size = sum(f.stat().st_size for f in exp.glob("*.zip"))

            # Find if this is an experiment folder
            table.add_row(
                exp.name,
                str(n_runs),
                f"{total_size / 1e6:.2f} MB",
                experiments_by_name.get(exp.name, ""),
            )

        console = rich.console.Console(force_jupyter=False)
        console.print(table, overflow="ignore", crop=False)


class BlindThreeGoals(Experiment):
    """
    Blind RGB version of ThreeGoalsEnv

    The agent is trained on a color-blind version of the environment, where the agent
    cannot distinguish between red and green channels of the image, but only the max
    of the two.
    """

    def get_arch(self) -> nn.Module:
        # Define the architecture
        return nn.Sequential(
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

    def get_env(self, full_color: bool) -> Callable[[], gym.Env]:
        return src.wrap(
            lambda: src.ThreeGoalsEnv(self.env_size, step_reward=0.0),
            lambda env: src.ColorBlindWrapper(env, reduction='max',
                                              reward_indistinguishable_goals=True, disabled=full_color),
            lambda env: src.AddTrueGoalToObsFlat(env),
        )

    def policy_kwargs(self) -> dict[str, object]:
        # We use L1 weight decay, not L2 here
        return dict(
            optimizer_kwargs=dict(weight_decay=0),
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
            f"eval/{type_}/true_goal_{true_goal}/end_type_{end_type}": stat[tg, et]
            for et, end_type in enumerate(["red", "green", "blue", "no goal"])
            for tg, true_goal in enumerate(["red", "green", "blue"])
            for type_, stat in stats.items()
        }


@dataclass
class BlindThreeGoalsOneHot(Experiment):
    """
    Blind one-hot version of ThreeGoalsEnv

    The agent is trained on a color-blind version of the environment, where the agent
    cannot distinguish between red and green, that is, whenever a cell contains a red
    or green goal, the agent sees a 1 in both the red and green channels.
    """

    def get_arch(self):
        arch = super().get_arch()
        return src.L1WeightDecay(arch, 0)

    def get_env(self, full_color: bool) -> Callable[[], gym.Env]:
        """Return a function that returns the environment"""
        return src.wrap(
            lambda: src.ThreeGoalsEnv(self.env_size, step_reward=0.0),
            lambda env: src.OneHotColorBlindWrapper(env, reward_indistinguishable_goals=True, disabled=full_color),
            lambda env: src.AddTrueGoalToObsFlat(env),
        )

@dataclass
class BlindThreeGoalsRgbChannelReg(BlindThreeGoals):
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

    def get_arch(self):
        arch = super().get_arch()
        return src.PerChannelL1WeightDecay(
            arch,
            0,
            # Only the weight of the first convolutional layer
            name_filter="mlp_extractor.policy_net.0.left.1.module.weight",
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
