#!/usr/bin/env python3.11

from __future__ import annotations

import json
import random
from pathlib import Path
from pprint import pprint

import click
import wandb
from joblib import Parallel, delayed
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from torch import nn

import __init__ as M

HERE = Path(__file__).parent
ROOT = HERE.parent
MODELS_DIR = ROOT / "models"


def find_filename(directory: Path, prefix: str = "", ext="zip") -> Path:
    """Find the first available filename with the given prefix"""
    idx = 0
    while (directory / f"{prefix}{idx}.{ext}").exists():
        idx += 1
    return directory / f"{prefix}{idx}.{ext}"


def blind_3_goals_one_hot(
        total_timesteps: int = 400_000,
        n_envs: int = 4,
        n_evals: int = 10_000,
        *,
        # Hyperparameters
        env_size: int = 4,
        initial_lr: float = 1e-3,
        final_wd: float = 8e-4,
        seed: int | None = None,
        # Meta-options
        use_wandb: bool = True,
):
    """Train a PPO agent on given environment"""

    # Set the random seed
    if seed is None:
        seed = random.randint(0, 2 ** 32 - 1)

    args = dict(locals())
    pprint(args)

    # Define the architecture
    arch = nn.Sequential(
        M.Split(
            -3,
            left=nn.Sequential(
                M.Rearrange("... (h w c) -> ... c h w", h=env_size, w=env_size, c=5),
                nn.Conv2d(5, 8, 3, padding=1),
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

    # Define the environment
    mk_env_generator = lambda full_color: M.wrap(
        lambda: M.ThreeGoalsEnv(env_size, step_reward=0.0),
        # lambda e: M.ColorBlindWrapper(e, reduction='max', reward_indistinguishable_goals=True, disabled=full_color),
        lambda e: M.OneHotColorBlindWrapper(e, reward_indistinguishable_goals=True, disabled=full_color),
        lambda e: M.AddTrueGoalToObsFlat(e),
    )
    mk_env = mk_env_generator(full_color=False)
    mk_env_full_color = mk_env_generator(full_color=True)

    # Define the policy network
    policy = PPO(
        M.CustomActorCriticPolicy,
        make_vec_env(mk_env, n_envs=n_envs),
        policy_kwargs=dict(
            arch=M.L1WeightDecay(arch, 0),
            optimizer_kwargs=dict(weight_decay=0),
        ),
        n_steps=2_048 // n_envs,
        tensorboard_log=str(ROOT / "run_logs"),
        learning_rate=lambda f: f * initial_lr,
        seed=seed,
        device='cpu',
    )
    args['num_params'] = sum(p.numel() for p in policy.policy.parameters())

    # Start wandb and define callbacks
    callbacks = [M.ProgressBarCallback(),
                 M.WeightDecayCallback(lambda f: (1 - f) * final_wd)]

    if use_wandb:
        wandb.init(
            sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
            save_code=True,
            config=args,
        )
        callbacks.append(M.WandbWithBehaviorCallback(mk_env()))

    # Train the agent
    policy.learn(
        total_timesteps=total_timesteps,
        callback=callbacks,
    )

    # Evaluate the agent
    stats = {
        name: M.make_stats(policy, env, n_episodes=n_evals,
                           wandb_name=name if use_wandb else None, plot=False)
        for name, env in [
            ("blind", mk_env()),
            ("full_color", mk_env_full_color()),
        ]
    }

    # Log the evaluation stats
    if use_wandb:
        wandb.log({
            f"eval/{type_}/true_goal_{true_goal}/end_type_{end_type}": stat[tg, et]
            for tg, true_goal in enumerate(["red", "green", "blue"])
            for et, end_type in enumerate(["red", "green", "blue", "no goal"])
            for type_, stat in stats.items()
        })
        wandb.finish()

    # Save the agent
    filename = find_filename(MODELS_DIR / "3-goal-blind-one-hot")
    for name, m in policy.policy.named_modules():
        if hasattr(m, "logger"):
            # We stored it on the WeightDecay modules...
            del m.logger
    policy.save(filename)

    # Save the stats
    to_save = {
        "stats_blind": stats["blind"].tolist(),
        "stats_full_color": stats["full_color"].tolist(),
        **args,
    }
    json.dump(to_save, filename.with_suffix(".json").open("w"))

    print(f"Saved model to {filename}")

    return None


EXPERIMENTS = [
    get_agent_3_goal_blind,
]


@click.command()
@click.argument("experiment", type=click.Choice([e.__name__ for e in EXPERIMENTS]))
@click.option("--total-timesteps", "--steps", type=int, help="Number of steps to train the agent for")
@click.option("--n-envs", type=int, help="Number of environments to train on")
@click.option("--env-size", type=int, help="Size of the environment")
@click.option("--n-evals", type=int, help="Number of episodes to evaluate the agent on")
@click.option("--initial-lr", "--lr", type=float, help="Learning rate")
@click.option("--final-wd", "--wd", type=float, help="Weight decay")
@click.option("--use-wandb/--no-wandb", is_flag=True, help="Disable wandb")
@click.option("--n-agents", default=1, help="Number of agents to train")
@click.option("--jobs", default=1, help="Number of jobs to run in parallel")
def train(experiment: str, jobs: int, n_agents: int, **kwargs):
    """Train a PPO agent on the SimpleEnv environment"""

    # Remove None values
    kwargs = {k: v for k, v in kwargs.items() if v is not None}

    exp = next(e for e in EXPERIMENTS if e.__name__ == experiment)

    if n_agents != 1:
        Parallel(n_jobs=jobs)(delayed(exp)(**kwargs) for _ in range(n_agents))
    else:
        exp(**kwargs)


if __name__ == "__main__":
    train()
