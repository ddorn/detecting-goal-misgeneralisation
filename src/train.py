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


def get_agent_3_goal_blind(
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
        save: bool = True,
        return_perfs: bool = True,
):
    """Train a PPO agent on given environment"""

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

    # Set the random seed
    if seed is None:
        seed = random.randint(0, 2 ** 32 - 1)

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

    # Start wandb and define callbacks
    callbacks = [M.ProgressBarCallback(),
                 M.WeightDecayCallback(lambda f: (1 - f) * final_wd)]
    num_params = sum(p.numel() for p in policy.policy.parameters())
    config = dict(
        num_params=num_params,
        initial_lr=initial_lr,
        final_wd=final_wd,
        seed=seed,
        total_timesteps=total_timesteps,
        n_envs=n_envs,
    )
    if use_wandb:
        wandb.init(
            sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
            save_code=True,
            config=config,
        )
        callbacks.append(M.WandbWithBehaviorCallback(mk_env()))

    # Train the agent
    policy.learn(
        total_timesteps=total_timesteps,
        callback=callbacks,
    )

    if not return_perfs:
        return policy

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

    # Save the agent and stats
    if save:
        directory = MODELS_DIR / "3-goal-blind-one-hot"
        idx = 0
        while (directory / f"{idx}.zip").exists():
            idx += 1
        name = directory / f"{idx}.zip"
        assert not name.exists()
        policy.save(name)

        to_save = {
            "stats_blind": stats["blind"].tolist(),
            "stats_full_color": stats["full_color"].tolist(),
            **config,
        }
        json.dump(to_save, name.with_suffix(".json").open("w"))
        pprint(to_save)
        print(f"Saved model to {name}")

    return policy, stats


@click.command()
@click.option("--steps", default=400_000, help="Number of steps to train the agent for")
@click.option("--n-agents", default=1, help="Number of agents to train")
@click.option("--n-envs", default=4, help="Number of environments to train on")
@click.option("--jobs", default=1, help="Number of jobs to run in parallel")
@click.option("--env-size", default=4, help="Size of the environment")
@click.option("--n-evals", default=10_000, help="Number of episodes to evaluate the agent on")
@click.option("--lr", default=1e-3, help="Learning rate")
@click.option("--wd", default=8e-4, help="Weight decay")
@click.option("--no-wandb", is_flag=True, help="Disable wandb")
def train(
        steps: int,
        n_agents: int,
        n_envs: int,
        jobs: int,
        env_size: int,
        n_evals: int,
        lr: float,
        wd: float,
        no_wandb: bool,
):
    """Train a PPO agent on the SimpleEnv environment"""

    def get():
        get_agent_3_goal_blind(
            total_timesteps=steps,
            n_envs=n_envs,
            env_size=env_size,
            initial_lr=lr,
            final_wd=wd,
            use_wandb=not no_wandb,
            n_evals=n_evals,
        )
        # Not returning anything to avoid pickling the agent for nothing
        # Joblib also throws an error if the agent is returned (cannot pickle _thread.lock objects)

    if n_agents != 1:
        Parallel(n_jobs=jobs)(delayed(get)() for _ in range(n_agents))
    else:
        get()


def test():
    get_agent_3_goal_blind(
        total_timesteps=10_000,
        use_wandb=False,
        n_evals=100,
    )


if __name__ == "__main__":
    train()
    # test()
