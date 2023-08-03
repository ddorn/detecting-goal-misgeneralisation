#!/usr/bin/env python3.11
from __future__ import annotations

import dataclasses
import json
import random
from pprint import pprint
from typing import Type, Callable

import click
import gymnasium as gym
from joblib import Parallel, delayed
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.policies import ActorCriticPolicy
from wandb.integration.sb3 import WandbCallback

import wandb
from main import Perfs, random_goal_env


def get_agent(
    env: Callable[[], gym.Env],
    total_timesteps: int = 100_000,
    n_envs: int = 1,
    *,
    # Hyperparameters
    net_arch: tuple = (30, 10),
    learning_rate: float = 0.001,
    n_epochs: int = 40,
    n_steps: int = 6_000,
    batch_size: int = 6_000,
    policy: str | Type[ActorCriticPolicy] = "MlpPolicy",
    policy_kwargs: dict | None = None,
    # weight_decay: float = 0,
    seed: int | None = None,
    # Meta-options
    verbose: int = 2,
    use_wandb: bool = False,
    save: bool = True,
    return_perfs: bool = True,
):
    """Train a PPO agent on given environment"""

    if not return_perfs and save:
        raise ValueError("Cannot save agent if not returning perfs")

    # Set the random seed
    if seed is None:
        seed = random.randint(0, 2**32 - 1)

    if policy_kwargs is None:
        policy_kwargs = {}

    policy_kwargs.setdefault("net_arch", net_arch)

    # Define the policy network
    policy = PPO(
        policy,
        make_vec_env(env, n_envs=n_envs, seed=seed),
        verbose=verbose >= 2,
        learning_rate=learning_rate,
        # learning_rate=lambda f: 0.001 * f,
        # learning_rate=lambda f: 0.01 * f ** 1.5,
        policy_kwargs=policy_kwargs,  # optimizer_kwargs=dict(weight_decay=weight_decay)),
        # arch_kwargs=dict(net_arch=net_arch, features_extractor_class=BaseFeaturesExtractor),
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        tensorboard_log="run_logs",
        device="cpu",
        seed=seed,
    )

    # Show nb of parameters
    if verbose >= 1:
        print("Number of parameters", sum(p.numel() for p in policy.policy.parameters()))

    # Train the agent
    if use_wandb:
        wandb.init(
            sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
            save_code=True,
        )
    policy.learn(
        total_timesteps=total_timesteps,
        callback=WandbCallback(verbose=2) if use_wandb else None,
    )

    if not return_perfs:
        return policy

    # Evaluate the agent
    perfs = Perfs.from_agent(
        policy,
        episodes=100,
        env_size=env_size,
        can_turn=can_turn,
        net_arch=net_arch,
        learning_rate=learning_rate,
        steps=total_timesteps,
    )

    if use_wandb:
        wandb.log({
            "general_env": perfs.general_env,
            "br_env": perfs.br_env,
            "general_br_freq": perfs.general_br_freq,
        })
        wandb.finish()

    # Save the agent
    if save:
        parts = {
            "env": env.__name__,
            "steps": total_timesteps,
            "gen": perfs.general_env,
            "br": perfs.br_env,
            "br_wrong": perfs.general_br_freq,
        }

        name = ""
        for k, v in parts.items():
            if isinstance(v, float):
                name += f"{v * 1000:03.0f}{k}_"
            else:
                name += f"{v}{k}_"
        name = f"agents/ppo_{name[:-1]}.zip"

        perfs.info["file"] = name
        policy.save(name)
        json.dump(dataclasses.asdict(perfs), open(name + ".json", "w"))
        if verbose >= 1:
            print(f"Saved model to {name}")

    if verbose >= 1:
        pprint(perfs)

    return policy, perfs


@click.command()
@click.option("--steps", default=50_000, help="Number of steps to train the agent for")
@click.option(
    "--br-prob",
    default=1.0,
    help="Probability of the goal being in the bottom right corner vs elsewhere",
)
@click.option("--n-agents", default=1, help="Number of agents to train")
@click.option("--n-envs", default=1, help="Number of environments to train on")
@click.option("--jobs", default=1, help="Number of jobs to run in parallel")
@click.option("--env-size", default=5, help="Size of the environment")
@click.option("-v", "--verbose", count=True, help="Verbosity level (0-2)")
@click.option("--n-epochs", default=40, help="Number of epochs per update")
@click.option("--steps-per-update", default=6_000, help="Number of steps per update")
@click.option("--batch-size", default=400, help="Number of steps per update")
@click.option("--lr", default=0.001, help="Learning rate")
@click.option("--wandb", is_flag=True, help="Whether to use wandb")
@click.option(
    "--arch",
    default="30:10",
    callback=lambda ctx, param, value: tuple(map(int, value.split(":"))),
    type=str,
    help="Neural network architecture, as a string of integers separated by colons (ex. 30:10)",
)
def train(
    steps: int,
    br_prob: float,
    n_agents: int,
    n_envs: int,
    jobs: int,
    env_size: int,
    verbose: int,
    n_epochs: int,
    steps_per_update: int,
    batch_size: int,
    lr: float,
    wandb: bool,
    arch: tuple[int, ...],
):
    """Train a PPO agent on the SimpleEnv environment"""

    def get():
        get_agent(
            lambda: random_goal_env(env_size, br_prob),
            total_timesteps=steps,
            n_envs=n_envs,
            verbose=verbose,
            net_arch=arch,
            learning_rate=lr,
            n_epochs=n_epochs,
            n_steps=steps_per_update // n_envs,
            batch_size=batch_size,
            use_wandb=wandb,
        )
        # Not returning anything to avoid pickling the agent for nothing
        # Joblib also throws an error if the agent is returned (cannot pickle _thread.lock objects)

    if n_agents > 1:
        Parallel(n_jobs=jobs)(delayed(get)() for _ in range(n_agents))
    else:
        get()


def test():
    env_size = 7
    n_envs = 50

    # For bottom_right_odds, None means uniform, 3 means three times more likely to be bottom right than anywhere else
    get_agent(
        bottom_right_prob=0.5,
        total_timesteps=100_000,
        net_arch=(10, 10),
        n_epochs=40,
        n_steps=8_000 // n_envs,
        batch_size=400,
        learning_rate=0.001,
        env_size=env_size,
        n_envs=n_envs,
        can_turn=False,
        # policy=transformer.CustomActorCriticPolicy,
        # policy_kwargs=dict(features_extractor_class=transformer.CustomFeaturesExtractor, arch=dict(d_model=20, d_head=6, heads=3, layers=1)),
        # use_wandb=True,
        save=False,
        return_perfs=False,
    )


if __name__ == "__main__":
    train()
    # test()
