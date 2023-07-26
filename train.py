#!/usr/bin/env python3.11
from __future__ import annotations

import dataclasses
import json
import time
from typing import Type

import click
from joblib import Parallel, delayed
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

import wandb
from wandb.integration.sb3 import WandbCallback

from main import wrap_env, SimpleEnv, uniform_distribution, Perfs


def get_agent(
        bottom_right_prob: float | None = None,
        total_timesteps: int = 50_000,
        n_envs: int = 1,
        env_size: int = 5,
        save: bool = True,
        verbose: int = 2,
        use_wandb: bool = False,
        *,
        net_arch: tuple = (30, 10),
        learning_rate: float = 0.001,
        n_epochs: int = 40,
        n_steps: int = 6_000,
        batch_size: int = 6_000,
        policy: str | Type[ActorCriticPolicy] = "MlpPolicy",
        policy_kwargs: dict | None = None,
        # weight_decay: float = 0,
):
    """Train a PPO agent on the SimpleEnv environment"""

    # Define the training environment
    goal_distrib = uniform_distribution((env_size - 1, env_size - 1))
    if bottom_right_prob == 1:
        goal_distrib = (-2, -2)
    elif bottom_right_prob is not None:
        # There are (env_size-2)**2-1 other positions
        goal_distrib[env_size - 2, env_size - 2] = bottom_right_prob / (1 - bottom_right_prob) * ((env_size - 2) ** 2 - 1)
    env = make_vec_env(
        lambda: wrap_env(
            SimpleEnv(
                size=env_size,
                goal_pos=goal_distrib,
                # goal_pos=(-2, -2),
                agent_start_pos=None,
                # agent_start_pos=(1, 1),
                # agent_start_dir=0,
                # render_mode='rgb_array'
            )),
        n_envs=n_envs,
    )

    if policy_kwargs is None:
        policy_kwargs = {}

    policy_kwargs.setdefault("net_arch", net_arch)

    # Define the policy network
    policy = PPO(
        policy,
        env,
        verbose=verbose >= 2,
        learning_rate=learning_rate,
        # learning_rate=lambda f: 0.001 * f,
        # learning_rate=lambda f: 0.01 * f ** 1.5,
        policy_kwargs=dict(**policy_kwargs),  # optimizer_kwargs=dict(weight_decay=weight_decay)),
        # arch_kwargs=dict(net_arch=net_arch, features_extractor_class=BaseFeaturesExtractor),
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,

        # buffer_size=10_000,
        # learning_starts=5_000,
        # gradient_steps=1,
        # target_update_interval=1_000,
        # exploration_fraction=0.2,
        # exploration_final_eps=0.2,
        gamma=1,
        tensorboard_log="run_logs",
        device="cpu",
    )
    if verbose >= 1:
        # nb of parameters
        print("Number of parameters", sum(p.numel() for p in policy.policy.parameters()))


    # Train the agent
    if use_wandb:
        wandb.init(
            sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
            save_code=True,
        )
    policy.learn(total_timesteps=total_timesteps,
                 callback=WandbCallback(verbose=2) if use_wandb else None)

    # Evaluate the agent
    perfs = Perfs.from_agent(policy, episodes=100, env_size=env_size,
                             net_arch=net_arch,
                             learning_rate=learning_rate,
                             steps=total_timesteps)

    if use_wandb:
        wandb.log({
            "general_env": perfs.general_env,
            "br_env": perfs.br_env,
            "general_br_freq": perfs.general_br_freq,
        })

    # Save the agent
    if save:
        name = (
            f"agents/ppo_{env_size}env_{total_timesteps}steps_{perfs.general_env * 1000:03.0f}gen_{perfs.br_env * 1000:03.0f}br_"
            f"{perfs.general_br_freq * 1000:03.0f}br_wrong_{bottom_right_odds}odds_{time.time():.0f}")
        perfs.info["file"] = name
        policy.save(name)
        json.dump(dataclasses.asdict(perfs), open(name + ".json", "w"))
        if verbose >= 1:
            print(f"Saved model to {name}")

    if verbose >= 1:
        print(perfs)

    return policy, perfs


@click.command()
@click.option("--steps", default=50_000, help="Number of steps to train the agent for")
@click.option("--br-odds", default=1, help="Odds of the goal being in the bottom right corner. 3 means three times more likely than not.")
@click.option("--n-agents", default=1, help="Number of agents to train")
@click.option("--n-envs", default=1, help="Number of environments to train on")
@click.option("--jobs", default=1, help="Number of jobs to run in parallel")
@click.option("--env-size", default=5, help="Size of the environment")
@click.option("-v", "--verbose", count=True, help="Verbosity level (0-2)")
@click.option(
    "--arch",
    default="30:10",
    callback=lambda ctx, param, value: tuple(map(int, value.split(":"))),
    type=str,
    help="Neural network architecture, as a string of integers separated by colons (ex. 30:10)",
)
def train(steps: int, br_odds: int, n_agents: int, n_envs: int, jobs: int, env_size: int,
            verbose: int,
          arch: tuple):
    """Train a PPO agent on the SimpleEnv environment"""

    def get():
        get_agent(br_odds, steps, n_envs, net_arch=arch, env_size=env_size, verbose=verbose)
        # Not returning anything to avoid pickling the agent for nothing
        # Joblib also throws an error if the agent is returned (cannot pickle _thread.lock objects)

    if n_agents > 1:
        Parallel(n_jobs=jobs)(delayed(get)() for _ in range(n_agents))
    else:
        get()


if __name__ == "__main__":
    train()
