#!/usr/bin/env python3.11
import time

import click
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from main import wrap_env, SimpleEnv, eval_agent, uniform_distribution

random_goal_env = lambda size: wrap_env(
    SimpleEnv(size=size, goal_pos=None, agent_start_pos=None, render_mode="rgb_array"))

br_env = lambda size: wrap_env(
    SimpleEnv(
        size=size,
        goal_pos=(size - 2, size - 2),
        agent_start_pos=None,
        render_mode="rgb_array",
    ))


def get_agent(
        bottom_right_odds: int,
        steps: int = 50_000,
        n_envs: int = 1,
        net_arch: tuple = (30, 10),
        env_size: int = 5,
        save: bool = True,
):
    # Define the training environment
    goal_distrib = uniform_distribution((env_size - 1, env_size - 1))
    # There are (envsize-2)**2-1 other positions
    goal_distrib[env_size - 2, env_size - 2] = (bottom_right_odds * (env_size - 2)**2 - 1)
    env = make_vec_env(
        lambda: wrap_env(
            SimpleEnv(
                size=env_size,
                goal_pos=goal_distrib,
                # goal_pos=(-2, -2),
                agent_start_pos=None,
                # render_mode='rgb_array'
            )),
        n_envs=n_envs,
    )

    # Define the policy network
    policy = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        # learning_rate=0.01,
        learning_rate=lambda f: 0.01 * f,
        policy_kwargs=dict(net_arch=net_arch),
        n_steps=2000 // n_envs,
        batch_size=100,
        n_epochs=40,
        gamma=1,
        tensorboard_log="run_logs",
        device="cpu",
    )
    # Train the agent
    policy.learn(total_timesteps=steps)

    # Evaluate the agent
    # br_success_rate = eval_agent(policy, base_env(env_size), 1000)
    # success_rate = eval_agent(policy, general_env(env_size), 1000)
    # print("Bottom right success rate:", br_success_rate)
    # print("Success rate:", success_rate)

    # Save the agent
    if save:
        # name = f"agents/ppo_{steps}steps_{success_rate * 1000:03.0f}gen_{br_success_rate * 1000:03.0f}br_{bottom_right_odds}odds_{time.time():.0f}"
        name = f"agents/ppo_{steps}steps_{bottom_right_odds}odds_{time.time():.0f}"
        policy.save(name)
        print(f"Saved model to {name}")

    return policy


@click.command()
@click.option("--steps", default=50_000, help="Number of steps to train the agent for")
@click.option("--br-odds", default=1, help="Odds of the goal being in the bottom right corner")
@click.option("--n-agents", default=1, help="Number of agents to train")
@click.option("--n-envs", default=1, help="Number of environments to train on")
@click.option("--jobs", default=1, help="Number of jobs to run in parallel")
@click.option(
    "--arch",
    default="30:10",
    callback=lambda ctx, param, value: tuple(map(int, value.split(":"))),
    type=str,
    help="Neural network architecture, as a string of integers separated by colons (ex. 30:10)",
)
def train(steps: int, br_odds: int, n_agents: int, n_envs: int, jobs: int, arch: tuple):
    """Train a PPO agent on the SimpleEnv environment"""
    if n_agents > 1:
        from joblib import Parallel, delayed

        Parallel(n_jobs=jobs)(delayed(get_agent)(br_odds, steps, n_envs, net_arch=arch)
                              for _ in range(n_agents))
    else:
        get_agent(br_odds, steps, n_envs)


if __name__ == "__main__":
    train()
