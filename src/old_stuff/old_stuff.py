from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import einops
import gymnasium as gym
import numpy as np
import plotly.express as px
from stable_baselines3 import PPO

from utils import Trajectory


class BottomRightAgent:
    """Baseline agent that always goes to the bottom right corner."""

    def __init__(self, can_turn: bool = True):
        self.can_turn = can_turn

    def predict(self, obs, deterministic=True):
        assert len(obs.shape) == 3
        size = obs.shape[0]

        map = np.argmax(obs, axis=-1)  # (size, size)

        # Find where the agent is
        for i, j in np.ndindex(size, size):
            # Those numbers refer to OneHotFullObsWrapper.MAPPING
            # and OneHotFullObsWrapper.MAPPING_NO_TURN
            if map[i, j] >= 3:
                agent_pos = (i, j)
                break
        else:
            raise ValueError("Could not find the agent position")

        if self.can_turn:
            direction = map[agent_pos] - 3  # between 0 and 3: right, down, left, up

            LEFT = 0, None
            RIGHT = 1, None
            FORWARD = 2, None

            if direction == 0:
                if agent_pos[0] == size - 1:
                    return RIGHT
                return FORWARD
            elif direction == 1:
                if agent_pos[1] == size - 1:
                    return LEFT
                return FORWARD
            elif direction == 2:
                return LEFT
            elif direction == 3:
                return RIGHT
            else:
                raise ValueError(f"Invalid direction: {direction}")

        else:
            # Numbers from minigrid.DIR_TO_VEC
            if agent_pos[0] == size - 1:  # on right wall
                return 1, None  # go down
            return 0, None  # go right


@dataclass
class Perfs:
    br_env: float
    general_env: float
    general_br_freq: float
    env_size: int
    info: dict[str, float] = field(default_factory=dict)

    @classmethod
    def from_agent(
            cls,
            policy: PPO,
            episodes: int = 100,
            env_size: int = 5,
            **info,
    ):
        # Should be enough to reach the goal
        # Higher values make the evaluation much slower,
        # because of trajectories where the agent is stuck.
        max_len = env_size * 4
        br_success_rate = eval_agent(policy,
                                     random_goal_env(env_size, 1.0),
                                     episodes,
                                     episode_len=max_len)
        success_rate = eval_agent(policy, random_goal_env(env_size), episodes, episode_len=max_len)
        br_freq = eval_agent(
            policy,
            random_goal_env(env_size, 1.0),
            episodes,
            episode_len=max_len,
            end_condition=lambda locals_: locals_["env"].agent_pos == (env_size - 1, env_size - 1),
        )
        return cls(br_success_rate, success_rate, br_freq, env_size, info)


def eval_agent(
        policy: PPO,
        env: gym.Env = random_goal_env(),
        episodes: int = 100,
        episode_len: int = 100,
        plot: bool = False,
        end_condition: Callable[[dict], bool] | None = None,
) -> float:
    nb_success = 0
    fails = []
    success_imgs = []
    for _ in range(episodes):
        trajectory = Trajectory.from_policy(
            policy,
            env,
            max_len=episode_len,
            end_condition=end_condition,
            no_images=True,
        )

        if end_condition is None:
            success = trajectory.ended == "terminated"
        else:
            success = trajectory.ended == "condition"

        if success:
            nb_success += 1
            add_to = success_imgs
        else:
            add_to = fails

        # If the initial position is not in the list, add it
        if all(np.any(img != trajectory.images[0]) for img in add_to):
            add_to.append(trajectory.images[0])

    # show fails
    if plot and fails and success_imgs:
        print(f"Success rate: {nb_success / episodes:.2%}")
        for imgs, title in [(fails, "failed"), (success_imgs, "succeeded")]:
            prop = len(imgs) / (len(fails) + len(success_imgs))
            # Max 100 images, and pad to multiple of 10
            imgs = imgs[:100]
            imgs += [np.zeros_like(imgs[0])] * (10 - len(imgs) % 10)

            img = einops.rearrange(imgs, "(row col) h w c -> (row h) (col w) c", row=10)
            px.imshow(
                img,
                title=f"Positions where the agent {title} to reach the goal. {prop:.2%}",
            ).show()

    return nb_success / episodes
