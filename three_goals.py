from __future__ import annotations

import math
from functools import partial
from itertools import chain
from pprint import pprint
from random import choice, choices, sample
from typing import Any, Callable, Self, Literal

import gymnasium as gym
import numpy as np
import pygame
import pygame.gfxdraw
import torch
from gymnasium import ObservationWrapper, Wrapper
from gymnasium.core import WrapperObsType, ObsType
from jaxtyping import Float, Int
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import Tensor
from torch import nn

from main import GridEnv, Cell, FlatOneHotWrapper, Distribution, Pos, sample_distribution


class ThreeGoalsEnv(GridEnv):
    GOAL_RED = Cell("r", "#F44336", manual=True)
    GOAL_BLUE = Cell("b", "#2196F3", manual=True)
    GOAL_GREEN = Cell("g", "#4CAF50", manual=True)

    GOAL_CELLS = [GOAL_RED, GOAL_GREEN, GOAL_BLUE]
    ALL_CELLS = GridEnv.ALL_CELLS + GOAL_CELLS

    def __init__(self,
                 size: int = 4,
                 *,
                 true_goal: Literal['red', 'green', 'blue'] | None = None,
                 agent_pos: Distribution[Pos] | None = None,
                 red_pos: Distribution[Pos] | None = None,
                 green_pos: Distribution[Pos] | None = None,
                 blue_pos: Distribution[Pos] | None = None,
                 ):
        self.red_pos_dist = red_pos
        self.green_pos_dist = green_pos
        self.blue_pos_dist = blue_pos

        self.true_goal_init = true_goal
        self.true_goal_idx = -42
        self.goal_positions = [(-1, -1)] * 3

        super().__init__(agent_pos, size, size)

    @property
    def true_goal(self) -> Cell:
        return self.GOAL_CELLS[self.true_goal_idx]

    def new_goal(self) -> int:
        if self.true_goal_init is None:
            return self.np_random.choice(len(self.GOAL_CELLS))
        elif self.true_goal_init == "red":
            return 0
        elif self.true_goal_init == "green":
            return 1
        elif self.true_goal_init == "blue":
            return 2
        else:
            raise ValueError(f"Invalid true_goal_init: {self.true_goal_init}")

    def make_grid(self):
        super().make_grid()
        dists = [self.red_pos_dist, self.green_pos_dist, self.blue_pos_dist]
        self.goal_positions = [
            self.place_obj(goal, dist)
            for goal, dist in zip(self.GOAL_CELLS, dists)
        ]
        self.true_goal_idx = self.new_goal()

    def handle_object(self, obj: Cell) -> tuple[bool, float, bool]:
        if obj is self.true_goal:
            return True, 1, True
        else:
            return True, 0, True

    def render_extra(self, img: pygame.Surface, resolution: int):
        # Add a star on the true goal
        x, y = self.goal_positions[self.true_goal_idx]
        cx = int((x + 0.5) * resolution) - 1  # just for prettiness
        cy = int((y + 0.5) * resolution) + 1
        radius = int(resolution / 3)
        # 5 points star
        angle = np.pi * 4 / 5
        shift = -np.pi / 2  # rotate 90 degrees, so that the star is pointing up
        points = [(cx + radius * np.cos(angle * i + shift), cy + radius * np.sin(angle * i + shift))
                  for i in range(5)]

        pygame.gfxdraw.aapolygon(img, points, (255, 255, 255))
        pygame.gfxdraw.filled_polygon(img, points, (255, 255, 255))

    @classmethod
    def constant(cls, size=4, true_goal: Distribution[str] = None) -> Self:
        """Return an environment that is always the same, even after reset."""
        true = sample_distribution(true_goal, choice(["red", "green", "blue"]))
        positions = [(x, y) for x in range(size) for y in range(size)]
        agent, red, green, blue = sample(positions, k=4)
        return cls(size, true_goal=true, agent_pos=agent, red_pos=red, green_pos=green, blue_pos=blue)

    @classmethod
    def interesting(cls, size: int = 4, n_random: int = 3, wrappers: list[Wrapper] | None = None) -> list[Self]:
        agent_pos = (0, 0)
        red_green_blue = [
            [(0, 1), (1, 0), (1, 1)],
            [(0, 1), (1, 1), (0, 2)],
            [(0, size-1), (size-1, 0), (size-1, size-1)],
            [(0, 1), (0, 2), (0, 3)],
        ]
        envs = [
            cls(size, true_goal="blue", agent_pos=agent_pos, red_pos=red_pos, green_pos=green_pos, blue_pos=blue_pos)
            for red_pos, green_pos, blue_pos in red_green_blue
        ] + [
            cls(size) for _ in range(n_random)
        ]
        if wrappers is None:
            wrappers = [FlatOneHotWrapper, AddTrueGoalWrapper]
        for wrapper in wrappers:
            envs = [wrapper(env) for env in envs]
        return envs


class AddTrueGoalWrapper(ObservationWrapper):
    unwrapped: ThreeGoalsEnv

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.observation_space = gym.spaces.Dict({
            "obs": env.observation_space,
            "switch": gym.spaces.Discrete(len(self.GOAL_CELLS)),
        })

    def observation(self, obs: WrapperObsType) -> ObsType:
        return {
            "obs": obs,
            "switch": self.unwrapped.true_goal_idx,
        }


class ColorBlindWrapper(ObservationWrapper):
    def __init__(self, env: gym.Env,
                 n_scenarios: int,
                 scenarios: Callable[[int], tuple[int, list[int]]],
                 disabled: bool = False,
                 ):
        super().__init__(env)
        obs = env.observation_space
        assert isinstance(obs, gym.spaces.Dict)
        assert set(obs.keys()) >= {"obs", "switch"}, f"AddTrueGoalWrapper must be applied first, {obs.keys()}"
        assert isinstance(obs["obs"], gym.spaces.MultiDiscrete)
        obs_size = obs["obs"].nvec
        flat_obs_size = obs_size.sum()
        self.observation_space = gym.spaces.Dict({
            **obs.spaces,
            "obs": gym.spaces.MultiBinary(flat_obs_size),
            "switch": gym.spaces.Discrete(n_scenarios),
        })
        self.n_cells = obs_size.max()
        self.scenarios = scenarios
        self.n_scenarios = n_scenarios
        self.disabled = disabled

    def observation(self, obs: WrapperObsType) -> ObsType:
        grid = obs["obs"]
        switch = obs["switch"]
        # I want a function that takes the current scenario
        # and returns the output scenario + which colors are indistinguishable.
        # Good.

        # Merge stuff
        w, h = grid.shape
        one_hot = np.zeros((w, h, self.n_cells), dtype=bool)
        for x in range(w):
            for y in range(h):
                one_hot[x, y, grid[x, y]] = True

        # Make indistinguishable
        new_switch, indistinguishable = self.scenarios(switch)
        if not self.disabled and indistinguishable:
            indistinguishable = list(indistinguishable)
            perceived_same = one_hot[..., indistinguishable].any(axis=-1)
            one_hot[..., indistinguishable] = perceived_same[..., None]

        return {
            **obs,
            "obs": one_hot.flatten(),
            "switch": new_switch,
        }

    @classmethod
    def merged(cls, env: gym.Env, *cells: Cell, disabled: bool = False) -> Self:
        """Merge the given cells so that all the agents see them as the same value (all 1s)."""
        assert isinstance(env.unwrapped, ThreeGoalsEnv), f"Can only merge cells in {ThreeGoalsEnv}, got {env.unwrapped}"
        assert all(cell in env.GOAL_CELLS for cell in cells), f"Cells must be in {env.GOAL_CELLS}, got {cells}"

        merged_ids = [i for i, goal in enumerate(env.ALL_CELLS) if goal in cells]
        non_merged = [cell for cell in env.GOAL_CELLS if cell not in cells]
        n_scenarios = len(non_merged) + 1

        def _scenarios(switch: int) -> tuple[int, list[int]]:
            goal = env.GOAL_CELLS[switch]
            if goal in cells:
                return 0, merged_ids
            else:
                return non_merged.index(goal) + 1, merged_ids

        return cls(env, n_scenarios, _scenarios, disabled=disabled)

    @classmethod
    def merged_multi(cls, env, *cells: list[Cell], disabled: bool = False) -> Self:
        """Create multiple scenarios, one for each list of cells, where the list of cells is merged."""
        def _scenarios(switch: int) -> tuple[int, list[int]]:
            cell = env.GOAL_CELLS[switch]
            possible = [i for i, merged_cells in enumerate(cells) if cell in merged_cells]
            if possible:
                s = choice(possible)
                return s, [i for i, goal in enumerate(env.ALL_CELLS) if goal in cells[s]]
            else:
                raise ValueError(f"Cell {cell} not in any of the given cells {cells}")

        return cls(env, len(cells), _scenarios, disabled=disabled)


def make_mlp(*dims, add_act_before: bool = False, add_act_after: bool = False,
             activation=nn.Tanh):
    """Create a multi-layer perceptron.

    Args:
        *dims: The dimensions of the MLP. The first is the input dimension, the last is the output dimension.
        add_act_before: Whether to add an activation before the first layer.
        add_act_after: Whether to add an activation after the last layer.
        activation: The activation function to use.
    """
    layers = [
        layer
        for dim_in, dim_out in zip(dims[:-1], dims[1:])
        for layer in (nn.Linear(dim_in, dim_out), activation())
    ]

    if not layers:
        return nn.Sequential()

    if add_act_before:
        layers.insert(0, activation())

    if not add_act_after:
        layers.pop()

    return nn.Sequential(*layers)


class L1WeightDecay(torch.nn.Module):
    # Adapted from https://github.com/szymonmaszke/torchlayers/blob/master/torchlayers/regularization.py#L150
    def __init__(self, module, weight_decay: float, name: str = None):
        super().__init__()
        assert weight_decay >= 0.0
        self.module = module
        self.weight_decay = weight_decay
        self.name_filter = name

        self.hook = self.module.register_full_backward_hook(self._weight_decay_hook)

    def remove(self):
        self.hook.remove()

    def _weight_decay_hook(self, *_):
        if self.weight_decay <= 0.0:
            return
        if self.name_filter is None:
            for param in self.module.parameters():
                if param.grad is None or torch.all(param.grad == 0.0):
                    param.grad = self.regularize(param)
        else:
            for name, param in self.module.named_parameters():
                if self.name_filter in name and (
                        param.grad is None or torch.all(param.grad == 0.0)
                ):
                    param.grad = self.regularize(param)

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def extra_repr(self) -> str:
        representation = f"weight_decay={self.weight_decay}"
        if self.name_filter is not None:
            representation += f", name_filter={self.name_filter}"
        return representation

    def regularize(self, parameter):
        return parameter.data.sign() * self.weight_decay


class SwitchMLP(nn.Module):
    def __init__(self, in_dim: int, hidden: tuple[int, ...], out_dim: int,
                 n_switches: int, switched_layer: int, activation=nn.Tanh,
                 ):
        super().__init__()

        dims = [in_dim, *hidden, out_dim]
        print(dims)
        self.n_switches = n_switches
        self.switched_layer = switched_layer
        self.dims = dims

        # Layer that get switched is dim[switched_layer] -> dim[switched_layer+1]
        self.switches = nn.ModuleList([
            nn.Linear(dims[switched_layer], dims[switched_layer + 1])
            for _ in range(n_switches)])

        # So pre-switch is dim[0] -> ... -> dim[switched_layer]
        self.pre_switch = make_mlp(*dims[:switched_layer + 1], add_act_after=True, activation=activation)

        # Post-switch is dim[switched_layer+1] -> ... -> dim[-1]
        self.post_switch = make_mlp(*dims[switched_layer + 1:], add_act_before=True, add_act_after=True,
                                    activation=activation)

    def forward(self, x: Tensor | dict, switch: int = None) -> Tensor:
        if switch is None:
            assert isinstance(x, dict), f"switch is None, but x is not a dict: {x}"
            assert x.keys() >= {"obs", "switch"}, f"Invalid keys: {x.keys()}"
            switch = x["switch"]
            x = x["obs"]

        x: Float[Tensor, "batch in_dim"]

        switch = switch.argmax(dim=-1).squeeze(-1)
        switch: Int[Tensor, "batch"]

        x = self.pre_switch(x)
        # To do the forward pass, through the switch, we need to take into
        # account the batch dimension. We run a forward pass for each switch
        # and then select the right one.
        all_x = torch.stack([layer(x) for layer in self.switches], dim=1)
        x = all_x[torch.arange(len(x)), switch]
        x = self.post_switch(x)
        return x

    def mk_net(self, switch: int) -> nn.Module:
        return nn.Sequential(self.pre_switch, self.switches[switch], self.post_switch)

    def mk_nets(self) -> list[nn.Module]:
        return [self.mk_net(switch) for switch in range(self.n_switches)]


class CustomFeaturesExtractor(BaseFeaturesExtractor):

    def __init__(self, observation_space: gym.spaces.MultiBinary):
        assert isinstance(observation_space, gym.spaces.MultiBinary), observation_space
        assert len(observation_space.shape) == 3, observation_space.shape

        super().__init__(observation_space, features_dim=np.prod(observation_space.shape[:-1]))

    def forward(self, observations: Tensor) -> Tensor:
        # We transform the one-hot vector into a discrete observation
        # and flatten the map into a vector.
        features = observations.argmax(dim=-1)
        features = features.reshape(features.shape[0], -1)
        return features


class NOPFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.MultiBinary):
        # Hack: we need to pass a features_dim > 0, but we don't actually use it.
        # Also, it is hard to compute in general. features_dim needs to be an integer,
        # so if someone tries to use features_dim, they will get an explicit error somewhere,
        # because we set it to a float, instead of silently failing.
        # noinspection PyTypeChecker
        super().__init__(observation_space, features_dim=1.42)

    def forward(self, observations: Tensor) -> Tensor:
        return observations


class SwitchPolicyValueNetwork(nn.Module):
    """
    Custom network for policy and value function.
    It receives as input the features extracted by the features extractor.

    :param feature_dim: dimension of the features extracted with the features_extractor (e.g. features from a CNN)
    :param last_layer_dim_pi: (int) number of units for the last layer of the policy network
    :param last_layer_dim_vf: (int) number of units for the last layer of the value network
    """

    def __init__(
            self,
            feature_dim: int,
            observation_space: gym.Space,
            use_separate_networks: bool = False,
            l1_reg: float = 0.0,
            **arch_kwargs,
    ):
        print("SwitchPolicyValueNetwork init")
        print("feature_dim", feature_dim)
        print("observation_space", observation_space)
        pprint(arch_kwargs)
        assert isinstance(observation_space, gym.spaces.Dict), observation_space
        assert observation_space.keys() == {'obs', 'switch'}, observation_space.keys()
        assert isinstance(observation_space['obs'], gym.spaces.MultiBinary), observation_space['obs']
        assert isinstance(observation_space['switch'], gym.spaces.Discrete), observation_space['switch']

        super().__init__()

        self.use_separate_networks = use_separate_networks
        self.observation_space = observation_space
        input_dim = math.prod(observation_space['obs'].shape)

        # Policy network
        policy_net = SwitchMLP(in_dim=input_dim, **arch_kwargs)
        self.policy_net = L1WeightDecay(policy_net, l1_reg)

        # Value network
        if use_separate_networks:
            value_net = SwitchMLP(in_dim=input_dim, **arch_kwargs)
            self.value_net = L1WeightDecay(value_net, l1_reg)
        else:
            self.value_net = self.policy_net

        # IMPORTANT:
        # Save output dimensions, it's used by SB3 to create the distributions
        self.latent_dim_pi = self.policy_net.module.dims[-1]
        self.latent_dim_vf = self.value_net.module.dims[-1]

    def forward(self, features: Tensor) -> tuple[Tensor, Tensor]:
        """
        :return: (th.Tensor, th.Tensor) latent_policy, latent_value of the specified network.
            If all layers are shared, then ``latent_policy == latent_value``
        """
        if self.use_separate_networks:
            return self.forward_actor(features), self.forward_critic(features)

        out = self.forward_actor(features)
        return out, out

    def forward_actor(self, features: Tensor) -> Tensor:
        return self.policy_net(features)

    def forward_critic(self, features: Tensor) -> Tensor:
        return self.value_net(features)

    # Convenience methods to access the weights of the network.

    @property
    def switch_weights(self) -> Float[Tensor, "switch out_dim in_dim"]:
        """Returns the weights of the switch layers in one tensor."""
        return torch.stack([layer.weight for layer in self.policy_net.module.switches]).detach().cpu()

    @property
    def switch_biases(self) -> Float[Tensor, "switch out_dim"]:
        """Returns the biases of the switch layers in one tensor."""
        return torch.stack([layer.bias for layer in self.policy_net.module.switches]).detach().cpu()

    @property
    def weights(self) -> Float[Tensor, "layer out_dim in_dim"]:
        """Returns the weights of the non-switch layers of the policy network in one tensor."""
        layers = chain(self.policy_net.module.pre_switch, self.policy_net.module.post_switch)
        return torch.stack([layer.weight for layer in layers]).detach().cpu()

    @property
    def biases(self) -> Float[Tensor, "layer out_dim"]:
        """Returns the biases of the non-switch layers of the policy network in one tensor."""
        layers = chain(self.policy_net.module.pre_switch, self.policy_net.module.post_switch)
        return torch.stack([layer.bias for layer in layers]).detach().cpu()


class SwitchActorCriticPolicy(ActorCriticPolicy):
    features_dim: int

    def __init__(
            self,
            observation_space: gym.spaces.Space,
            action_space: gym.spaces.Space,
            lr_schedule: Callable[[float], float],
            *args,
            **kwargs,
    ):
        print("SwitchActorCriticPolicy init")
        pprint(kwargs)
        print("Observation space", observation_space)
        print("Action space", action_space)
        self.arch_kwargs = kwargs.pop("arch_kwargs", {})
        print("Arch kwargs")
        pprint(self.arch_kwargs)

        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            # features_extractor_kwargs=dict(features_dim=observation_space.nvec.size),
            # Pass remaining arguments to base class
            features_extractor_class=NOPFeaturesExtractor,
            *args,
            **kwargs,
        )

    def _build_mlp_extractor(self) -> None:
        print(self.observation_space)
        print("Features dim", self.features_dim)
        self.mlp_extractor = SwitchPolicyValueNetwork(
            self.features_dim, self.observation_space, **self.arch_kwargs)


if __name__ == "__main__":

    env = lambda: (
        AddTrueGoalWrapper(FlatOneHotWrapper(ThreeGoalsEnv(None, 4)))
    )
    print(env())
    print(env().observation_space)
    print(env().action_space)

    net = SwitchMLP(4, (2, 2), 2, 2, 0)
    print(net)
    net(torch.ones(1, 4), torch.tensor([0, 1]))

    if int() == 0:
        policy = PPO(
            SwitchActorCriticPolicy,
            make_vec_env(env, n_envs=10, seed=42),
            policy_kwargs=dict(
                arch_kwargs=dict(
                    switched_layer=0,
                    hidden=[64],
                    out_dim=64,
                    n_switches=3,
                ),
            ),
            verbose=2,
            n_epochs=40,
            n_steps=8_000 // 10,
            batch_size=400,
            learning_rate=0.001,
            # policy_kwargs=policy_kwargs,  # optimizer_kwargs=dict(weight_decay=weight_decay)),
            # arch_kwargs=dict(net_arch=net_arch, features_extractor_class=BaseFeaturesExtractor),
            # tensorboard_log="run_logs",
        )
        policy.learn(total_timesteps=100_000)
