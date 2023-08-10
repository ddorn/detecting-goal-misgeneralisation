from __future__ import annotations

import math
from itertools import chain
from pprint import pprint
from typing import Callable

import gymnasium as gym
import torch
from jaxtyping import Float, Int
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import Tensor
from torch import nn


__all__ = [
    "make_mlp",
    "L1WeightDecay",
    "SwitchMLP",
    "SwitchActorCriticPolicy",
    "SwitchPolicyValueNetwork",
    "NOPFeaturesExtractor",
    "CustomFeaturesExtractor",
]


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
                # noinspection PyTypeChecker
                if param.grad is None or torch.all(param.grad == 0.0):
                    param.grad = self.regularize(param)
        else:
            for name, param in self.module.named_parameters():
                # noinspection PyTypeChecker
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

        super().__init__(observation_space, features_dim=math.prod(observation_space.shape[:-1]))

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

        kwargs.setdefault("features_extractor_class", NOPFeaturesExtractor)
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            # features_extractor_kwargs=dict(features_dim=observation_space.nvec.size),
            # Pass remaining arguments to base class
            *args,
            **kwargs,
        )

    def _build_mlp_extractor(self) -> None:
        print(self.observation_space)
        print("Features dim", self.features_dim)
        self.mlp_extractor = SwitchPolicyValueNetwork(
            self.features_dim, self.observation_space, **self.arch_kwargs)
