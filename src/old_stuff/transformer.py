from __future__ import annotations

import math
from typing import Callable, Tuple
from typing import Type

import einops
import numpy as np
import torch
import torch as th
from gymnasium import spaces
from jaxtyping import Float, Int
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import Tensor
from torch import nn


class MultiAttention(nn.Module):

    def __init__(
        self,
        d_model: int,
        heads: int = 8,
        d_head: int = 64,
    ):
        """
        A simple multi-head attention layer without causal mask.

        Args:
            d_model: the input dimensions of the different tensors.
            heads: number of independent attention heads.
            d_head: dimension of each head.
        """

        super().__init__()
        self.d_model = d_model
        self.heads = heads
        self.d_head = d_head
        # Read d_model to d_head, for each head
        self.queries = nn.Linear(d_model, heads * d_head)
        self.keys = nn.Linear(d_model, heads * d_head)
        self.values = nn.Linear(d_model, heads * d_head)
        # Write from d_head to d_model
        self.out = nn.Linear(heads * d_head, d_model)

    def separate_heads(
        self,
        x: Float[Tensor,
                 "batch token head_x_d_head"]) -> Float[Tensor, "token all_token head d_head"]:
        """Utility function to convert the key/queries/values into a tensor with a head dimension."""
        return einops.rearrange(x,
                                "batch token (head d_head) -> batch token head d_head",
                                head=self.heads)

    def forward(self, x: Float[Tensor,
                               "batch token d_model"]) -> Float[Tensor, "batch token d_model"]:
        qs = self.queries(x)  # (batch, token, heads * d_head)
        q = self.separate_heads(qs)  # (batch, token, head, d_head)

        ks = self.keys(x)
        k = self.separate_heads(ks)

        vs = self.values(x)
        v = self.separate_heads(vs)

        scores = einops.einsum(
            q,
            k,
            "batch query head d_head, batch key head d_head -> batch head query key",
        )
        pattern = (scores / np.sqrt(self.d_head)).softmax(-1)

        weighted = einops.einsum(
            pattern,
            v,
            "batch head query key, batch key head d_head -> batch query head d_head",
        ).flatten(-2)

        return self.out(weighted)


class MLP(nn.Sequential):

    def __init__(
        self,
        d_model: int,
        hidden_multiplier: int = 4,
        activation: Type[nn.Module] = nn.ReLU,
    ):
        """
        A fully connected feed-forward network with one hidden layer.

        Args:
            d_model: input and output dimension.
            hidden_multiplier: the hidden layer will have `hidden_multiplier * d_model` units.
        """
        super().__init__(
            nn.Linear(d_model, hidden_multiplier * d_model),
            activation(),
            nn.Linear(hidden_multiplier * d_model, d_model),
        )


class Transformer(nn.Module):

    def __init__(
        self,
        vocab_size_in: int,
        context_size: int,
        d_model: int = 64,
        layers: int = 4,
        heads: int = 4,
        d_head: int = 16,
    ):
        """A simple transformer with no unembedding"""
        super().__init__()

        self.vocab_size_in = vocab_size_in
        self.context_size = context_size
        self.d_model = d_model
        self.layers = layers
        self.heads = heads
        self.d_head = d_head

        self.embedding = nn.Embedding(vocab_size_in, d_model)
        self.positional_encoding = nn.Embedding(context_size, d_model)

        self.components = nn.ModuleList([
            mod for _layer in range(layers)
            for mod in [MultiAttention(d_model, heads, d_head),
                        MLP(d_model)]
        ])

    def forward(self, x: Int[Tensor, "batch token"]) -> Float[Tensor, "batch vocab_out"]:
        stream = self.embedding(x) + self.positional_encoding(torch.arange(x.shape[-1]))

        for module in self.components:
            # Residual connection
            stream = module(stream) + stream

        # return stream[:, 0]
        return stream.flatten(-2)
        # return stream.mean(dim=-2)  # Average over the tokens


class CustomFeaturesExtractor(BaseFeaturesExtractor):

    def __init__(self, observation_space: spaces.MultiBinary):
        assert isinstance(observation_space, spaces.MultiBinary), observation_space
        assert len(observation_space.shape) == 3, observation_space.shape

        super().__init__(observation_space, features_dim=math.prod(observation_space.shape[:-1]))

    def forward(self, observations: th.Tensor) -> th.Tensor:
        # We transform the one-hot vector into a discrete observation
        # and flatten the map into a vector.
        features = observations.argmax(dim=-1)
        features = features.reshape(features.shape[0], -1)
        return features


class CustomNetwork(nn.Module):
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
        observation_space: spaces.MultiBinary,
        use_separate_networks: bool = False,
        **arch_kwargs,
    ):
        assert isinstance(observation_space, spaces.MultiBinary), observation_space
        super().__init__()

        self.use_separate_networks = use_separate_networks
        self.observation_space = observation_space
        vocab_size_in = observation_space.shape[-1]

        # Policy network
        self.policy_net = Transformer(vocab_size_in, feature_dim, **arch_kwargs)
        # Value network
        if use_separate_networks:
            self.value_net = Transformer(vocab_size_in, feature_dim, **arch_kwargs)
        else:
            self.value_net = self.policy_net

        # IMPORTANT:
        # Save output dimensions, used to create the distributions
        self.latent_dim_pi = self.policy_net.d_model * feature_dim
        self.latent_dim_vf = self.value_net.d_model * feature_dim

    def forward(self, features: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        """
        :return: (th.Tensor, th.Tensor) latent_policy, latent_value of the specified network.
            If all layers are shared, then ``latent_policy == latent_value``
        """
        if self.use_separate_networks:
            return self.forward_actor(features), self.forward_critic(features)

        out = self.forward_actor(features)
        return out, out

    def forward_actor(self, features: th.Tensor) -> th.Tensor:
        return self.policy_net(features)

    def forward_critic(self, features: th.Tensor) -> th.Tensor:
        return self.value_net(features)


class CustomActorCriticPolicy(ActorCriticPolicy):

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Callable[[float], float],
        *args,
        **kwargs,
    ):
        assert isinstance(observation_space, spaces.MultiBinary), observation_space
        self.arch_kwargs = kwargs.pop("arch", {})

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
        assert isinstance(self.observation_space, spaces.MultiBinary), self.observation_space
        assert isinstance(self.features_dim, int), self.features_dim
        self.mlp_extractor = CustomNetwork(self.features_dim, self.observation_space,
                                           **self.arch_kwargs)
