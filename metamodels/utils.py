from __future__ import annotations

from math import ceil
from typing import Callable, Literal

import einops
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.subplots
import torch
import torch.nn as nn
from jaxtyping import Float
from torch import Tensor
from tqdm.autonotebook import tqdm

DataGenerator = Callable[
    [int], tuple[Float[Tensor, "batch dim"], Float[Tensor, "batch output_dim"]]
]

def batch_to_device(x_y, net: nn.Module):
    device = next(net.parameters()).device
    return x_y[0].to(device), x_y[1].to(device)

def get_accuracy(
    net: nn.Module,
    data_generator: DataGenerator,
    batch_size: int = 100,
    epochs: int = 100,
) -> float:
    correct = 0
    for _ in range(epochs):
        x, y = batch_to_device(data_generator(batch_size), net)
        y_pred = net(x)
        correct += ((y_pred > 0.5) == y).sum().item()
    return correct / (100 * 100)


def train(
    net: nn.Module,
    data_generator: DataGenerator,
    steps: int = 10_000,
    adversary: DataGenerator | None = None,
    lr: float = 1e-3,
    batch_size: int = 20,
    adv_odds: int = 10,
    wd: float = 0.0,
    loss_fn: nn.Module = nn.BCELoss(),
    log_every: int = 1000,
    reuse_perfs: bool = False,
    _perfs = ([], [])
):
    if not reuse_perfs:
        _perfs[0].clear()
        _perfs[1].clear()
    losses, maes = _perfs

    optimizer = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=wd)

    generators = [data_generator] * adv_odds
    if adversary is not None:
        generators.append(adversary)

    running_mae = 0

    for step in tqdm(range(1, steps + 1)):
        optimizer.zero_grad()
        batch = generators[step % len(generators)](batch_size)
        x, y = batch_to_device(batch, net)

        y_pred = net(x)
        loss = loss_fn(y_pred, y.float())
        loss.backward()
        optimizer.step()

        mae = (y_pred - y).abs().mean().item()
        running_mae = 0.95 * running_mae + 0.05 * mae
        losses.append(loss.item())
        maes.append(mae)

        if step % log_every == 0:
            print(f"Step {step}: running MAE {running_mae:.3f}")

    show_example(net, data_generator)

    # Plot losses
    # Use log scale
    df = pd.DataFrame({"loss": losses, "mae": maes})
    fig = px.line(df, y=df.columns, log_y=True, title="Losses")
    fig.show()


def show_size(n: nn.Module):
    print("Nb params:", sum(p.numel() for p in n.parameters() if p.requires_grad))


def show_example(net: nn.Module, data_generator: DataGenerator):
    # Show an example
    x, y = batch_to_device(data_generator(1), net)
    y_pred = net(x)
    print("Example input:\n", x[0], "\nGot:\n", y_pred[0], "\nExpected:\n", y[0])


class MLP(nn.Module):
    """A simple Multi-Layer Perceptron (MLP)"""

    def __init__(
        self, in_dim: int, *hidden_multipliers: int, out_dim: int | None = None
    ):
        super().__init__()
        dims = [in_dim] + [in_dim * m for m in hidden_multipliers] + [out_dim or in_dim]
        layers = [
            layer
            for dim_in, dim_out in zip(dims[:-1], dims[1:])
            for layer in [nn.Linear(dim_in, dim_out), nn.ReLU()]
        ]
        self.layers = nn.Sequential(*layers[:-1])

    def forward(
        self, x: Float[Tensor, "*batch d_model"]
    ) -> Float[Tensor, "*batch d_model"]:
        for layer in self.layers:
            x = layer(x)
        return x


class DotProductLayer(nn.Module):
    def __init__(
        self, in_dim: int, n_heads: int, d_head: int, out_dim: int | None = None
    ):
        super().__init__()
        self.in_dim = in_dim
        self.n_heads = n_heads
        self.out_dim = out_dim or in_dim
        self.d_head = d_head
        self.read_head = nn.Linear(in_dim, n_heads * d_head * 2)
        self.write_head = nn.Linear(n_heads, self.out_dim)

    def forward(
        self, x: Float[Tensor, "*batch in_dim"]
    ) -> Float[Tensor, "*batch out_dim"]:
        x = self.read_head(x)
        x = x.reshape(*x.shape[:-1], self.n_heads, self.d_head, 2)
        dot_products = einops.einsum(
            x[..., 0],
            x[..., 1],
            "... head dhead, ... head dhead -> ... head",
        )
        return self.write_head(dot_products)


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
        self, x: Float[Tensor, "batch token head_x_d_head"]
    ) -> Float[Tensor, "token all_token head d_head"]:
        """Utility function to convert the key/queries/values into a tensor with a head dimension."""
        return einops.rearrange(
            x, "batch token (head d_head) -> batch token head d_head", head=self.heads
        )

    def forward(
        self, x: Float[Tensor, "batch token d_model"]
    ) -> Float[Tensor, "batch token d_model"]:
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


class PoolingLayer(nn.Module):
    def __init__(self, kind: Literal["mean", "max", "flatten"]):
        super().__init__()
        self.kind = kind

    def forward(self, x: Float[Tensor, "batch token d_model"]):
        if self.kind == "mean":
            return x.mean(dim=1)
        elif self.kind == "max":
            return x.max(dim=1).values
        elif self.kind == "flatten":
            return x.flatten(1)
        else:
            raise ValueError(f"Unknown pooling kind {self.kind}")


class SkipSequential(nn.Sequential):
    def forward(self, x: Float[Tensor, "batch token d_model"]):
        for layer in self:
            x = layer(x) + x
        return x


class MatrixTokenizer(nn.Module):

    def forward(self, x: Float[Tensor, "batch row col"]):
        # Transpose the columns
        cols = x.transpose(-1, -2)
        # Concatenate the rows and columns
        out = torch.cat([x, cols], dim=-2)
        return out

class EmbeddingExtend(nn.Module):
    def __init__(self, extra_space: int):
        self.extra_space = extra_space
        super().__init__()

    def forward(self, x: Float[Tensor, "batch token d_model"]):
        *batch, tokens, d_model = x.shape
        # Extend the rows with zeros
        new_space = torch.zeros(*batch, tokens, self.extra_space)
        out = torch.cat([x, new_space], dim=-1)
        return out
