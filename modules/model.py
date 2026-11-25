"""Plain MLP mapping physical B-spline coefficients (+ mu) to Stokes
B-spline coefficients.

Every other piece of this codebase (data loading, the emulator's
encode/decode, the CNN/Transformer models) keeps each physical/Stokes
channel as its own tensor, precisely because concatenating channels that
each have their own independent coefficient count and knot placement would
produce a single axis that is *not* monotonic in depth or wavelength
(channel i's last coefficient sits right next to channel i+1's first, even
though they describe unrelated points on the grid, or even different
grids). An MLP, however, fundamentally needs one flat feature vector -- so
it is the one place in the codebase that flattens, and it does so itself,
internally, from the per-channel dict it's handed and back into one on the
way out, rather than requiring the rest of the pipeline to hand it a
pre-flattened (and therefore order-ambiguous) tensor.
"""
from __future__ import annotations

from typing import Dict, Sequence

import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(
        self,
        phys_names: Sequence[str],
        phys_sizes: Dict[str, int],
        stokes_names: Sequence[str],
        stokes_sizes: Dict[str, int],
        hidden_dims: Sequence[int] = (256, 256, 256),
        activation: type = nn.SiLU,
        dropout: float = 0.0,
        layer_norm: bool = False,
    ):
        super().__init__()
        self.phys_names = list(phys_names)
        self.phys_sizes = dict(phys_sizes)
        self.stokes_names = list(stokes_names)
        self.stokes_sizes = dict(stokes_sizes)

        in_dim = sum(self.phys_sizes.values()) + 1  # + 1 for mu
        out_dim = sum(self.stokes_sizes.values())

        dims = [in_dim, *hidden_dims]
        layers = []
        for d_in, d_out in zip(dims[:-1], dims[1:]):
            layers.append(nn.Linear(d_in, d_out))
            if layer_norm:
                layers.append(nn.LayerNorm(d_out))
            layers.append(activation())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(dims[-1], out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, coeffs: Dict[str, torch.Tensor], mu: torch.Tensor) -> Dict[str, torch.Tensor]:
        """coeffs: name -> (B, phys_sizes[name]), mu: (B,) or (B, 1) ->
        name -> (B, stokes_sizes[name])."""
        if mu.dim() == 1:
            mu = mu.unsqueeze(-1)
        x = torch.cat([coeffs[name] for name in self.phys_names] + [mu], dim=-1)
        y = self.net(x)
        out = {}
        idx = 0
        for name in self.stokes_names:
            n = self.stokes_sizes[name]
            out[name] = y[:, idx: idx + n]
            idx += n
        return out


class _ResidualBlock(nn.Module):
    """Pre-norm residual block ``x + W2 act(W1 LN(x))``. ``W2`` starts at
    zero, so every block starts as the identity and the network as a single
    linear map, which keeps deep stacks easy to optimize."""

    def __init__(self, width: int, expansion: int, activation: type, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.fc1 = nn.Linear(width, expansion * width)
        self.act = activation()
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.fc2 = nn.Linear(expansion * width, width)
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.fc2(self.drop(self.act(self.fc1(self.norm(x)))))


class ResidualMLP(nn.Module):
    """Drop-in replacement for :class:`MLP` (same per-channel dict
    interface): a linear embedding to ``width``, ``num_blocks`` pre-norm
    residual blocks with hidden size ``expansion * width``, and a final
    LayerNorm + linear head. The skip connections give every block a direct
    gradient path, so depth can be increased without the optimization
    difficulties of a plain deep MLP."""

    def __init__(
        self,
        phys_names: Sequence[str],
        phys_sizes: Dict[str, int],
        stokes_names: Sequence[str],
        stokes_sizes: Dict[str, int],
        width: int = 256,
        num_blocks: int = 4,
        expansion: int = 2,
        activation: type = nn.SiLU,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.phys_names = list(phys_names)
        self.phys_sizes = dict(phys_sizes)
        self.stokes_names = list(stokes_names)
        self.stokes_sizes = dict(stokes_sizes)

        in_dim = sum(self.phys_sizes.values()) + 1  # + 1 for mu
        out_dim = sum(self.stokes_sizes.values())
        self.embed = nn.Linear(in_dim, width)
        self.blocks = nn.Sequential(*[_ResidualBlock(width, expansion, activation, dropout)
                                      for _ in range(num_blocks)])
        self.head = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, out_dim))

    def forward(self, coeffs: Dict[str, torch.Tensor], mu: torch.Tensor) -> Dict[str, torch.Tensor]:
        if mu.dim() == 1:
            mu = mu.unsqueeze(-1)
        x = torch.cat([coeffs[name] for name in self.phys_names] + [mu], dim=-1)
        y = self.head(self.blocks(self.embed(x)))
        out = {}
        idx = 0
        for name in self.stokes_names:
            n = self.stokes_sizes[name]
            out[name] = y[:, idx: idx + n]
            idx += n
        return out
