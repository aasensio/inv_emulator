"""Sequence-aware encoder-decoder architectures for the B-spline coefficient
regression problem.

The plain :class:`~mlp_bspline.model.MLP` treats every input and output
coefficient as an unordered feature: nothing in its architecture reflects
the fact that, within each physical or Stokes channel, the B-spline
coefficients are *ordered* along depth/wavelength and describe an
underlying quantity that is fundamentally continuous and locally smooth. To
predict a smoothly varying Stokes profile it therefore has to learn, from
data alone and with no help from its inductive bias, that neighbouring
output coefficients should co-vary -- which is one plausible explanation
for the excess small-scale wiggle these emulators can show in Q/U/V.

Both architectures below take/return per-channel dicts of coefficient
tensors -- name -> ``(B, sizes[name])`` -- exactly like the rest of the
pipeline (:mod:`mlp_bspline.data`, :class:`~mlp_bspline.emulator.StokesEmulator`),
never a single tensor with channels concatenated along one axis. That
matters because each channel has its own independent B-spline coefficient
count and knot placement: concatenating them would produce an axis that
isn't monotonic in depth/wavelength (channel i's last coefficient sits
right next to channel i+1's first coefficient, even though they describe
unrelated, possibly non-adjacent points, or even different physical
grids). Keeping channels separate lets convolution/attention act only
along an axis where neighbour-hood is physically meaningful. Both are
drop-in replacements for each other and for :class:`~mlp_bspline.model.MLP`
(see :func:`build_model`), including inside
:class:`~mlp_bspline.emulator.StokesEmulator`, and are encoder-decoder in
shape because the number of input coefficients (per physical channel) and
output coefficients (per Stokes channel) generally differ.
"""
from __future__ import annotations

import math
from typing import Dict, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .model import MLP, ResidualMLP


def _num_groups(channels: int, max_groups: int = 8) -> int:
    """Largest group count <= max_groups that evenly divides channels (so
    GroupNorm works for any user-chosen channel width, not just multiples of
    8)."""
    for g in range(min(max_groups, channels), 0, -1):
        if channels % g == 0:
            return g
    return 1


# --------------------------------------------------------------------------- #
# CNN encoder-decoder
# --------------------------------------------------------------------------- #

class _ResidualConvBlock1d(nn.Module):
    """Two dilated ``same``-padded convs with a residual connection: acts
    only on a coefficient and its neighbours, biasing the block towards
    locally smooth, correlated outputs rather than independent ones."""

    def __init__(self, channels: int, kernel_size: int, dilation: int = 1, activation: type = nn.SiLU):
        super().__init__()
        padding = dilation * (kernel_size - 1) // 2
        groups = _num_groups(channels)
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation)
        self.norm1 = nn.GroupNorm(groups, channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation)
        self.norm2 = nn.GroupNorm(groups, channels)
        self.act = activation()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.act(self.norm1(self.conv1(x)))
        h = self.norm2(self.conv2(h))
        return self.act(x + h)


class _ChannelCNNEncoder(nn.Module):
    """1-D CNN over one channel's ordered coefficients, pooled to a
    fixed-size embedding (avg + max over the sequence)."""

    def __init__(self, channels: int, num_layers: int, kernel_size: int):
        super().__init__()
        self.stem = nn.Conv1d(1, channels, kernel_size, padding=kernel_size // 2)
        self.blocks = nn.ModuleList(
            [_ResidualConvBlock1d(channels, kernel_size, dilation=2 ** i) for i in range(num_layers)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, L) -> (B, 2 * channels)."""
        h = self.stem(x.unsqueeze(1))
        for block in self.blocks:
            h = block(h)
        return torch.cat([h.mean(dim=-1), h.amax(dim=-1)], dim=-1)


class _ChannelCNNDecoder(nn.Module):
    """Resamples a shared trunk feature map to this channel's exact
    coefficient count, then refines it with local convolutions."""

    def __init__(self, channels: int, out_len: int, num_layers: int, kernel_size: int):
        super().__init__()
        self.out_len = out_len
        self.blocks = nn.ModuleList(
            [_ResidualConvBlock1d(channels, kernel_size, dilation=2 ** i) for i in range(num_layers)]
        )
        self.out_conv = nn.Conv1d(channels, 1, kernel_size=1)

    def forward(self, trunk: torch.Tensor) -> torch.Tensor:
        """trunk: (B, channels, trunk_len) -> (B, out_len)."""
        h = F.interpolate(trunk, size=self.out_len, mode="linear", align_corners=False)
        for block in self.blocks:
            h = block(h)
        return self.out_conv(h).squeeze(1)


class CNNEncoderDecoder(nn.Module):
    """Per-channel 1-D CNN encoder -> shared latent bottleneck -> shared
    trunk -> per-channel 1-D CNN decoder.

    Each physical channel is encoded independently (its own conv stack) and
    pooled; the pooled embeddings plus mu are combined into a single latent
    vector (this is where cross-channel physics, e.g. temperature
    stratification informing the magnetic imprint, can mix). A shared
    decoder trunk expands that latent back into a feature sequence, which
    each Stokes channel resamples to its own coefficient count and refines
    with its own local convolutions -- giving every predicted coefficient a
    direct, weight-shared dependence on its neighbours.
    """

    def __init__(
        self,
        phys_names: Sequence[str],
        phys_sizes: Dict[str, int],
        stokes_names: Sequence[str],
        stokes_sizes: Dict[str, int],
        channels: int = 32,
        num_layers: int = 2,
        kernel_size: int = 5,
        latent_dim: int = 256,
        mu_embed_dim: int = 16,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.phys_names = list(phys_names)
        self.phys_sizes = dict(phys_sizes)
        self.stokes_names = list(stokes_names)
        self.stokes_sizes = dict(stokes_sizes)
        self.channels = channels

        self.phys_encoders = nn.ModuleDict(
            {name: _ChannelCNNEncoder(channels, num_layers, kernel_size) for name in self.phys_names}
        )
        self.mu_encoder = nn.Sequential(nn.Linear(1, mu_embed_dim), nn.SiLU())

        bottleneck_in = 2 * channels * len(self.phys_names) + mu_embed_dim
        self.bottleneck = nn.Sequential(
            nn.Linear(bottleneck_in, latent_dim),
            nn.SiLU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(latent_dim, latent_dim),
            nn.SiLU(),
        )

        base_len = 4
        max_out_len = max(self.stokes_sizes.values())
        num_upsamples = max(1, math.ceil(math.log2(max(max_out_len / base_len, 1))))
        self.base_len = base_len
        self.trunk_proj = nn.Linear(latent_dim, channels * base_len)
        self.trunk_upsamples = nn.ModuleList(
            [nn.ConvTranspose1d(channels, channels, kernel_size=4, stride=2, padding=1)
             for _ in range(num_upsamples)]
        )
        self.trunk_act = nn.SiLU()

        self.stokes_decoders = nn.ModuleDict(
            {
                name: _ChannelCNNDecoder(channels, self.stokes_sizes[name], num_layers, kernel_size)
                for name in self.stokes_names
            }
        )

    def forward(self, coeffs: Dict[str, torch.Tensor], mu: torch.Tensor) -> Dict[str, torch.Tensor]:
        """coeffs: name -> (B, phys_sizes[name]), mu: (B,) or (B, 1) ->
        name -> (B, stokes_sizes[name])."""
        if mu.dim() == 1:
            mu = mu.unsqueeze(-1)

        enc = [self.phys_encoders[name](coeffs[name]) for name in self.phys_names]
        enc.append(self.mu_encoder(mu))
        z = self.bottleneck(torch.cat(enc, dim=-1))

        trunk = self.trunk_proj(z).view(-1, self.channels, self.base_len)
        for up in self.trunk_upsamples:
            trunk = self.trunk_act(up(trunk))

        return {name: self.stokes_decoders[name](trunk) for name in self.stokes_names}


# --------------------------------------------------------------------------- #
# Transformer encoder-decoder
# --------------------------------------------------------------------------- #

class TransformerEncoderDecoder(nn.Module):
    """Self-attention encoder-decoder over B-spline coefficients.

    Every physical-channel coefficient (plus mu) becomes one source token:
    its scalar value, plus a learned embedding of which channel it belongs
    to and its position within that channel. The Transformer encoder lets
    every physical coefficient attend to every other, across channels and
    along depth. Every Stokes-channel output coefficient is produced from a
    learned, content-free query (channel identity + within-channel
    position, à la DETR object queries) that cross-attends to the encoded
    physical state and self-attends to every other output coefficient --
    including its neighbours within the same channel -- so the network can
    directly learn correlation/smoothness along wavelength instead of
    predicting each coefficient independently, as the flat MLP does. Output
    length is fixed by construction (one query per target coefficient), so
    decoding is a single non-autoregressive forward pass, not
    token-by-token generation.

    Like the rest of the pipeline, ``forward`` takes/returns a per-channel
    dict, never a single concatenated tensor. Internally it still has to
    lay every token out along one sequence axis to run the shared attention
    layers -- that's an unavoidable, purely computational reshape, not a
    loss of information: every token also carries an explicit (channel,
    position) embedding, so unlike a raw concatenated coefficient vector,
    the model never treats sequence-adjacency as if it meant
    depth/wavelength-adjacency.
    """

    def __init__(
        self,
        phys_names: Sequence[str],
        phys_sizes: Dict[str, int],
        stokes_names: Sequence[str],
        stokes_sizes: Dict[str, int],
        d_model: int = 128,
        nhead: int = 4,
        num_encoder_layers: int = 4,
        num_decoder_layers: int = 4,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.phys_names = list(phys_names)
        self.phys_sizes = dict(phys_sizes)
        self.stokes_names = list(stokes_names)
        self.stokes_sizes = dict(stokes_sizes)

        self.value_proj = nn.Linear(1, d_model)

        n_src_channels = len(self.phys_names) + 1  # + 1 for the mu token
        max_src_len = max(self.phys_sizes.values())
        self.src_channel_embed = nn.Embedding(n_src_channels, d_model)
        self.src_pos_embed = nn.Embedding(max(max_src_len, 1), d_model)

        n_tgt_channels = len(self.stokes_names)
        max_tgt_len = max(self.stokes_sizes.values())
        self.tgt_channel_embed = nn.Embedding(n_tgt_channels, d_model)
        self.tgt_pos_embed = nn.Embedding(max(max_tgt_len, 1), d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, dropout, activation="gelu",
            batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers, enable_nested_tensor=False)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model, nhead, dim_feedforward, dropout, activation="gelu",
            batch_first=True, norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_decoder_layers)

        self.out_proj = nn.Linear(d_model, 1)

        # Precomputed (channel, position)-index lookups for every flat input
        # / output slot, so forward() never has to rebuild them. Registered
        # as non-persistent buffers so they move with .to(device) but aren't
        # duplicated in checkpoints.
        src_chan_idx, src_pos_idx = [], []
        for i, name in enumerate(self.phys_names):
            n = self.phys_sizes[name]
            src_chan_idx += [i] * n
            src_pos_idx += list(range(n))
        src_chan_idx += [len(self.phys_names)]  # mu gets its own channel id
        src_pos_idx += [0]
        self.register_buffer("src_chan_idx", torch.tensor(src_chan_idx, dtype=torch.long), persistent=False)
        self.register_buffer("src_pos_idx", torch.tensor(src_pos_idx, dtype=torch.long), persistent=False)

        tgt_chan_idx, tgt_pos_idx = [], []
        for i, name in enumerate(self.stokes_names):
            n = self.stokes_sizes[name]
            tgt_chan_idx += [i] * n
            tgt_pos_idx += list(range(n))
        self.register_buffer("tgt_chan_idx", torch.tensor(tgt_chan_idx, dtype=torch.long), persistent=False)
        self.register_buffer("tgt_pos_idx", torch.tensor(tgt_pos_idx, dtype=torch.long), persistent=False)

    def forward(self, coeffs: Dict[str, torch.Tensor], mu: torch.Tensor) -> Dict[str, torch.Tensor]:
        """coeffs: name -> (B, phys_sizes[name]), mu: (B,) or (B, 1) ->
        name -> (B, stokes_sizes[name])."""
        if mu.dim() == 1:
            mu = mu.unsqueeze(-1)
        batch = mu.shape[0]

        # One value per source token, laid out along the sequence axis in
        # src_chan_idx/src_pos_idx order so the (channel, position)
        # embeddings below line up; see the class docstring for why this
        # internal reshape doesn't imply false depth/wavelength adjacency.
        x = torch.cat([coeffs[name] for name in self.phys_names] + [mu], dim=-1)
        src = self.value_proj(x.unsqueeze(-1))
        src = src + self.src_channel_embed(self.src_chan_idx) + self.src_pos_embed(self.src_pos_idx)
        memory = self.encoder(src)

        query = self.tgt_channel_embed(self.tgt_chan_idx) + self.tgt_pos_embed(self.tgt_pos_idx)
        query = query.unsqueeze(0).expand(batch, -1, -1)
        tgt = self.decoder(query, memory)
        out = self.out_proj(tgt).squeeze(-1)

        result = {}
        idx = 0
        for name in self.stokes_names:
            n = self.stokes_sizes[name]
            result[name] = out[:, idx: idx + n]
            idx += n
        return result


# --------------------------------------------------------------------------- #
# Factory / registry, shared by train.py and emulator.py so checkpoints can
# record just (model_type, model_kwargs) and be reconstructed identically.
# --------------------------------------------------------------------------- #

MODEL_REGISTRY = {
    "mlp": MLP,
    "resmlp": ResidualMLP,
    "cnn": CNNEncoderDecoder,
    "transformer": TransformerEncoderDecoder,
}


def build_model(model_type: str, model_kwargs: Dict) -> nn.Module:
    if model_type not in MODEL_REGISTRY:
        raise ValueError(f"model_type must be one of {list(MODEL_REGISTRY)}, got {model_type!r}")
    return MODEL_REGISTRY[model_type](**model_kwargs)
