"""Depth-gradient features appended to the physical B-spline coefficients.

The coefficients of each physical channel already determine its
derivatives, which a first linear layer could in principle learn, but
asymmetries of the Stokes profiles are driven by *products* of gradients
(e.g. velocity and field gradients together produce net circular
polarization), which an MLP has to build from scratch. This module gives
them to it directly.

Every channel is a B-spline ``f(x) = sum_i c_i B_i(x)`` on the depth grid
``x`` (e.g. log tau), so its value and derivative at any point are linear
in the coefficients, ``f^(nu)(x_k) = sum_i c_i B_i^(nu)(x_k)``: one frozen
matrix per (channel, order), evaluated at a common set of ``eval_points``
so that products of different channels line up in depth. Alternatively
(``mode="diff"``) a derivative is represented by the coefficients of the
derivative spline, ``p (c[i+1] - c[i]) / (t[i+p+1] - t[i+1])``, i.e. the
scaled differences of consecutive coefficients.

Features are computed from the *unscaled* coefficients (physical units),
then standardized per element with statistics fit on the training set and,
for products, optionally passed through ``asinh`` to tame their heavy tails.
All of it is stored in the module's ``state_dict``, so the emulator stays
self-contained and differentiable with respect to the stratification.
"""
from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from scipy.interpolate import BSpline

from .bspline import BSplineGroup
from .config import DepthFeatureConfig

# Added to |B|^2 (G^2) so that |B| and d|B| stay differentiable at B = 0.
_B_EPS = 1.0


def _key(name: str) -> str:
    return name.replace("*", "_x_")


def default_eval_points(cfg: DepthFeatureConfig, phys_group: BSplineGroup) -> List[float]:
    if cfg.eval_points is not None:
        return list(cfg.eval_points)
    return [float(x) for x in np.unique(phys_group.projectors[phys_group.names[0]].knots)]


class DepthFeatures(nn.Module):
    def __init__(self, cfg: DepthFeatureConfig, phys_names: Sequence[str],
                 phys_coeff_sizes: Dict[str, int], eval_points: Sequence[float]):
        super().__init__()
        self.cfg = cfg
        self.phys_names = list(phys_names)
        self.eval_points = list(eval_points)
        n_eval = len(self.eval_points)
        known = set(self.phys_names) | {"B"}
        if "B" in self.phys_names:
            raise ValueError("'B' is reserved for the field strength and cannot be a channel name")
        if any(c not in self.phys_names for c in cfg.field_channels) and self._uses_field_strength():
            raise ValueError(f"field_channels {cfg.field_channels} not all in {self.phys_names}")

        # feature name -> ("diff", channel) or list of (channel, order) factors
        self.recipes: Dict[str, object] = {}
        for ch in cfg.derivatives:
            if ch not in known:
                raise ValueError(f"unknown channel {ch!r} in derivatives")
            if cfg.mode == "diff":
                if ch == "B":
                    raise ValueError("|B| is not linear in the coefficients: use mode='spline' for 'B'")
                self.recipes[f"d{ch}"] = ("diff", ch)
            else:
                self.recipes[f"d{ch}"] = [(ch, 1)]
        for p in cfg.products:
            self.recipes[p] = [self._parse_factor(t.strip(), known) for t in p.split("*")]

        ops: Dict[str, Tuple[int, int]] = {}
        for recipe in self.recipes.values():
            if isinstance(recipe, tuple):
                ch = recipe[1]
                ops[f"{ch}_diff"] = (phys_coeff_sizes[ch] - 1, phys_coeff_sizes[ch])
                continue
            for ch, nu in recipe:
                chans = cfg.field_channels if ch == "B" else (ch,)
                orders = range(nu + 1) if ch == "B" else (nu,)
                for c in chans:
                    for o in orders:
                        ops[f"{c}_{o}"] = (n_eval, phys_coeff_sizes[c])
        self.ops = nn.ParameterDict({
            k: nn.Parameter(torch.zeros(shape), requires_grad=False) for k, shape in sorted(ops.items())
        })

        self.feature_names = list(self.recipes)
        self.feature_sizes = {
            name: (phys_coeff_sizes[r[1]] - 1 if isinstance(r, tuple) else n_eval)
            for name, r in self.recipes.items()
        }
        self.offset = nn.ParameterDict({
            _key(n): nn.Parameter(torch.zeros(s), requires_grad=False) for n, s in self.feature_sizes.items()
        })
        self.scale = nn.ParameterDict({
            _key(n): nn.Parameter(torch.ones(s), requires_grad=False) for n, s in self.feature_sizes.items()
        })

    def _uses_field_strength(self) -> bool:
        tokens = list(self.cfg.derivatives) + [t.strip() for p in self.cfg.products for t in p.split("*")]
        return any(t in ("B", "dB") for t in tokens)

    @staticmethod
    def _parse_factor(token: str, known: set) -> Tuple[str, int]:
        if token in known:
            return token, 0
        if token.startswith("d") and token[1:] in known:
            return token[1:], 1
        raise ValueError(f"cannot parse factor {token!r}: expected one of {sorted(known)}, "
                         f"optionally prefixed by 'd'")

    def set_matrices(self, phys_group: BSplineGroup):
        x = np.asarray(self.eval_points, dtype=np.float64)
        with torch.no_grad():
            for key, param in self.ops.items():
                ch, kind = key.rsplit("_", 1)
                proj = phys_group.projectors[ch]
                t, p, n = proj.knots, proj.degree, proj.n
                if kind == "diff":
                    m = np.zeros((n - 1, n))
                    for i in range(n - 1):
                        h = t[i + p + 1] - t[i + 1]
                        if h > 0:
                            m[i, i], m[i, i + 1] = -p / h, p / h
                else:
                    xc = np.clip(x, t[p], t[-p - 1])
                    m = BSpline(t, np.eye(n), p).derivative(int(kind))(xc) if int(kind) else \
                        BSpline(t, np.eye(n), p)(xc)
                param.copy_(torch.as_tensor(m, dtype=torch.float32))

    def _factor(self, coeffs: Dict[str, torch.Tensor], ch: str, nu: int) -> torch.Tensor:
        if ch != "B":
            return coeffs[ch] @ self.ops[f"{ch}_{nu}"].t().to(coeffs[ch].dtype)
        comps = [self._factor(coeffs, c, 0) for c in self.cfg.field_channels]
        b = torch.sqrt(sum(c * c for c in comps) + _B_EPS)
        if nu == 0:
            return b
        return sum(c * self._factor(coeffs, f, 1) for c, f in zip(comps, self.cfg.field_channels)) / b

    def raw(self, coeffs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Unscaled coefficients (name -> (B, n)) -> unscaled features."""
        out = {}
        for name, recipe in self.recipes.items():
            if isinstance(recipe, tuple):
                ch = recipe[1]
                out[name] = coeffs[ch] @ self.ops[f"{ch}_diff"].t().to(coeffs[ch].dtype)
            else:
                vals = [self._factor(coeffs, ch, nu) for ch, nu in recipe]
                out[name] = vals[0] if len(vals) == 1 else vals[0] * vals[1]
        return out

    def fit(self, coeffs: Dict[str, torch.Tensor]):
        """Fits the per-element standardization on (training) coefficients."""
        with torch.no_grad():
            for name, f in self.raw(coeffs).items():
                std = f.std(dim=0)
                self.offset[_key(name)].copy_(f.mean(dim=0).float())
                self.scale[_key(name)].copy_(torch.where(std < 1e-12, torch.ones_like(std), std).float())

    def forward(self, coeffs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Unscaled coefficients -> standardized (and squashed) features."""
        out = {}
        for name, f in self.raw(coeffs).items():
            z = (f - self.offset[_key(name)].to(f.dtype)) / self.scale[_key(name)].to(f.dtype)
            if self.cfg.squash_products and name in self.cfg.products:
                z = torch.asinh(z)
            out[name] = z
        return out


def build_depth_features(cfg: DepthFeatureConfig, phys_group: BSplineGroup) -> DepthFeatures:
    features = DepthFeatures(cfg, phys_group.names, phys_group.sizes, default_eval_points(cfg, phys_group))
    features.set_matrices(phys_group)
    return features
