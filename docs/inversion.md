# Inversion: how it works

The machinery lives in `modules/inversion.py` (`Inverter`,
`MultiRegionInverter`) and `modules/regularization.py`. Every `invert_*.py`
script configures it. This page describes what happens; the
[configuration page](configuration.md) lists every option.

## Unknowns

Each pixel $(x,y)$ carries:

* **depth variables** `T, vz, vmic, Bx, By, Bz`: their values at the nodes of
  each variable, $\theta_{v}(x,y)\in\mathbb R^{n_v}$;
* **per-pixel variables** (`slab_variable`), with one value per pixel: the He
  I slab parameters `He*_Bx, ..., He*_height` and the telluric parameters
  `Tel_shift, ...`.

The optimizer sees $\theta/\mathrm{scale}$, where `scale` is the typical
size of each variable (500 K, 1 km/s, 0.5 km/s, 200 G, ...). A learning
rate of 0.1 therefore means steps of about 0.1 × the typical size, and all
variables are equally well conditioned.

## Forward model per iteration

For a chunk of pixels:

```
θ (nodes)  ──M_v──▶  stratification x_v(τ) on 80 points          (interpolating B-spline, per variable)
                     v_ref = vz(log τ = −1);  vz ← vz − v_ref
           ──────▶  synthesizer(x, μ, v_ref, λ_obs, slab)          (SpectralSynthesizer / SiHeSynthesizer)
                     └ emulators: B-spline projection → ResMLP → PCA → inverse symlog
                     └ (He I slabs) → telluric × I → instrumental PSF → Doppler shift → interpolate to λ_obs
           ──────▶  synthetic S (npix, 4, nl), valid mask
```

See [B-splines](bsplines.md#inversion-side-nodes-and-interpolating-b-splines)
for the node interpolation and [Synthesis](synthesis.md) for the
synthesizers.

## Objective

$$
J(\theta) = \sum_{\text{pixels}} \chi^2_\text{pixel}
\;+\; \sum_{v}\sum_{r\in\mathcal R_v} w_r\,R_r(\theta_v\text{ maps}),
\qquad
\chi^2_\text{pixel} = \sum_{s\in\{I,Q,U,V\}} \frac{w_s}{\sigma_s^2}\sum_{\lambda\ \text{valid}}\left(S^\text{syn}_{s\lambda}-S^\text{obs}_{s\lambda}\right)^2 .
$$

* $\sigma_s$ is `InversionConfig.noise[s]` and $w_s$ is
  `stokes_weights[s]`.
* An observed point is **valid** if it is finite (NaN marks missing data) and
  lies inside the shifted synthetic grid. Invalid points are ignored.
* The regularizers $R_r$ act on the full 2-D map of every node of a
  variable, in physical units (K, km/s, G), so the weights have physical
  meaning.
* For several regions (`MultiRegionInverter`, `normalize=True`) the data term
  of a pixel is $\sum_r \mathrm{weight}_r\,\chi^2_r/n_r$, with $n_r$
  the number of valid points of region $r$. Each region then counts by its
  weight whatever its number of wavelengths, and regularization weights are
  relative to a $\chi^2$ per point.

## Optimization: proximal Adam with box constraints

Every iteration:

1. **Learning rate:** cosine decay from `lr` to `lr · lr_final_fraction`
   over `n_iter` iterations.
2. **Gradient of the data term:** accumulated chunk by chunk
   (`chunk_size` pixels per forward/backward pass), so any field of view
   fits in memory.
3. **Adam step** (or `"adamw"`, `"sgd"`) on the free variables. Variables
   with `fixed=True` are not optimized.
4. **Proximal step** for the spatial regularizers, on the full maps:

   $
   \theta_v \leftarrow \operatorname{prox}_{\tau R}(\theta_v)
   = \arg\min_z \tfrac12\|z-\theta_v\|^2 + \tau\,w\,R(z).
   $

   The step $\tau$ of each node map is Adam's current effective step
   size $\mathrm{lr}/(\sqrt{\hat v}+\epsilon)$, averaged over the map and
   converted to physical units ($\times\mathrm{scale}^2$). With this
   choice the fixed point of the iteration approximates the minimizer of
   $J$, and the regularization weights keep the meaning they have in
   $J$. Several regularizers on one variable are applied one after the
   other.
5. **Projection** onto the per-node box bounds.

The data term is smooth and handled by gradients. The regularizers can be
non-smooth (TV, L1) and are handled exactly by their proximal operators.

### Spatial regularizers

| `Regularizer(kind, weight, options)` | $R(z)$ | Proximal operator |
|------|------|------|
| `"l2"` | $\sum\lvert\nabla z\rvert^2$ (forward differences, Neumann boundaries) | exact: solves $(\mathbb 1 + 2\tau\nabla^\top\nabla)z=x$ with an orthonormal DCT-II |
| `"tv"` | $\sum\sqrt{(\partial_x z)^2+(\partial_y z)^2}$ (isotropic TV) | fast gradient projection (Beck & Teboulle 2009), `options={"n_iter": 50}` |
| `"l1"` | $\sum\lvert z\rvert$ | soft threshold |
| `"wavelet_l1"` | $\ell_1$ of the detail coefficients of an orthonormal periodic 2-D wavelet transform | exact: $W^\top\mathrm{soft}(Wx)$; `options={"wavelet": "db2", "levels": 3}` |

`weight` is a scalar or **one value per node**, so the heights of a depth
variable can be regularized differently (0 switches a node off).

(choosing-weights)=
#### Choosing weights: `REG_STRENGTH / jump²`

All scripts use

```python
REG_STRENGTH = 5.0
def smooth(jump):
    return [Regularizer("l2", REG_STRENGTH / np.asarray(jump, dtype=np.float64) ** 2)]
```

With $w=\mathrm{REG\_STRENGTH}/\mathrm{jump}^2$, a difference of `jump`
(in the units of the variable) between neighbouring pixels costs
`REG_STRENGTH` units of $\chi^2$, and larger jumps are penalized
quadratically. With ~1700 data points per pixel (4 Stokes × ~420 λ),
$\chi^2$ fluctuates by $\sqrt{2\cdot1700}\approx60$ per pixel from
noise alone. So `REG_STRENGTH ≈ 10` is mild and `≈ 100` is strong. Raise
`REG_STRENGTH` to smooth everything, or lower the `jump` of one variable to
smooth only that one.

#### Ambiguity-invariant transverse field

With $B_x\ge 0$ (the lower bound that removes the 180° ambiguity) the
solution jumps from $(0, B_t)$ to $(0,-B_t)$ where the azimuth
crosses ±90°. Smoothing $B_x$ and $B_y$ separately would turn that
jump into a spurious line of $B_t\approx0$. `TransverseRegularizer("Bx",
"By", [...])` applies the regularizers instead to

$$
u = B_t\cos2\phi,\qquad v = B_t\sin2\phi ,
$$

which are the same for both ambiguous solutions. The result is mapped back
with $B_x\ge0$. A jump in $B_t$ costs as much as a jump in
$B_x$ or $B_y$, and an azimuth change $\mathrm d\phi$ costs as much
as a jump of $2B_t\,\mathrm d\phi$. The two variables must have the same
number of nodes and no regularizers of their own.

## Spatial PSF

`Inverter(..., spatial_psf=SpatialPSF(psf, size=63))` convolves the synthetic
maps of the whole field of view with a 2-D telescope PSF (by FFT, for all
Stokes parameters and wavelengths at once, with reflecting padding) before
the comparison. The PSF couples pixels in different chunks, so the gradient
is computed in **two passes**:

1. synthesize all chunks without an autograd graph, giving the full maps
   $S$;
2. make $S$ a leaf tensor, convolve it, compute $\chi^2$ and
   backpropagate to get $\partial\chi^2/\partial S$ (a small graph: FFT
   plus $\chi^2$);
3. re-synthesize every chunk with a graph and call
   `backward(S_chunk, grad=∂χ²/∂S[chunk])`.

The resulting gradient is exact, memory stays at one chunk's graph, and the
cost is about 1.5 × an iteration without the PSF. With a single chunk the
graph is kept and there is one pass.

## Initial guess

`initial_guess(observations, wave_obs, config, logtau, mean_strat, wfa=..., cog_lines=...)`:

* **`T`, `vmic`** (and any variable without a specific estimate): the median
  training stratification interpolated at the nodes, in every pixel.
* **`vz`**: the centre-of-gravity velocity of I in the windows
  `cog_lines = [(lambda0, half_width), ...]`, averaged over lines, at every
  node.
* **`Bx, By, Bz`**: a per-pixel least-squares weak-field estimate from the
  `WeakFieldConfig` lines. The solution with $B_x\ge0$ is chosen, and
  $B_\perp$ is raised to at least `bperp_floor = 30` G, because
  $B_x=B_y=0$ is a stationary point of Q and U (both are quadratic in
  the transverse field).

He I slabs: `he_slabs_initial_guess` estimates the velocity from the centre of
gravity of the red component, $\tau$ from its depth relative to the
continuum at 10831.3–10832.5 Å, and the field from the weak-field
approximation of the red blend ($\bar g=1.13$, textbook Q/U sign), rotated
to the vertical frame with μ. With several slabs, $\tau$ is divided among
them and their velocities are spread by ±2 km/s so they do not start
identical. Telluric lines start from the position and depth of the I minimum
near the line.

## Output

`Inverter.run` returns an `InversionResult`: node values (physical units,
absolute `vz`), the stratifications on the 80-point grid, `v_ref`, the
synthetic profiles, $\chi^2$ per valid point for each pixel, and the
history. `save_result` writes it to HDF5. See [Output](output.md).

## Performance notes

* `chunk_size` sets GPU memory use. The Fe I script uses 196 608 pixels per
  chunk, and the Si I + He I script uses 1024 (the slab formal solution on a
  0.02 Å grid is much heavier).
* Typical runs use 500 iterations at `lr = 0.1` with cosine decay to
  `0.05 · lr`.
* The log prints, every `log_every` iterations, the data term, the
  regularization term and the learning rate.
