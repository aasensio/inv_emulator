# B-splines for the depth stratification

The physical variables $T,\,v_z,\,v_\mathrm{mic},\,B_x,\,B_y,\,B_z$ are
functions of the optical depth $\log\tau$ (at 5000 Å), sampled on the
80-point grid of the training models, from $\log\tau=1.5$ (deepest, index
0) to $-7.5$ (index 79). B-splines appear in two places, and both are
linear:

| Where | Direction | Matrix | Purpose |
|-------|-----------|--------|---------|
| emulator, `modules/bspline.py` | stratification (80) to coefficients (10) | $P = \mathrm{pinv}(B)$, 10×80 | compact, smooth network input |
| inversion, `modules/inversion.py` | node values ($n$) to stratification (80) | $M$, 80×$n$ | few free parameters per variable |

Combined, the emulator input is a **linear function of the inverted node
values**, $c = P\,M\,\theta$. Everything between the optimizer and the
network is therefore a fixed matrix product.

## B-spline basics

A B-spline of degree $p$ with knot vector
$t_0\le t_1\le\dots\le t_{n+p}$ is

$$
f(x) = \sum_{i=0}^{n-1} c_i\,N_{i,p}(x),
$$

where $N_{i,p}$ are the B-spline basis functions (piecewise polynomials of
degree $p$, non-negative, each nonzero over at most $p+1$ knot
intervals, summing to one). The code uses **clamped** knot vectors: the
end knots are repeated $p+1$ times, so the spline interpolates its first
and last coefficient at the boundaries of the grid.

```python
from modules.bspline import clamped_uniform_knots, clamped_knots_from_positions

# n coefficients, degree p, uniform interior knots over [x_min, x_max]
t = clamped_uniform_knots(-7.5, 1.5, num_coeffs=10, degree=3)
# -> [-7.5]*4 + [-6.214, -4.929, -3.643, -2.357, -1.071, 0.214] + [1.5]*4

# or explicit interior knots (strictly increasing, strictly inside the interval)
t = clamped_knots_from_positions(-7.5, 1.5, positions=[-5, -3, -2, -1, 0], degree=3)
# -> len(positions) + degree + 1 = 9 coefficients
```

## Emulator side: projection onto B-splines

### Design and projection matrices

On a grid $x_1,\dots,x_D$ the **design matrix** is
$B_{ji}=N_{i,p}(x_j)$, of size $D\times n$. A sampled stratification
$\mathbf f$ is represented by the least-squares coefficients

$$
\mathbf c = \arg\min_{\mathbf c}\|\mathbf f - B\mathbf c\|^2 = P\,\mathbf f,\qquad P = B^{+} = (B^\top B)^{-1}B^\top ,
$$

and $B\mathbf c$ is the smooth reconstruction. `BSplineProjector`
precomputes both matrices once:

```python
from modules.bspline import BSplineProjector
proj = BSplineProjector(grid=logtau, num_coeffs=10, degree=3)   # uniform clamped knots
proj.B.shape, proj.P.shape        # (80, 10), (10, 80)
c = proj.project(T_profile)       # (..., 80) -> (..., 10)
T_smooth = proj.reconstruct(c)    # (..., 10) -> (..., 80)
```

The grid may run in either direction. The design matrix is built on the
sorted grid and returned in the original order. `BSplineGroup` holds one
projector per channel. Each channel can have its own number of coefficients,
degree and knots (`ChannelSplineConfig(num_coeffs, degree, knots)`).

### Configuration of the trained emulators

All photospheric emulators use, for **every** physical channel,

```python
ChannelSplineConfig(num_coeffs=10, degree=3)    # uniform clamped knots on [-7.5, 1.5]
```

that is, 6 interior knots and 8 distinct knot positions
$\{-7.5, -6.21, -4.93, -3.64, -2.36, -1.07, 0.21, 1.5\}$, spaced every
$9/7\approx1.29$ in $\log\tau$. Only `P_phys` (10×80 per channel) is
stored in the emulator. At inference time the projection is one matrix
product per channel:

$$
\hat c_k = \frac{P_k\,x_k - \text{offset}_k}{\text{scale}_k}.
$$

```{important}
The projection is a low-pass filter. Two stratifications with the same 10
coefficients give the **same** Stokes profiles. Structure finer than the knot
spacing (≈1.3 dex) is invisible to the emulator, and the inversion cannot
constrain it either.
```

### Derivative-spline features

The derivative of a degree-$p$ spline is a degree-$(p-1)$ spline on
the same knots (minus one at each end), with coefficients

$$
c'_i = \frac{p\,(c_{i+1}-c_i)}{t_{i+p+1}-t_{i+1}},\qquad i=0,\dots,n-2 .
$$

This is a fixed $(n-1)\times n$ matrix $\Delta_k$ (`modules/features.py`,
`mode="diff"`). The emulators add $\Delta_k c_k$ for all six channels as
inputs: 9 values each, standardized on the training set. They give the network
direct access to $\mathrm{d}x/\mathrm{d}\log\tau$, which controls the
asymmetries of the profiles.

## Inversion side: nodes and interpolating B-splines

### From nodes to stratification

In the inversion, each depth variable is parameterized by its values
$\theta_1,\dots,\theta_n$ at a few **nodes**
$\tau_1<\dots<\tau_n$ (positions in $\log\tau$, set independently per
variable in `VariableConfig(nodes=...)`). The stratification is the
**interpolating** B-spline through the nodes, evaluated on the 80-point grid:

$$
\mathbf x = M\,\boldsymbol\theta,\qquad M = \texttt{node\_interpolation\_matrix}(\log\tau,\ \text{nodes}),
$$

with degree $\min(3,\,n-1)$, built with `scipy.interpolate.make_interp_spline`
(not-a-knot conditions for the cubic case). The number of nodes sets the
shape freedom:

| Nodes | Degree | Stratification |
|-------|--------|----------------|
| 1 | 0 | constant with depth |
| 2 | 1 | linear in $\log\tau$ (a constant gradient) |
| 3 | 2 | quadratic |
| ≥ 4 | 3 | cubic spline through the nodes |

**Beyond the outermost nodes the stratification is constant.** The grid is
clipped to $[\tau_1,\tau_n]$ before evaluation. For example, the nodes
`[-4.5, -3.0, -1.5, 0.0]` of `invert_multiregion.py` hold $T$ constant
above $\log\tau=-4.5$ and below $\log\tau=0$. Put nodes at −7.5 and 1.5
to let a variable vary over the whole atmosphere.

```python
from modules.inversion import node_interpolation_matrix
M = node_interpolation_matrix(logtau, [-7.5, -4.0, -2.5, -1.0, 0.0, 1.5])   # (80, 6)
T_strat = node_values @ M.T                                               # (..., 6) -> (..., 80)
```

The matrices are built once per variable (`_DepthGeometry`), stored on the
device, and applied to a whole chunk of pixels at once.

### How node and emulator B-splines interact

The emulator sees $c_k = P_k M_k \theta_k$, a $10\times n_k$ linear
map of the nodes. This has three consequences:

* **Gradients are cheap and exact.** $\partial c/\partial\theta = P M$.
* **Nodes closer than the emulator knot spacing are partly redundant.** Their
  difference is smoothed away by $P$. Keep nodes ≳1–1.5 dex apart, or
  regularize them.
* **Nodes outside the sensitivity range of a line are poorly constrained.**
  They are kept in range by their bounds and by spatial regularization. The
  scripts use nodes at −7.5 and 1.5 for the 2-node variables, so the
  inversion fits a gradient over the whole atmosphere. The line itself
  constrains it only where it forms.

### Velocity reference

The emulators were trained with $v_z(\log\tau=-1)=0$. The `vz` nodes of
the inversion are **absolute** velocities. After building $v_z(\tau)=M\theta$,
the code splits

$$
v_z(\tau) = v_\mathrm{ref} + \delta v(\tau),\qquad v_\mathrm{ref} = v_z(\log\tau_\mathrm{ref}),
$$

where $\tau_\mathrm{ref}$ is the grid point closest to
`InversionConfig.reference_depth = -1.0` (index 22, $\log\tau=-1.006$).
The emulator receives $\delta v$, and $v_\mathrm{ref}$
Doppler-shifts the synthetic spectrum. Positive velocities are redshifts
(downflows). For a region whose emulators were trained with absolute
velocities, set `Region(reference_depth=None)`.

### Bounds per node

Box bounds are given per node, in physical units. The scripts derive them from
the training models:

```python
def training_bounds(name, nodes):
    return (interpolate_in_depth(nodes, logtau, lo_strat[name]),     # 0.1 percentile at each node height
            interpolate_in_depth(nodes, logtau, hi_strat[name]))     # 99.9 percentile
```

`interpolate_in_depth` is `np.interp` for a grid in either order. Since the
stratification interpolates the nodes, a cubic spline can overshoot between
nodes even when every node is within bounds. In practice the overshoot is
small for smooth node sets.

### Choosing nodes

Starting points from the scripts:

| Variable | Typical nodes | Comment |
|----------|---------------|---------|
| `T` | 5–7 nodes, e.g. `[-7.5, -4.0, -2.5, -1.0, 0.0, 1.5]` | the best constrained variable; cubic |
| `vz` | `[-7.5, 1.5]` (linear) or 3 nodes | a gradient produces line asymmetries |
| `vmic` | `[-1.0]` (constant) | weakly constrained |
| `Bx`, `By`, `Bz` | `[-7.5, 1.5]` (linear), or `[-1.0]` (constant) for the transverse field | Q and U are weak; a constant transverse field is more robust |

For a single photospheric line (Si I or Fe I), more than about 6 `T` nodes
and 3 nodes for the other variables are usually not supported by the data.
The [scripts](scripts/index.md) show working combinations.
