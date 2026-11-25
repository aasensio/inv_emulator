# PCA of the Stokes profiles

The emulators do not predict Stokes profiles wavelength by wavelength. They
predict the coefficients of each Stokes parameter in a **principal-component
basis** fit to the training profiles (`modules/pca.py`). One basis is fit per
channel (I, Q, U, V independently), on the quantity that is actually fit,
i.e. after the [Stokes transform](#what-is-decomposed).

## Decomposition

For one channel, let $X\in\mathbb{R}^{N\times L}$ hold the $N$
training profiles (rows) on $L$ wavelengths, with mean profile
$\bar z = \frac1N\sum_n X_n$ and centred matrix $X_c = X - \bar z$.
The principal directions are the eigenvectors of the covariance

$$
C = \frac1N X_c^\top X_c = V\Lambda V^\top,\qquad \lambda_1\ge\lambda_2\ge\dots\ge0 .
$$

Keeping the leading $n$ eigenvectors $E=[v_1,\dots,v_n]\in\mathbb{R}^{L\times n}$
(orthonormal, $E^\top E=\mathbb{1}$):

$$
\text{projection: } a = E^\top (z-\bar z),\qquad
\text{reconstruction: } \hat z = \bar z + E\,a .
$$

`PCAProjector.fit` diagonalizes whichever matrix is smaller:

* $L\le N$ (the usual case): the $L\times L$ covariance, accumulated in
  chunks of 65 536 profiles so that no centred copy of a large training set
  is ever made;
* $N<L$: the $N\times N$ Gram matrix $X_cX_c^\top u=s\,u$, mapped
  back to wavelength space with $v = X_c^\top u/\sqrt{s}$.

The arbitrary sign of each eigenvector is fixed so that its largest-magnitude
entry is positive, which makes refits reproducible.

### Number of components

`ChannelPCAConfig` takes exactly one of:

* `num_components=n`: keep $n$ components;
* `variance_fraction=f`: keep the fewest components whose cumulative
  explained variance $\sum_{i\le n}\lambda_i/\sum_i\lambda_i$ reaches
  $f$.

`center=True` (the default) subtracts the mean before the decomposition. It
should almost always be left on, in particular for Stokes I.

The photospheric emulators use `variance_fraction=0.99999` for every
channel, which gives:

| Emulator | $L$ | I | Q | U | V | total outputs |
|----------|------:|--:|--:|--:|--:|--------------:|
| Si I 10827 | 350 | 18 | 49 | 49 | 44 | 160 |
| Fe I 6301 | 70 | 22 | 39 | 40 | 36 | 137 |
| Fe I 6302 | 80 | 20 | 34 | 34 | 32 | 120 |
| Ca II 8542 | 150 | 38 | 55 | 55 | 51 | 199 |

On the Si I validation set this reconstructs the held-out physical profiles
to about 10⁻⁴ rms, so the PCA truncation error is well below the network
error. The He I slab emulator uses `num_components=30` for each of its 11
channels (330 outputs), which reconstructs to about 10⁻⁶ rms (see
[He I 10830](emulators.md#he-i-10830)).

## What is decomposed

Q, U and V can span several orders of magnitude across the database, and they
have to stay exactly zero without a field. A linear basis fit to raw Q, U, V
would be dominated by the strongest-field profiles. The PCA is therefore fit
to

$$
z_I = I,\qquad z_{Q,U,V} = \operatorname{symlog}\!\left(a\,\frac{Q,U,V}{I}\right),\qquad
\operatorname{symlog}(y)=\operatorname{sign}(y)\log(1+|y|),
$$

with $a=100$ (Si I) or $a=30$ (Fe I, Ca II). This is
`StokesTransformConfig(amplitude=a)` in the training scripts. The emulator
inverts it exactly after reconstruction, so `emulator(physical, mu)` always
returns physical profiles.

## Scaling of the coefficients

The coefficients are standardized with **one scale per channel**:

```python
ScalerConfig(kind="standard", per_coeff=False)
```

Because $E$ is orthonormal,
$\|z-\hat z\|^2 = \|a-\hat a\|^2$. A single scale per channel keeps the
training MSE proportional to the profile-space error of that channel.
Per-coefficient standardization (`per_coeff=True`) would instead give the
trailing, low-variance (and noise-dominated) components the same weight as
the leading ones.

## Storage in the emulator

When an emulator is built, the eigenvectors of each channel are copied into
`B_stokes[s]` ($L\times n_s$) and the mean profile into `stokes_mean[s]`.
Decoding is then

```python
c = y_scaled[s] * stokes_scale[s] + stokes_offset[s]      # undo standardization
z = c @ B_stokes[s].T + stokes_mean[s]                     # PCA reconstruction
```

followed by the inverse symlog transform. The basis is fixed after training,
so it costs nothing at inference beyond one $n_s\times L$ matrix product
per channel.

## Using the PCA classes directly

```python
from modules.config import ChannelPCAConfig
from modules.pca import PCAProjector, PCAGroup

proj = PCAProjector(grid=wave, cfg=ChannelPCAConfig(variance_fraction=0.99999)).fit(profiles)  # (N, L)
proj.n, proj.explained_variance_fraction
a = proj.project(profiles)          # (N, n)
rec = proj.reconstruct(a)           # (N, L)

group = PCAGroup(["I", "Q", "U", "V"], wave, {s: ChannelPCAConfig(num_components=30) for s in "IQUV"})
group.fit(stokes)                    # (N, 4, L)
coeffs = group.project(stokes)       # dict s -> (N, n_s)
stokes_rec = group.reconstruct(coeffs)
```

`PCAGroup` has the same `project`/`reconstruct`/`sizes` interface as
`BSplineGroup`, so the rest of the pipeline is independent of the basis.

## Training with the PCA basis

Abridged from `si10827.py`:

```python
from modules.config import ChannelPCAConfig, ChannelSplineConfig, ScalerConfig, StokesTransformConfig, DepthFeatureConfig
from modules.train import TrainConfig, train_emulator

phys_cfg = {ch: ChannelSplineConfig(num_coeffs=10, degree=3) for ch in ("T", "vz", "vmic", "Bx", "By", "Bz")}
phys_scaler_cfg = {"T": ScalerConfig("fixed", False, 5000.0, 0.0),
                   "vz": ScalerConfig("fixed", False, 2.0, 0.0), "vmic": ScalerConfig("fixed", False, 2.0, 0.0),
                   "Bx": ScalerConfig("fixed", False, 500.0, 0.0), "By": ScalerConfig("fixed", False, 500.0, 0.0),
                   "Bz": ScalerConfig("fixed", False, 500.0, 0.0)}

result = train_emulator(
    models, stokes, mu, depth, wave,                    # (N, 6, 80), (N, 4, L), (N,), (80,), (L,)
    phys_cfg, None,                                     # no Stokes spline config with PCA
    phys_scaler_configs=phys_scaler_cfg,
    stokes_scaler_configs={s: ScalerConfig(kind="standard", per_coeff=False) for s in "IQUV"},
    stokes_transform=StokesTransformConfig(amplitude=100.0),
    stokes_basis="pca",
    stokes_pca_configs={s: ChannelPCAConfig(variance_fraction=0.99999) for s in "IQUV"},
    depth_features=DepthFeatureConfig(derivatives=["T", "vz", "vmic", "Bx", "By", "Bz"], mode="diff"),
    train_cfg=TrainConfig(epochs=300, batch_size=1024, lr=1e-3, model_type="resmlp",
                          model_kwargs=dict(width=512, num_blocks=6, expansion=2),
                          optimizer="adamw", weight_decay=1e-5, scheduler="cosine", patience=None),
    checkpoint_path="weights/emulator_si10827_pca.pt",
)
```

The basis is fit on the training split only (at most
`TrainConfig.pca_fit_models = 200 000` models when there are several angles
per model). The validation split measures the generalization of both the
basis and the network.
