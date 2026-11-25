# API reference

This page covers the classes and functions used by the synthesis and
inversion scripts, restricted to the B-spline depth basis and PCA Stokes
basis that the shipped emulators use.

## `modules.bspline`

| Name | Description |
|------|-------------|
| `clamped_uniform_knots(x_min, x_max, num_coeffs, degree)` | clamped knot vector with uniformly spaced interior knots, giving `num_coeffs` basis functions |
| `clamped_knots_from_positions(x_min, x_max, positions, degree)` | clamped knot vector with explicit interior knots, giving `len(positions) + degree + 1` coefficients |
| `bspline_design_matrix(grid, knots, degree)` | dense `B (len(grid), n)`; the grid may be in any order |
| `BSplineProjector(grid, num_coeffs, degree=3, knots=None)` | `.B (D, n)`, `.P = pinv(B) (n, D)`, `.project(f)`, `.reconstruct(c)`, `.knots` |
| `BSplineGroup(names, grid, channel_configs)` | one projector per channel; `.project(array (N, C, D))` returns a dict, `.reconstruct(dict)` returns `(N, C, D)`, `.sizes` |

## `modules.pca`

| Name | Description |
|------|-------------|
| `PCAProjector(grid, ChannelPCAConfig)` | `.fit(f (N, L))`, then `.mean`, `.B (L, n)` (eigenvectors), `.P = B.T`, `.eigenvalues`, `.n`, `.explained_variance_fraction`, `.project`, `.reconstruct` |
| `PCAGroup(names, grid, configs)` | `.fit(array (N, C, L))`, same `.project`/`.reconstruct`/`.sizes` interface as `BSplineGroup` |

## `modules.config`

| Name | Fields |
|------|--------|
| `ChannelSplineConfig` | `num_coeffs`, `degree=3`, `knots=None` (full clamped vector of length `num_coeffs + degree + 1`) |
| `ChannelPCAConfig` | exactly one of `num_components` or `variance_fraction`; `center=True` |
| `ScalerConfig` | `kind` (`"standard"`, `"minmax"`, `"fixed"`, `"none"`), `per_coeff=True`, `scale`, `offset` (for `"fixed"`) |
| `StokesTransformConfig` | `amplitude` (the $a$ of symlog(a·X/I)), `intensity_channel="I"`, `ratio_channels=None` (all others) |
| `DepthFeatureConfig` | `derivatives`, `mode` (`"spline"`/`"diff"`), `products`, `eval_points`, `squash_products`, `field_channels` |
| `WeakFieldLine`, `WeakFieldConfig` | weak-field line definitions, used by the inversion scripts **only for the initial guess** |

## `modules.transforms`

`symlog(x, a)`, `inv_symlog(y, a)`,
`apply_stokes_transform(stokes, names, cfg)`, and
`invert_stokes_transform(targets, names, cfg)`, which works on numpy arrays
or torch tensors.

## `modules.emulator`

| Name | Description |
|------|-------------|
| `StokesEmulator` | `nn.Module`; `forward(physical (B, 6, D), mu (B,))` returns `(B, 4, L)`; `encode`, `decode`; frozen buffers `P_phys`, `B_stokes`, `stokes_mean`, scalers |
| `load_emulator(path, map_location=None)` | returns `(emulator, meta)` |
| `save_emulator(path, emulator, meta)` | writes `{"state_dict", "meta"}` |

## `modules.train`

`train_emulator(physical, stokes, mu, phys_grid, stokes_grid,
phys_spline_configs, stokes_spline_configs, phys_scaler_configs=None,
stokes_scaler_configs=None, ..., stokes_transform=None, train_cfg=TrainConfig(),
checkpoint_path=None, stokes_basis="bspline", stokes_pca_configs=None,
depth_features=None)` (the shipped emulators were trained with
`stokes_basis="pca"`, `stokes_spline_configs=None` being allowed in that case) returns `TrainResult(emulator, meta, history)`. The best
validation checkpoint is written to `checkpoint_path` whenever the validation
loss improves.

`TrainConfig` fields: `epochs`, `batch_size`, `lr`, `weight_decay`,
`model_type` (`"resmlp"` for the shipped emulators), `model_kwargs`
(`width`, `num_blocks`, `expansion`), `optimizer` (`"adam"`/`"adamw"`),
`scheduler` (`"plateau"`/`"cosine"`), `warmup_epochs`, `val_fraction`,
`stokes_loss_weights`, `patience`, `log_every`, `plot_every`,
`pca_fit_models`.

## `modules.inversion`

### Forward model

| Name | Description |
|------|-------------|
| `LineEmulator(checkpoint, wave)` | an emulator checkpoint plus the absolute wavelengths (Å) of its output |
| `SpectralSynthesizer(lines, psf=None, map_location="cpu", telluric=None)` | `forward(physical, mu, v_ref, wave_obs_rel, slab)` returns `(stokes (B, 4, nl), valid (B, nl))`; also `synthesize(physical, mu)`, `relative_wavelengths(wave)`, `phys_names`, `depth_size`, `slab_names`, `wave64`, `wave_ref` |
| `gaussian_psf(fwhm, step)` | normalized Gaussian kernel |
| `linear_interpolate(x, y, xq)`, `cubic_spline_interpolate(x, y, d2, xq)` | differentiable interpolation; both return `(values, inside_mask)` |

### Depth parameterization

| Name | Description |
|------|-------------|
| `node_interpolation_matrix(grid, nodes)` | `M (D, n)`: interpolating B-spline of degree `min(3, n-1)`, constant beyond the outer nodes |
| `interpolate_in_depth(x, logtau, profile)` | `np.interp` for a grid in either order |

### Configuration

| Name | Description |
|------|-------------|
| `VariableConfig(nodes, scale, bounds=None, regularization=(), fixed=False, depth=True)` | one inverted variable |
| `slab_variable(scale, bounds=None, regularization=(), fixed=False)` | a variable with one value per pixel |
| `InversionConfig(variables, stokes_weights, noise, optimizer, lr, lr_final_fraction, n_iter, chunk_size, reference_depth=-1.0, device, log_every, transverse_regularization=())` | optimizer settings |
| `training_statistics(models_file, percentiles=(0.1, 99.9), reference_index=22)` | `AtmosphereStatistics(logtau, median, lower, upper)` with `.bounds(name, nodes)` |
| `photosphere_variables(nodes, stats, scales=None, regularization=None, fixed=(), vz_bounds=(-20, 20), positive_bx=True)` | default photospheric `VariableConfig`s |

### Initial guess

| Name | Description |
|------|-------------|
| `initial_guess(observations, wave_obs, config, logtau, mean_stratification, wfa=None, cog_lines=(), bperp_floor=30.0)` | starting node values: median T/vmic, centre-of-gravity vz, weak-field B |
| `center_of_gravity_velocity(stokes (N, 4, nl), wave, lines)` | km/s, positive is a redshift |
| `weak_field_estimate(stokes, wave, WeakFieldConfig)` | per-pixel least-squares `(Bx, By, Bz)` |

### Inverters and results

| Name | Description |
|------|-------------|
| `Inverter(synthesizer, config, logtau, spatial_psf=None)` | `.run(observations, wave_obs, mu, init)` returns `InversionResult` |
| `Region(name, synthesizer, stokes, wavelength, mu=None, noise=None, stokes_weights=None, weight=1.0, wave_range=None, logtau=None, reference_depth=-1.0, spatial_psf=None)` | one spectral region; `Region.from_file(name, path, synthesizer, **kw)` |
| `MultiRegionInverter(regions, config, logtau, normalize=True)` | `.run(init, mu=None)` returns `MultiRegionResult` |
| `InversionResult` | `nodes`, `node_positions`, `stratification`, `logtau`, `v_ref`, `synthetic`, `chi2`, `history`, `synthetic_nopsf` |
| `save_result(path, result, wave_obs, observations=None)`, `save_multiregion_result(path, result)` | write HDF5 |

## `modules.regularization`

| Name | Description |
|------|-------------|
| `Regularizer(kind, weight, options={})` | `kind` ∈ `"l2"`, `"tv"`, `"l1"`, `"wavelet_l1"`; `weight` is a scalar or one value per node; `.prox(maps, step)`, `.value(maps)` |
| `TransverseRegularizer(bx, by, regularization)` | regularizes $B_t(\cos2\phi,\sin2\phi)$, invariant under the 180° ambiguity |
| `prox_l2_gradient`, `prox_tv`, `prox_l1`, `prox_wavelet_l1` | the proximal operators |

## `modules.hei` / `modules.hei_pca`

| Name | Description |
|------|-------------|
| `HeIPCAEmulator.load(path)` | He I slab emulator; `forward(pars (B, 7), x)` returns `(eps, eta, rho)` |
| `SiHeSynthesizer(photosphere, chromosphere, step=0.02, wave_range=None, psf=None, hazel_boundary_tilt=True, pumping="boundary", telluric=None, prefix="He_")` | photosphere plus He I slab(s); same interface as `SpectralSynthesizer` |
| `slab_operators(eta, rho, ds)`, `slab_formal_solution(...)` | exact constant-slab solution |
| `he_slab_variables(prefix="He_", fixed=("beta", "height"), field_bound=1500.0)` | default slab `VariableConfig`s |
| `he_slab_initial_guess(stokes, wave, mu, prefix="He_", height=3.0, ...)`, `he_slabs_initial_guess(..., prefixes, velocity_offsets=None)` | starting slab parameters |
| `i0_allen(wavelength, mu)` | Allen quiet-Sun continuum intensity |

## `modules.telluric`

| Name | Description |
|------|-------------|
| `TelluricLine(wavelength, prefix="Tel_")` | variables `<prefix>shift, width, depth, a`; `.transmission(wave_rel, params)` |
| `voigt(a, v)` | differentiable Voigt function $H(a,v)$ (Weideman 1994) |
| `telluric_variables(line, shift_range=0.3, fixed=())`, `telluric_initial_guess(stokes, wave, line, continuum=None, search=0.3, width=0.04, a=0.5)` | defaults and starting values |

## `modules.spatial_psf`

| Name | Description |
|------|-------------|
| `load_psf(path)` | PSF array from FITS, HDF5 or `.npy` |
| `crop_psf(psf, size)`, `enclosed_energy(psf, size)` | cropping helpers |
| `SpatialPSF(psf, size=None, boundary="reflect")` | FFT convolution of `(nx, ny, ...)` maps |
