# `invert_multiregion.py`: several spectral regions, one atmosphere

This script runs a joint inversion of Fe I 6301.5 + 6302.5 (with two
telluric O₂ lines) and Si I 10827 + He I 10830 (with a telluric line),
observed on the same pixels, using **one** model atmosphere. The inputs are
synthetic, with noise added.

```bash
python invert_multiregion.py                         # all regions jointly
python invert_multiregion.py fe630                   # only Fe I, same configuration
python invert_multiregion.py fe630 sihe10830 --iterations 1000 --output joint.h5
```

## Regions

A `Region` bundles observations, the synthesizer that models them, its noise
and its weight:

```python
wave_fe = np.linspace(6300.8921, 6303.2671, 150)
o2 = [TelluricLine(6302.0005, prefix="O2a_"), TelluricLine(6302.7629, prefix="O2b_")]
fe_synth = SpectralSynthesizer(
    [LineEmulator("weights/emulator_fe6301_pca_resmlp512_cosine_deriv.pt", wave_fe[wave_fe - LAMBDA_6301 < 0.5]),
     LineEmulator("weights/emulator_fe6302_pca_resmlp512_cosine_deriv.pt", wave_fe[wave_fe - LAMBDA_6302 > -0.5])],
    psf=None, telluric=o2)

tel_10832 = TelluricLine(10832.0, prefix="Tel_")
sihe_synth = SiHeSynthesizer(
    SpectralSynthesizer([LineEmulator("weights/emulator_si10827_pca_resmlp512_cosine_deriv.pt",
                                      np.linspace(10824.0, 10833.0, 350))]),
    HeIPCAEmulator.load("weights/emulator_he10830_pca.pt"),
    step=0.02, psf=None, pumping="boundary", telluric=tel_10832, prefix="He_")

REGIONS = {
    "fe630":     Region.from_file("fe630", "synthetic_fe6301_6302.h5", fe_synth, noise=1e-3, weight=1.0),
    "sihe10830": Region.from_file("sihe10830", "synthetic_si10827_he10830.h5", sihe_synth, noise=1e-3, weight=1.0),
}
```

```{note}
The script as distributed refers to the checkpoints by their training-time
names (`emulator_fe6301_pca.pt`, ...). Point them at the files in `weights/`
as shown above.
```

`Region` fields: `stokes (nx, ny, 4, nl)`, `wavelength`, `mu`, `noise`
(scalar or per Stokes), `stokes_weights`, `weight`, `wave_range` (fit only
part of the region), `logtau` (if its emulators use another depth grid),
`reference_depth` (`None` for emulators trained with absolute velocities),
and `spatial_psf`.

## Shared variables

Variables are **shared by name**:

* every region uses the photospheric nodes `T, vz, vmic, Bx, By, Bz`, each on
  its own emulators' depth grid with its own velocity reference;
* extra components bring their own prefixed variables: `He_*` (slab),
  `Tel_*` (10832 Å), `O2a_*`, `O2b_*` (6302 Å O₂ lines).

Using the same prefix in two regions ties a component together, and
different prefixes keep components apart.

```python
stats = training_statistics(MODELS)
nodes = {"T": [-4.5, -3.0, -1.5, 0.0], "vz": [-3.0, -1.0, 0.5], "vmic": [-1.0],
         "Bx": [-2.5, -0.5], "By": [-2.5, -0.5], "Bz": [-2.5, -0.5]}
variables = (photosphere_variables(nodes, stats)
             | he_slab_variables("He_", fixed=("beta", "height"))
             | telluric_variables(tel_10832)
             | telluric_variables(o2[0], shift_range=0.05, fixed=("a",))
             | telluric_variables(o2[1], shift_range=0.05, fixed=("a",)))
# keep only the variables used by the selected regions
```

These nodes span only the formation region of the lines. The stratification
is constant above the top node and below the bottom node (see
[B-splines](../bsplines.md#from-nodes-to-stratification)). Three `vz` nodes
give a quadratic velocity profile.

## Objective

With `normalize=True` (the default of `MultiRegionInverter`):

$$
J = \sum_\text{pixels}\sum_r \mathrm{weight}_r\,\frac{\chi^2_r}{n_r} + \text{regularization},
$$

so each region counts by its weight, independently of its number of
wavelengths. Regularization weights are then relative to a $\chi^2$ per
point, which is $\mathcal O(1)$ for a good fit. They are therefore much
smaller than in the single-region scripts, for example
`regularization={"T": [Regularizer("l2", 1.0 / 100.0 ** 2)]}` in
`photosphere_variables`.

## Initial guess

The photosphere is initialized from the first selected region that has a
weak-field configuration. Fe I uses the 6301/6302 WFA with centre-of-gravity
windows of 0.25/0.2 Å. Si I uses its own WFA. The He I slab and telluric
lines are initialized from their own regions (`he_slab_initial_guess`,
`telluric_initial_guess`).

## Adding a region

1. Build its synthesizer and a `Region` (for example Ca II 8542 with
   `LineEmulator("weights/emulator_caii8542_pca_resmlp512_cosine_deriv.pt", ...)`).
2. Add any new variables its synthesizer needs (`synthesizer.slab_names`).
3. Add their initial values to `init`.

If its emulator was trained with absolute velocities, give the region
`reference_depth=None`.

## Output

`save_multiregion_result` writes the shared `nodes/`, `node_positions/`,
`stratification/`, `logtau`, `mu`, `objective` and `history/`, plus
`regions/<name>/{synthetic, observations, wavelength, chi2, v_ref}` for each
region.
