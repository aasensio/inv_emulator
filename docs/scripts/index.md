# Inversion scripts

Every `invert_*.py` script is a self-contained, editable configuration of the
same machinery ([Inversion](../inversion.md),
[Configuring an inversion](../configuration.md)). They differ in the
emulators, the data, the node placement and the regularization.

| Script | Lines | Synthesizer | Inverter |
|--------|-------|-------------|----------|
| [`invert_fe6301_6302.py`](fe6301_6302.md) | Fe I 6301.5 + 6302.5 | `SpectralSynthesizer` with 2 emulators | `Inverter`, optional spatial PSF |
| [`invert_si10827_he10830.py`](si10827_he10830.md) | Si I 10827 + He I 10830 (2 slabs) + telluric | `SiHeSynthesizer` | `Inverter` |
| [`invert_si10827_he10830_l1.py`](#variant-l1) | same, L1-sparse second slab | `SiHeSynthesizer` | `Inverter` |
| [`invert_multiregion.py`](multiregion.md) | Fe I + Si I/He I jointly | several `Region`s | `MultiRegionInverter` |
| [`invert_caii8542.py`](caii8542.md) | Ca II 8542 (CRISP) | `SpectralSynthesizer` + FPI profile | `Inverter` |

## Common skeleton

```python
# 1. forward model
synthesizer = SpectralSynthesizer([LineEmulator(ckpt, wave), ...], psf=...)      # or SiHeSynthesizer(...)

# 2. depth grid, median stratification and percentile bounds from the training models
logtau, mean_strat, lo_strat, hi_strat = ...;  lo_strat["Bx"] *= 0

# 3. variables: nodes in log tau, scale, bounds, regularization
REG_STRENGTH = 5.0
smooth = lambda jump: [Regularizer("l2", REG_STRENGTH / np.asarray(jump) ** 2)]
variables = {"T": VariableConfig(nodes_T, 500.0, training_bounds("T", nodes_T), smooth(20.0)), ...}

# 4. optimizer
config = InversionConfig(variables=variables, noise={...}, lr=0.1, n_iter=500, chunk_size=...)

# 5. observations (nx, ny, 4, nl), absolute wavelengths, mu
# 6. initial guess (weak field + centre of gravity + median stratification [+ He I, telluric])
init = initial_guess(observations, wave_obs, config, logtau, mean_strat, wfa=wfa, cog_lines=[...])

# 7. run and save
result = Inverter(synthesizer, config, logtau).run(observations, wave_obs, mu, init)
save_result(output, result, wave_obs, observations)
```

Run any of them from the repository root after editing the paths:

```bash
python invert_fe6301_6302.py
```

```{toctree}
:hidden:

fe6301_6302
si10827_he10830
multiregion
caii8542
```
