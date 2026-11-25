# Output and visualization

## `save_result` (single region)

`save_result(path, result, wave_obs, observations)` writes an HDF5 file:

| Dataset | Shape | Content |
|---------|-------|---------|
| `nodes/<var>` | (nx, ny, n_nodes) | inverted node values in physical units; `vz` is **absolute**. Per-pixel variables (`He*_...`, `Tel_...`) have `n_nodes = 1` |
| `node_positions/<var>` | (n_nodes,) | $\log\tau$ of the nodes (0 for per-pixel variables) |
| `stratification/<var>` | (nx, ny, 80) | interpolating B-spline through the nodes on the depth grid (absolute `vz`) |
| `logtau` | (80,) | depth grid (1.5 … −7.5) |
| `v_ref` | (nx, ny) | `vz` at $\log\tau=-1$, the Doppler shift applied to the spectrum (km/s) |
| `synthetic` | (nx, ny, 4, nl) | best-fit profiles at the observed wavelengths (NaN where not synthesized); convolved with the spatial PSF if one was used |
| `synthetic_nopsf` | (nx, ny, 4, nl) | only with a spatial PSF: the profiles before the convolution |
| `chi2` | (nx, ny) | weighted $\chi^2$ per valid point |
| `wavelength` | (nl,) | observed wavelengths |
| `observations` | (nx, ny, 4, nl) | the fitted data |
| `history_chi2` | (n_iter,) | data term per iteration |

```{note}
The stratification of the emulator input is $v_z - v_\mathrm{ref}$. The
file stores the absolute $v_z$, which is what a classical inversion code
would report.
```

Reading it back:

```python
import h5py
with h5py.File("inversions/inversion_result_fe6301_6302_nopsf.h5") as f:
    T = f["stratification/T"][:]          # (nx, ny, 80)
    logtau = f["logtau"][:]
    Bz_nodes = f["nodes/Bz"][:]          # (nx, ny, 2)
    chi2 = f["chi2"][:]
i = np.argmin(np.abs(logtau + 1.0))
T_at_m1 = T[:, :, i]                     # temperature map at log tau = -1
```

## `save_multiregion_result`

This writes the shared `nodes/`, `node_positions/`, `stratification/`,
`logtau`, `mu`, `objective` (data term per pixel) and `history/<quantity>`
(`chi2`, `chi2_<region>`, `regularization`, `lr`), plus for every region
`regions/<name>/synthetic`, `observations`, `wavelength`, `chi2`, `v_ref`
(and `synthetic_nopsf`).

## `view_inversion.py`

An interactive viewer (pyqtgraph and PyQt6) for any single-region result:

```bash
python view_inversion.py inversions/inversion_result_si10827_he10830.h5 [--telluric 10832.0]
```

* **Top row:** a map of a photospheric variable at the $\log\tau$ chosen
  with a slider, and a map of a per-pixel quantity (He I slab, telluric line
  parameter, or $\chi^2$).
* **Bottom:** hovering over a map selects a pixel. The plots show the depth
  stratification of every photospheric variable (line) with its node values
  (dots), and the observed (grey) and synthetic (red) Stokes profiles.
* Clicking freezes or unfreezes the pixel. Zoom and pan are linked between
  the maps.
