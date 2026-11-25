# inv_emulator

Neural emulators of Si I 10827 Å, He I 10830 Å and Fe I 6301.5/6302.5 Å for fast,
differentiable Stokes synthesis and spatially regularized inversion of full
fields of view.

---

## Purpose

An inversion of spectropolarimetric data looks for the model atmosphere whose
emergent Stokes profiles best match the observed ones. Codes such as SIR or
Hazel solve the polarized radiative transfer equation at every iteration of
every pixel. That is accurate but slow. It also makes it hard to couple
neighbouring pixels through spatial regularization, or to account for the
telescope's spatial PSF.

This code replaces the radiative transfer step with **emulators**. These are
neural networks trained on millions of synthetic profiles. Each one returns the
Stokes profiles of a model atmosphere in a single forward pass, and the pass
can be differentiated with PyTorch's autograd. As a result:

* **Synthesis** of a batch of atmospheres costs a few matrix multiplications
  per pixel.
* **Inversion** of a full map (10⁵–10⁶ pixels) runs on a GPU. All pixels are
  optimized simultaneously by gradient descent. Spatial regularizers are
  applied as proximal steps on the maps of every inverted quantity, and a
  telescope PSF can be included.

## Emulators

| Emulator | Region | Role |
|----------|--------|------|
| **Si I 10827** | 10824–10833 Å | photospheric line in the He I 10830 window |
| **Fe I 6301** and **Fe I 6302** | 6300.9–6303.3 Å | the Hinode SP doublet (one emulator per line) |
| **He I 10830** | 10823–10834 Å | chromospheric constant-property slab (emits ε, η, ρ), combined with an exact formal solution |
| **Ca II 8542** | 8540-8544 Å | chromospheric line |

## The two representations

The whole pipeline relies on two low-dimensional representations:

1. **B-splines for the depth stratification.** Each physical variable $T,
   v_z, v_\mathrm{mic}, B_x, B_y, B_z$ is a function of $\log\tau$.
   * Inside the emulator, the 80-point stratification is projected by least
     squares onto 10 cubic B-splines, which are the network inputs.
   * In the inversion, each variable is described by its values at a few
     *nodes*. An interpolating B-spline through them gives the 80-point
     stratification.

   See [B-splines for the depth stratification](bsplines.md).
2. **PCA for the Stokes profiles.** Each Stokes parameter, after an intensity
   normalization and symlog compression for Q, U and V, is expanded in its
   own principal-component basis learned from the training profiles. The
   network predicts 20–50 PCA coefficients per Stokes parameter rather than
   hundreds of wavelength samples. See
   [PCA of the Stokes profiles](pca.md).

```
                inversion                               emulator (one per line)
  ┌──────────────────────────────┐   ┌───────────────────────────────────────────────────────────┐
  node values ─▶ interpolating ─▶ stratification ─▶ P (B-spline     ─▶ residual ─▶ PCA        ─▶ Stokes
  (few / var)    B-spline M        (6 × 80)          projection,        MLP         reconstruction   (4 × L)
                                                    10 coeffs/var       6×512       + inverse symlog
```

## Contents

```{toctree}
:maxdepth: 2
:caption: Background

installation
architecture
bsplines
pca
emulators
```

```{toctree}
:maxdepth: 2
:caption: Using the code

synthesis
inversion
configuration
scripts/index
output
api
```
