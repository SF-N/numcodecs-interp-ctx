[![image](https://img.shields.io/github/actions/workflow/status/SF-N/numcodecs-interp-ctx/ci.yml?branch=main)](https://github.com/SF-N/numcodecs-interp-ctx/actions/workflows/ci.yml?query=branch%3Amain)
[![image](https://img.shields.io/pypi/v/numcodecs-interp-ctx.svg)](https://pypi.python.org/pypi/numcodecs-interp-ctx)
[![image](https://img.shields.io/pypi/l/numcodecs-interp-ctx.svg)](https://github.com/SF-N/numcodecs-interp-ctx/blob/main/LICENSE)
[![image](https://img.shields.io/python/required-version-toml?tomlFilePath=https%3A%2F%2Fraw.githubusercontent.com%2FSF-N%2Fnumcodecs-interp-ctx%2Frefs%2Fheads%2Fmain%2Fpyproject.toml)](https://pypi.python.org/pypi/numcodecs-interp-ctx)
[![image](https://readthedocs.org/projects/numcodecs-interp-ctx/badge/?version=latest)](https://numcodecs-interp-ctx.readthedocs.io/en/latest/?badge=latest)

# numcodecs-interp-ctx

`InterpolationContextMixingCodec` for the [`numcodecs`] buffer compression API.

The `InterpolationContextMixingCodec` is a lossy codec with a pointwise absolute error bound (`|x_dec - x| <= eb`) for smooth gridded data, interpreted as `[..., rows, cols]`. Like SZ3's interpolation mode, each slice is coded coarse-to-fine: the coarsest sub-grid is predicted causally, then every refinement level predicts the new points by cubic (or linear, chosen per pass) interpolation from already reconstructed points on both sides. The prediction residuals are quantised in the loop with step `2 * eb` and coded with a binary arithmetic coder driven by PAQ-style context mixing (contexts: level and axis, stencil gradient and curvature, neighbouring residuals in the current and previous slice), built on the primitives of [`numcodecs-context-mixing`](https://github.com/SF-N/numcodecs-context-mixing). Two-sided interpolation prediction is much more accurate than causal pixel prediction for smooth fields at low bitrates, where this codec typically outperforms wavelet coders.

```python
from numcodecs_interp_ctx import InterpolationContextMixingCodec

codec = InterpolationContextMixingCodec(eb=0.5)  # e.g. wind speed with a 0.5 m/s bound
```

NaN and infinite values are not supported; combine with a masking meta-codec such as [`numcodecs-mask`](https://numcodecs-mask.readthedocs.io) to remove them first. The codec is implemented with [numba](https://numba.pydata.org) and optimised for compression ratio rather than speed.

[`numcodecs`]: https://numcodecs.readthedocs.io/en/stable/

## License

Licensed under the Mozilla Public License, Version 2.0 ([LICENSE](LICENSE) or https://www.mozilla.org/en-US/MPL/2.0/).


## Funding

The `numcodecs-interp-ctx` package has been developed as part of [ESiWACE3](https://www.esiwace.eu), the third phase of the Centre of Excellence in Simulation of Weather and Climate in Europe.

Funded by the European Union. This work has received funding from the European High Performance Computing Joint Undertaking (JU) under grant agreement No 101093054.
