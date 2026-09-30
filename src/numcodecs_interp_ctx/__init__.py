"""
[`InterpolationContextMixingCodec`][numcodecs_interp_ctx.InterpolationContextMixingCodec] for the [`numcodecs`][numcodecs] buffer compression API.
"""

__all__ = ["InterpolationContextMixingCodec"]

import math
from functools import reduce
from io import BytesIO

import leb128
import numcodecs.compat
import numcodecs.registry
import numpy as np
from numcodecs.abc import Codec
from typing_extensions import Buffer  # MSPV 3.12

from . import _interp


def _as_slices(shape: tuple[int, ...]) -> tuple[int, int, int]:
    if len(shape) == 0:
        return (1, 1, 1)
    if len(shape) == 1:
        return (1, 1, shape[0])
    return (reduce(lambda a, b: a * b, shape[:-2], 1), shape[-2], shape[-1])


class InterpolationContextMixingCodec(Codec):
    """
    Lossy codec with a pointwise absolute error bound for smooth gridded data,
    combining hierarchical interpolation prediction with context-mixing
    arithmetic coding.

    Like SZ3's interpolation mode, every 2D slice of the data (interpreted as
    `[..., rows, cols]`) is coded coarse-to-fine: the coarsest sub-grid
    (stride `2**levels`) is predicted causally with a median edge detector,
    then each refinement level predicts the new points by cubic (or linear,
    chosen per pass by the encoder) interpolation along one axis from already
    reconstructed points on both sides. Prediction residuals are quantised
    with step `2 * eb` in the loop (predictions only use reconstructed
    values), so that `|x_dec - x| <= eb` holds for every value, and coded with
    a binary arithmetic coder driven by PAQ-style context mixing (contexts:
    level and axis, stencil gradient and curvature, neighbouring residuals in
    the current and previous slice). The reconstruction is verified during
    encoding and the step is shrunk if floating-point rounding would exceed
    the bound.

    Compared to pixel-wise predictive coding, the two-sided interpolation
    prediction is much more accurate for smooth fields at low bitrates. NaN
    and infinite values are not supported; combine with a masking meta-codec
    such as [`numcodecs_mask.MaskMetaCodec`](https://numcodecs-mask.readthedocs.io)
    to remove them first.

    Parameters
    ----------
    eb : float
        The positive absolute error bound.
    levels : None | int, optional
        The number of refinement levels (coarsest stride `2**levels`), or
        [`None`][None] to derive it from the slice shape.
    mixer_rate : float, optional
        Learning rate of the logistic mixer.
    model_floor : float, optional
        Minimum adaptation rate of the context models' probabilities.
    """

    __slots__: tuple[str, ...] = ("_eb", "_levels", "_mixer_rate", "_model_floor")
    _eb: float
    _levels: None | int
    _mixer_rate: float
    _model_floor: float

    codec_id: str = "interp_ctx"  # type: ignore

    def __init__(
        self,
        *,
        eb: float,
        levels: None | int = None,
        mixer_rate: float = 0.008,
        model_floor: float = 1 / 256,
    ) -> None:
        if not (math.isfinite(eb) and eb > 0):
            raise ValueError("eb must be finite and positive")
        if levels is not None and (int(levels) < 1 or int(levels) > 30):
            raise ValueError("levels must be in [1, 30]")
        if not (math.isfinite(mixer_rate) and mixer_rate > 0):
            raise ValueError("mixer_rate must be finite and positive")
        if not (math.isfinite(model_floor) and 0 < model_floor <= 1):
            raise ValueError("model_floor must be in (0, 1]")

        self._eb = float(eb)
        self._levels = None if levels is None else int(levels)
        self._mixer_rate = float(mixer_rate)
        self._model_floor = float(model_floor)

    def _resolve_levels(self, Y: int, X: int) -> int:
        if self._levels is not None:
            return self._levels
        return max(1, int(math.floor(math.log2(max(Y, X, 1)))) - 2)

    def encode(self, buf: Buffer) -> bytes:
        """
        Encode the data in `buf`.

        Parameters
        ----------
        buf : Buffer
            Floating-point data to be encoded. May be any object supporting
            the new-style buffer protocol.

        Returns
        -------
        enc : bytes
            Encoded data as a bytestring.
        """

        a = numcodecs.compat.ensure_ndarray(buf)
        dtype, shape = a.dtype, a.shape

        if not np.issubdtype(dtype, np.floating):
            raise TypeError("can only encode floating point values")
        if not np.all(np.isfinite(a)):
            raise ValueError("cannot encode non-finite values, mask them first")

        T, Y, X = _as_slices(shape)
        x3 = np.ascontiguousarray(a.astype(np.float64).reshape(T, Y, X))
        levels = self._resolve_levels(Y, X)

        rec = np.zeros_like(x3)
        out = np.zeros(x3.size * 8 + 4096, np.uint8)
        dummy = np.zeros(1, np.uint8)

        # quantise in the loop; shrink the step if the rounding of the
        # reconstruction in the target dtype would exceed the bound
        step = 2.0 * self._eb * (1.0 - 1e-6)
        for _ in range(16):
            n = _interp.code_all(
                x3,
                rec,
                T,
                Y,
                X,
                step,
                levels,
                out,
                dummy,
                True,
                self._mixer_rate,
                self._model_floor,
            )
            error = np.abs(rec.astype(dtype).astype(np.float64) - x3)
            error_max = float(error.max()) if error.size > 0 else 0.0
            if error_max <= self._eb:
                break
            step *= (self._eb / error_max) * (1.0 - 1e-9)
        else:
            raise ValueError(
                f"cannot satisfy the error bound {self._eb} with dtype {dtype}, "
                "which does not resolve the data finely enough"
            )

        # message: dtype shape levels step rates coded
        message: list[bytes | bytearray] = []

        message.append(leb128.u.encode(len(dtype.str)))
        message.append(dtype.str.encode("ascii"))

        message.append(leb128.u.encode(len(shape)))
        for s in shape:
            message.append(leb128.u.encode(s))

        message.append(leb128.u.encode(levels))
        message.append(
            np.array([step, self._mixer_rate, self._model_floor], dtype="<f8").tobytes()
        )

        message.append(out[:n].tobytes())

        return b"".join(message)

    def decode(self, buf: Buffer, out: None | Buffer = None) -> Buffer:
        """
        Decode the data in `buf`.

        Parameters
        ----------
        buf : Buffer
            Encoded data. Must be an object representing a bytestring, e.g.
            [`bytes`][bytes] or a 1D array of [`np.uint8`][numpy.uint8]s etc.
        out : Buffer, optional
            Writeable buffer to store decoded data. N.B. if provided, this
            buffer must be exactly the right size to store the decoded data.

        Returns
        -------
        dec : Buffer
            Decoded data. May be any object supporting the new-style buffer
            protocol.
        """

        b = numcodecs.compat.ensure_bytes(buf)

        b_io = BytesIO(b)

        dtype = np.dtype(b_io.read(leb128.u.decode_reader(b_io)[0]).decode("ascii"))
        shape = tuple(
            leb128.u.decode_reader(b_io)[0]
            for _ in range(leb128.u.decode_reader(b_io)[0])
        )
        levels = leb128.u.decode_reader(b_io)[0]
        step, mixer_rate, model_floor = np.frombuffer(
            b_io.read(24), dtype="<f8", count=3
        )

        T, Y, X = _as_slices(shape)
        rec = np.zeros((T, Y, X), np.float64)
        # the range decoder may read a few bytes past the end of the stream
        inp = np.concatenate(
            [np.frombuffer(b_io.read(), np.uint8), np.zeros(16, np.uint8)]
        )
        dummy = np.zeros(1, np.uint8)
        _interp.code_all(
            rec,
            rec,
            T,
            Y,
            X,
            float(step),
            levels,
            dummy,
            inp,
            False,
            float(mixer_rate),
            float(model_floor),
        )

        decoded = rec.reshape(shape).astype(dtype)

        return numcodecs.compat.ndarray_copy(decoded, out)  # type: ignore

    def get_config(self) -> dict:
        """
        Returns the configuration of this codec.

        [`numcodecs.registry.get_codec(config)`][numcodecs.registry.get_codec]
        can be used to reconstruct this codec from the returned config.

        Returns
        -------
        config : dict
            Configuration of this codec.
        """

        return dict(
            id=type(self).codec_id,
            eb=self._eb,
            levels=self._levels,
            mixer_rate=self._mixer_rate,
            model_floor=self._model_floor,
        )

    def __repr__(self) -> str:
        return f"{type(self).__name__}(eb={self._eb!r}, levels={self._levels!r}, mixer_rate={self._mixer_rate!r}, model_floor={self._model_floor!r})"


numcodecs.registry.register_codec(InterpolationContextMixingCodec)
