import numcodecs
import numcodecs.registry
import numpy as np
import pytest


def test_from_config():
    codec = numcodecs.registry.get_codec(dict(id="interp_ctx", eb=0.5))
    assert codec.__class__.__name__ == "InterpolationContextMixingCodec"
    assert codec.__class__.__module__ == "numcodecs_interp_ctx"
    config = codec.get_config()
    assert config == dict(
        id="interp_ctx", eb=0.5, levels=None, mixer_rate=0.008, model_floor=1 / 256
    )
    assert numcodecs.registry.get_codec(config).get_config() == config


def test_invalid():
    from numcodecs_interp_ctx import InterpolationContextMixingCodec

    with pytest.raises(ValueError):
        InterpolationContextMixingCodec(eb=0.0)
    with pytest.raises(ValueError):
        InterpolationContextMixingCodec(eb=1.0, levels=0)
    with pytest.raises(ValueError):
        InterpolationContextMixingCodec(eb=1.0, mixer_rate=-1.0)
    with pytest.raises(TypeError):
        InterpolationContextMixingCodec(eb=1.0).encode(np.arange(10))
    with pytest.raises(ValueError):
        InterpolationContextMixingCodec(eb=1.0).encode(np.array([1.0, np.nan]))


def test_below_resolution_is_lossless():
    from numcodecs_interp_ctx import InterpolationContextMixingCodec

    # an error bound below the float32 spacing (for the larger values) is still
    # satisfied, since the reconstruction is rounded back to the original
    data = np.linspace(-30.0, 30.0, 1000, dtype=np.float32).reshape(20, 50)
    codec = InterpolationContextMixingCodec(eb=1e-7)
    decoded = np.asarray(codec.decode(codec.encode(data)))
    assert np.all(np.abs(decoded.astype(np.float64) - data.astype(np.float64)) <= 1e-7)
    np.testing.assert_array_equal(decoded[np.abs(data) > 1.0], data[np.abs(data) > 1.0])


def smooth_field(shape, seed=0, noise=0.0):
    rng = np.random.default_rng(seed)
    grids = np.meshgrid(*[np.linspace(0, 1, s) for s in shape], indexing="ij")
    field = (
        20.0 * np.sin(3 * grids[-1]) * np.cos(2 * grids[-2])
        if len(shape) >= 2
        else 20.0 * np.sin(3 * grids[0])
    )
    if len(shape) == 3:
        field = field + 5.0 * grids[0]
    return field + noise * rng.normal(size=shape)


def check_roundtrip(data: np.ndarray, eb: float, **kwargs):
    codec = numcodecs.registry.get_codec(dict(id="interp_ctx", eb=eb, **kwargs))

    encoded = codec.encode(data)
    decoded = np.asarray(codec.decode(encoded))

    assert decoded.dtype == data.dtype
    assert decoded.shape == data.shape
    assert np.all(np.abs(decoded.astype(np.float64) - data.astype(np.float64)) <= eb)

    out = np.empty_like(data)
    codec.decode(encoded, out=out)
    np.testing.assert_array_equal(out, decoded)

    return len(encoded)


def test_roundtrip():
    data = smooth_field((61, 83), noise=0.1)
    for eb in (1.0, 0.1, 0.01, 1e-4):
        check_roundtrip(data, eb)
    check_roundtrip(data.astype(np.float32), 0.01)
    check_roundtrip(data, 0.1, levels=2)
    check_roundtrip(data, 0.1, levels=6)
    check_roundtrip(smooth_field((3, 40, 50), noise=0.05), 0.05)
    check_roundtrip(smooth_field((97,)), 0.1)
    check_roundtrip(np.array(3.5), 0.1)
    check_roundtrip(np.zeros((0, 4)), 0.1)
    check_roundtrip(np.full((7, 9), 1e6), 0.5)
    check_roundtrip(1e6 + smooth_field((30, 30)), 1e-3)


def test_compression():
    data = smooth_field((200, 300))
    size = check_roundtrip(data, 0.05)
    # a smooth field compresses to well below one bit per value
    assert size < data.size / 8
    # a finer bound needs more bits
    assert check_roundtrip(data, 0.005) > size


def test_masked():
    from numcodecs_mask import MaskMetaCodec

    rng = np.random.default_rng(5)
    data = smooth_field((60, 80), noise=0.05)
    mask = rng.random(data.shape) < 0.3
    mask[10:20, 30:50] = True
    values = data.copy()
    values[mask] = np.nan

    codec = MaskMetaCodec(
        mask=np.nan,
        codec=dict(id="interp_ctx", eb=0.05),
        bitmap_codec=dict(id="packbits"),
    )
    encoded = codec.encode(values)
    decoded = np.asarray(codec.decode(encoded))
    np.testing.assert_array_equal(np.isnan(decoded), mask)
    assert np.all(np.abs(decoded[~mask] - data[~mask]) <= 0.05)

    # a lot cheaper than coding a fill value for the masked points
    filled = np.where(mask, 0.0, data)
    plain = numcodecs.registry.get_codec(dict(id="interp_ctx", eb=0.05))
    assert len(encoded) < len(plain.encode(filled))
