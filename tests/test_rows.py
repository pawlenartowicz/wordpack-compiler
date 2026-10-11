import numpy as np
import pytest

from wordpack_compiler.build import compute_abtt, quantize
from wordpack_compiler.reader import dequantize


def _unit(x):
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def _int8_rows(n=300, d=16, seed=0):
    rng = np.random.default_rng(seed)
    # a shared offset makes the mean and top component non-trivial, as in real embeddings
    x = rng.standard_normal((n, d)) + 2.0 * rng.standard_normal(d)
    return quantize(_unit(x))


def _reference_abtt(x0, m):
    """Float64 ABTT as ssdiff's Embeddings.normalize does it: centre, eigh, remove top m, L2."""
    x = x0.astype(np.float64)
    mean = x.mean(axis=0)
    xc = x - mean
    _, vecs = np.linalg.eigh(xc.T @ xc)
    top = vecs[:, ::-1][:, :m].T
    y = xc - (xc @ top.T) @ top
    return _unit(y)


def test_quantize_range():
    q = _int8_rows()
    assert q.dtype == np.int8
    assert q.min() > -128
    assert (np.abs(q.astype(int)).max(axis=1) == 127).all()


def test_dequantize_m0_is_unit_and_close_to_source():
    rng = np.random.default_rng(1)
    src = _unit(rng.standard_normal((200, 16)))
    x = dequantize(quantize(src), None, 0)
    assert x.dtype == np.float32
    assert np.allclose(np.linalg.norm(x, axis=1), 1, atol=1e-6)
    assert (np.sum(x * src, axis=1) > 0.999).all()


@pytest.mark.parametrize("m", [1, 3, 10])
def test_dequantize_matches_float64_abtt(m):
    q = _int8_rows()
    abtt = compute_abtt(q)
    got = dequantize(q, abtt, m)
    want = _reference_abtt(dequantize(q, None, 0), m)
    assert np.abs(got - want).max() < 1e-5


def test_compute_abtt_shape_and_mean():
    q = _int8_rows()
    abtt = compute_abtt(q)
    assert abtt.shape == (11, 16) and abtt.dtype == np.float32 and abtt.flags.c_contiguous
    assert np.allclose(abtt[0], dequantize(q, None, 0).mean(axis=0), atol=1e-6)
    assert np.allclose(abtt[1:] @ abtt[1:].T, np.eye(10), atol=1e-5)


def test_dequantize_rejects_bad_m():
    q = _int8_rows()
    abtt = compute_abtt(q)
    with pytest.raises(ValueError):
        dequantize(q, abtt, 11)
    with pytest.raises(ValueError):
        dequantize(q, abtt, -1)
    with pytest.raises(ValueError):
        dequantize(q, None, 1)


def test_compute_abtt_needs_ten_dimensions():
    with pytest.raises(ValueError):
        compute_abtt(quantize(_unit(np.random.default_rng(0).standard_normal((50, 8)))))
