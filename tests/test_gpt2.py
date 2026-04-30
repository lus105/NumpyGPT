import numpy as np
import pytest

from src.gpt2 import attention, gelu, layer_norm, linear, softmax


class TestGelu:
    def test_zero(self):
        assert gelu(0.0) == pytest.approx(0.0)

    def test_large_positive_approaches_identity(self):
        x = np.float64(10.0)
        assert gelu(x) == pytest.approx(x, rel=1e-3)

    def test_large_negative_approaches_zero(self):
        assert gelu(-10.0) == pytest.approx(0.0, abs=1e-4)

    def test_array_shape_preserved(self):
        x = np.array([[1.0, -1.0], [2.0, -2.0]])
        assert gelu(x).shape == x.shape


class TestSoftmax:
    def test_sums_to_one(self):
        x = np.array([1.0, 2.0, 3.0])
        np.testing.assert_allclose(softmax(x).sum(), 1.0, rtol=1e-6)

    def test_all_values_positive(self):
        x = np.array([-100.0, 0.0, 100.0])
        assert np.all(softmax(x) > 0)

    def test_numerical_stability_with_large_values(self):
        x = np.array([1000.0, 1001.0, 1002.0])
        result = softmax(x)
        assert np.all(np.isfinite(result))
        np.testing.assert_allclose(result.sum(), 1.0, rtol=1e-6)

    def test_2d_rows_sum_to_one(self):
        x = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        row_sums = softmax(x).sum(axis=-1)
        np.testing.assert_allclose(row_sums, [1.0, 1.0], rtol=1e-6)


class TestLayerNorm:
    def test_normalizes_to_zero_mean_unit_var(self):
        rng = np.random.default_rng(0)
        x = rng.standard_normal((5, 16))
        g = np.ones(16)
        b = np.zeros(16)
        out = layer_norm(x, g, b)
        np.testing.assert_allclose(out.mean(axis=-1), 0.0, atol=1e-6)
        np.testing.assert_allclose(out.var(axis=-1), 1.0, atol=1e-4)

    def test_scale_and_shift_applied(self):
        x = np.array([[1.0, 2.0, 3.0, 4.0]])
        g = np.full(4, 2.0)
        b = np.full(4, 1.0)
        out = layer_norm(x, g, b)
        # scale doubles the std, shift adds 1 to mean
        np.testing.assert_allclose(out.mean(axis=-1), 1.0, atol=1e-6)


class TestLinear:
    def test_output_shape(self):
        x = np.ones((5, 4))
        w = np.ones((4, 8))
        b = np.zeros(8)
        assert linear(x, w, b).shape == (5, 8)

    def test_known_values(self):
        x = np.array([[1.0, 0.0], [0.0, 1.0]])
        w = np.array([[1.0, 2.0], [3.0, 4.0]])
        b = np.array([10.0, 20.0])
        expected = np.array([[11.0, 22.0], [13.0, 24.0]])
        np.testing.assert_allclose(linear(x, w, b), expected)


class TestAttention:
    def test_output_shape(self):
        rng = np.random.default_rng(0)
        n_q, n_k, d_k, d_v = 4, 4, 8, 8
        q = rng.standard_normal((n_q, d_k))
        k = rng.standard_normal((n_k, d_k))
        v = rng.standard_normal((n_k, d_v))
        mask = (1 - np.tri(n_q)) * -1e10
        assert attention(q, k, v, mask).shape == (n_q, d_v)

    def test_causal_mask_prevents_future_leakage(self):
        rng = np.random.default_rng(1)
        n = 4
        d = 8
        q = rng.standard_normal((n, d))
        k = rng.standard_normal((n, d))
        v = rng.standard_normal((n, d))
        mask = (1 - np.tri(n, dtype=float)) * -1e10

        out1 = attention(q, k, v, mask)

        # Modifying future keys/values must not change outputs at earlier positions
        k2, v2 = k.copy(), v.copy()
        k2[1:] += 999.0
        v2[1:] += 999.0
        out2 = attention(q, k2, v2, mask)

        np.testing.assert_allclose(out1[0], out2[0], rtol=1e-5)
