"""Tests for gagf.rnns.utils module."""

import pytest
import numpy as np

from gagf.rnns.utils import (
    get_power_1d,
    get_power_2d_adele,
    topk_template_freqs_1d,
    topk_template_freqs,
)


class TestGetPower1D:
    """Tests for get_power_1d function."""

    def test_output_shape(self):
        """Test that output shape is correct."""
        p = 10
        signal = np.random.randn(p)
        
        power, freqs = get_power_1d(signal)
        
        expected_len = p // 2 + 1
        assert power.shape == (expected_len,), f"power shape mismatch: {power.shape}"
        assert freqs.shape == (expected_len,), f"freqs shape mismatch: {freqs.shape}"

    def test_parseval_theorem(self):
        """Test that Parseval's theorem holds (total power ≈ norm squared)."""
        p = 16
        signal = np.random.randn(p)
        
        power, _ = get_power_1d(signal)
        total_power = np.sum(power)
        norm_squared = np.linalg.norm(signal) ** 2
        
        np.testing.assert_allclose(
            total_power, norm_squared, rtol=1e-6,
            err_msg="Parseval's theorem violated"
        )

    def test_parseval_theorem_odd_length(self):
        """Test Parseval's theorem for odd-length signals."""
        p = 15
        signal = np.random.randn(p)
        
        power, _ = get_power_1d(signal)
        total_power = np.sum(power)
        norm_squared = np.linalg.norm(signal) ** 2
        
        np.testing.assert_allclose(
            total_power, norm_squared, rtol=1e-6,
            err_msg="Parseval's theorem violated for odd length"
        )

    def test_dc_component(self):
        """Test that DC component power is correct for constant signal."""
        p = 8
        constant_value = 3.0
        signal = np.full(p, constant_value)
        
        power, freqs = get_power_1d(signal)
        
        # DC component should contain all the power for constant signal
        expected_dc_power = constant_value ** 2 * p
        np.testing.assert_allclose(power[0], expected_dc_power, rtol=1e-6)
        
        # All other components should be zero
        assert np.allclose(power[1:], 0, atol=1e-10)


class TestGetPower2DAdele:
    """Tests for get_power_2d_adele function."""

    def test_output_shape(self):
        """Test that output shape is correct."""
        M, N = 8, 10
        signal = np.random.randn(M, N)
        
        freqs_u, freqs_v, power = get_power_2d_adele(signal)
        
        expected_power_shape = (M, N // 2 + 1)
        assert power.shape == expected_power_shape, f"power shape mismatch: {power.shape}"
        assert freqs_u.shape == (M,), f"freqs_u shape mismatch: {freqs_u.shape}"
        assert freqs_v.shape == (N // 2 + 1,), f"freqs_v shape mismatch: {freqs_v.shape}"

    def test_output_shape_no_freq(self):
        """Test output when no_freq=True."""
        M, N = 8, 10
        signal = np.random.randn(M, N)
        
        result = get_power_2d_adele(signal, no_freq=True)
        
        # Should only return power
        expected_shape = (M, N // 2 + 1)
        assert result.shape == expected_shape

    def test_parseval_theorem(self):
        """Test that Parseval's theorem holds."""
        M, N = 12, 12
        signal = np.random.randn(M, N)
        
        power = get_power_2d_adele(signal, no_freq=True)
        total_power = np.sum(power)
        norm_squared = np.linalg.norm(signal) ** 2
        
        np.testing.assert_allclose(
            total_power, norm_squared, rtol=1e-6,
            err_msg="Parseval's theorem violated for 2D"
        )

    def test_parseval_theorem_rectangular(self):
        """Test Parseval's theorem for rectangular arrays."""
        M, N = 7, 11  # Both odd
        signal = np.random.randn(M, N)
        
        power = get_power_2d_adele(signal, no_freq=True)
        total_power = np.sum(power)
        norm_squared = np.linalg.norm(signal) ** 2
        
        np.testing.assert_allclose(
            total_power, norm_squared, rtol=1e-6,
            err_msg="Parseval's theorem violated for rectangular array"
        )


class TestTopkTemplateFreqs1D:
    """Tests for topk_template_freqs_1d function."""

    def test_returns_top_k(self):
        """Test that function returns exactly K frequencies."""
        p = 16
        K = 3
        template = np.random.randn(p)
        
        top_freqs = topk_template_freqs_1d(template, K)
        
        assert len(top_freqs) == K, f"Expected {K} frequencies, got {len(top_freqs)}"

    def test_returns_sorted_by_power(self):
        """Test that frequencies are sorted by descending power."""
        p = 16
        K = 5
        template = np.random.randn(p)
        
        top_freqs = topk_template_freqs_1d(template, K)
        power, _ = get_power_1d(template)
        
        # Get powers for returned frequencies
        returned_powers = [power[f] for f in top_freqs]
        
        # Should be in descending order
        assert returned_powers == sorted(returned_powers, reverse=True)

    def test_empty_for_zero_signal(self):
        """Test that zero signal with high min_power returns empty list."""
        p = 8
        template = np.zeros(p)
        
        top_freqs = topk_template_freqs_1d(template, K=3, min_power=1e-10)
        
        assert top_freqs == []

    def test_handles_k_larger_than_freqs(self):
        """Test behavior when K is larger than available frequencies."""
        p = 6
        K = 10  # More than available frequencies
        template = np.random.randn(p)
        
        top_freqs = topk_template_freqs_1d(template, K)
        
        # Should return at most p//2 + 1 frequencies
        assert len(top_freqs) <= p // 2 + 1


class TestTopkTemplateFreqs:
    """Tests for topk_template_freqs function (2D)."""

    def test_returns_top_k(self):
        """Test that function returns exactly K frequency pairs."""
        p1, p2 = 8, 8
        K = 3
        template = np.random.randn(p1, p2)
        
        top_freqs = topk_template_freqs(template, K)
        
        assert len(top_freqs) == K, f"Expected {K} frequency pairs, got {len(top_freqs)}"

    def test_returns_tuples(self):
        """Test that returned values are (kx, ky) tuples."""
        p1, p2 = 8, 8
        K = 3
        template = np.random.randn(p1, p2)
        
        top_freqs = topk_template_freqs(template, K)
        
        for freq in top_freqs:
            assert isinstance(freq, tuple), f"Expected tuple, got {type(freq)}"
            assert len(freq) == 2, f"Expected 2-tuple, got {len(freq)}-tuple"

    def test_empty_for_zero_signal(self):
        """Test that zero signal returns empty list."""
        p1, p2 = 6, 6
        template = np.zeros((p1, p2))
        
        top_freqs = topk_template_freqs(template, K=3, min_power=1e-10)
        
        assert top_freqs == []
