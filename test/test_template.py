"""Tests for src.template module."""

import numpy as np
import pytest

import src.template as template


class TestOneHot:
    """Tests for template.one_hot function."""

    def test_output_shape(self):
        """Test that output shape is correct."""
        p = 7
        tpl = template.one_hot(p)

        assert tpl.shape == (p,), f"Expected shape ({p},), got {tpl.shape}"

    def test_mean_centered(self):
        """Test that the template is mean-centered."""
        p = 10
        tpl = template.one_hot(p)

        np.testing.assert_allclose(tpl.mean(), 0, atol=1e-10)

    def test_has_spike(self):
        """Test that template has a spike at index 1."""
        p = 5
        tpl = template.one_hot(p)

        zeroth_freq = 10 / p
        expected_spike_value = 10 - zeroth_freq

        np.testing.assert_allclose(tpl[1], expected_spike_value, rtol=1e-5)


class TestFixedCn:
    """Tests for template.fixed_cn function."""

    def test_output_shape(self):
        """Test that output shape is correct."""
        group_size = 8
        fourier_coef_mags = [0, 5, 3, 2, 1]

        tpl = template.fixed_cn(group_size, fourier_coef_mags)

        assert tpl.shape == (group_size,), f"Expected shape ({group_size},), got {tpl.shape}"

    def test_mean_centered(self):
        """Test that the template is mean-centered."""
        group_size = 10
        fourier_coef_mags = [0, 5, 3, 2]

        tpl = template.fixed_cn(group_size, fourier_coef_mags)

        np.testing.assert_allclose(tpl.mean(), 0, atol=1e-10)

    def test_real_valued(self):
        """Test that the template is real-valued."""
        group_size = 8
        fourier_coef_mags = [0, 5, 3]

        tpl = template.fixed_cn(group_size, fourier_coef_mags)

        assert np.isreal(tpl).all()


class TestFixedCnxcn:
    """Tests for template.fixed_cnxcn function."""

    def test_output_shape(self):
        """Test that output shape is correct (flattened)."""
        image_length = 6
        fourier_coef_mags = [0, 5, 3, 2]

        tpl = template.fixed_cnxcn(image_length, fourier_coef_mags)

        expected_size = image_length * image_length
        assert tpl.shape == (expected_size,), f"Expected shape ({expected_size},), got {tpl.shape}"

    def test_mean_centered(self):
        """Test that the template is mean-centered."""
        image_length = 5
        fourier_coef_mags = [0, 5, 3]

        tpl = template.fixed_cnxcn(image_length, fourier_coef_mags)

        np.testing.assert_allclose(tpl.mean(), 0, atol=1e-10)

    def test_real_valued(self):
        """Test that the template is real-valued."""
        image_length = 4
        fourier_coef_mags = [0, 5]

        tpl = template.fixed_cnxcn(image_length, fourier_coef_mags)

        assert np.isreal(tpl).all()


class TestFixedGroup:
    """Tests for template.fixed_group function."""

    @pytest.fixture
    def dihedral_group(self):
        """Create a DihedralGroup for testing."""
        from escnn.group import DihedralGroup

        return DihedralGroup(N=3)

    def test_output_shape(self, dihedral_group):
        """Test that output shape matches group order."""
        group_order = dihedral_group.order()
        num_irreps = len(list(dihedral_group.irreps()))
        fourier_coef_diag_values = [1.0] * num_irreps

        tpl = template.fixed_group(dihedral_group, fourier_coef_diag_values)

        assert tpl.shape == (group_order,), f"Expected shape ({group_order},), got {tpl.shape}"

    def test_mean_centered(self, dihedral_group):
        """Test that the template is mean-centered."""
        num_irreps = len(list(dihedral_group.irreps()))
        fourier_coef_diag_values = [1.0] * num_irreps

        tpl = template.fixed_group(dihedral_group, fourier_coef_diag_values)

        np.testing.assert_allclose(tpl.mean(), 0, atol=1e-10)

    def test_wrong_num_coefs_error(self, dihedral_group):
        """Test that mismatched number of coefficients raises error."""
        wrong_num_coefs = [1.0, 2.0]

        with pytest.raises(AssertionError):
            template.fixed_group(dihedral_group, wrong_num_coefs)

    def test_real_valued(self, dihedral_group):
        """Test that the template is real-valued."""
        num_irreps = len(list(dihedral_group.irreps()))
        fourier_coef_diag_values = [1.0] * num_irreps

        tpl = template.fixed_group(dihedral_group, fourier_coef_diag_values)

        assert np.isreal(tpl).all()
