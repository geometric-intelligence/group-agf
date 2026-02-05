"""Tests for group_agf.binary_action_learning.datasets module."""

import pytest
import numpy as np
import torch

from group_agf.binary_action_learning.datasets import (
    cn_dataset,
    cnxcn_dataset,
    group_dataset,
    move_dataset_to_device_and_flatten,
)


class TestCnDataset:
    """Tests for cn_dataset function."""

    def test_output_shape(self):
        """Test that output shapes are correct."""
        group_size = 7
        template = np.random.randn(group_size)
        
        X, Y = cn_dataset(template)
        
        n_samples = group_size ** 2
        assert X.shape == (n_samples, 2, group_size), f"X shape mismatch: {X.shape}"
        assert Y.shape == (n_samples, group_size), f"Y shape mismatch: {Y.shape}"

    def test_modular_addition_property(self):
        """Test that Y is the rolled template by (a+b) mod p."""
        group_size = 5
        template = np.arange(group_size).astype(float)  # [0, 1, 2, 3, 4]
        
        X, Y = cn_dataset(template)
        
        # Check a specific case: a=1, b=2 -> q=(1+2)%5=3
        # Index = a * group_size + b = 1 * 5 + 2 = 7
        idx = 1 * group_size + 2
        expected_y = np.roll(template, 3)  # rolled by 3
        np.testing.assert_allclose(Y[idx], expected_y)

    def test_covers_all_pairs(self):
        """Test that all pairs (a, b) are covered."""
        group_size = 4
        template = np.random.randn(group_size)
        
        X, Y = cn_dataset(template)
        
        # Should have exactly group_size^2 samples
        assert X.shape[0] == group_size ** 2


class TestCnxcnDataset:
    """Tests for cnxcn_dataset function."""

    def test_output_shape(self):
        """Test that output shapes are correct."""
        image_length = 4
        template = np.random.randn(image_length * image_length)
        
        X, Y = cnxcn_dataset(template)
        
        n_samples = image_length ** 4
        n_features = image_length * image_length
        assert X.shape == (n_samples, 2, n_features), f"X shape mismatch: {X.shape}"
        assert Y.shape == (n_samples, n_features), f"Y shape mismatch: {Y.shape}"

    def test_covers_all_combinations(self):
        """Test that all combinations are covered."""
        image_length = 3
        template = np.random.randn(image_length * image_length)
        
        X, Y = cnxcn_dataset(template)
        
        expected_n = image_length ** 4
        assert X.shape[0] == expected_n


class TestGroupDataset:
    """Tests for group_dataset function."""

    @pytest.fixture
    def dihedral_group(self):
        """Create a DihedralGroup for testing."""
        from escnn.group import DihedralGroup
        return DihedralGroup(N=3)  # D3

    def test_output_shape(self, dihedral_group):
        """Test that output shapes are correct for D3."""
        group_order = dihedral_group.order()  # 6 for D3
        template = np.random.randn(group_order)
        
        X, Y = group_dataset(dihedral_group, template)
        
        n_samples = group_order ** 2
        assert X.shape == (n_samples, 2, group_order), f"X shape mismatch: {X.shape}"
        assert Y.shape == (n_samples, group_order), f"Y shape mismatch: {Y.shape}"

    def test_template_length_mismatch_error(self, dihedral_group):
        """Test that mismatched template length raises error."""
        wrong_size = dihedral_group.order() + 1
        template = np.random.randn(wrong_size)
        
        with pytest.raises(AssertionError):
            group_dataset(dihedral_group, template)


class TestMoveDatasetToDeviceAndFlatten:
    """Tests for move_dataset_to_device_and_flatten function."""

    def test_output_shape_and_type(self):
        """Test that output shapes and types are correct."""
        group_size = 5
        n_samples = 10
        
        X = np.random.randn(n_samples, 2, group_size)
        Y = np.random.randn(n_samples, group_size)
        
        X_tensor, Y_tensor, device = move_dataset_to_device_and_flatten(X, Y, device="cpu")
        
        assert isinstance(X_tensor, torch.Tensor)
        assert isinstance(Y_tensor, torch.Tensor)
        assert X_tensor.shape == (n_samples, 2 * group_size)
        assert Y_tensor.shape == (n_samples, group_size)

    def test_flattening(self):
        """Test that X is correctly flattened."""
        group_size = 4
        n_samples = 5
        
        X = np.arange(n_samples * 2 * group_size).reshape(n_samples, 2, group_size).astype(float)
        Y = np.random.randn(n_samples, group_size)
        
        X_tensor, Y_tensor, device = move_dataset_to_device_and_flatten(X, Y, device="cpu")
        
        # Check first sample
        expected_flat = np.concatenate([X[0, 0, :], X[0, 1, :]])
        np.testing.assert_allclose(X_tensor[0].numpy(), expected_flat)

    def test_device_cpu(self):
        """Test explicit CPU device."""
        X = np.random.randn(5, 2, 4)
        Y = np.random.randn(5, 4)
        
        X_tensor, Y_tensor, device = move_dataset_to_device_and_flatten(X, Y, device="cpu")
        
        assert X_tensor.device.type == "cpu"
        assert Y_tensor.device.type == "cpu"
