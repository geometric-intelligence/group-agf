"""Tests for gagf.rnns.datamodule module."""

import numpy as np
import pytest

from src.datamodule import (
    OnlineModularAdditionDataset1D,
    OnlineModularAdditionDataset2D,
    build_modular_addition_sequence_dataset_1d,
    build_modular_addition_sequence_dataset_2d,
    build_modular_addition_sequence_dataset_D3,
)


class TestBuildModularAdditionSequenceDataset1D:
    """Tests for build_modular_addition_sequence_dataset_1d."""

    @pytest.fixture
    def template_1d(self):
        """Create a simple 1D template."""
        p = 7
        template = np.random.randn(p).astype(np.float32)
        return template

    def test_output_shape_sampled(self, template_1d):
        """Test output shapes in sampled mode."""
        p = len(template_1d)
        k = 3
        num_samples = 100

        X, Y, sequence = build_modular_addition_sequence_dataset_1d(
            p=p, template=template_1d, k=k, mode="sampled", num_samples=num_samples
        )

        assert X.shape == (num_samples, k, p), f"X shape mismatch: {X.shape}"
        assert Y.shape == (num_samples, p), f"Y shape mismatch: {Y.shape}"
        assert sequence.shape == (num_samples, k), f"sequence shape mismatch: {sequence.shape}"

    def test_output_shape_exhaustive(self, template_1d):
        """Test output shapes in exhaustive mode."""
        p = len(template_1d)
        k = 2

        X, Y, sequence = build_modular_addition_sequence_dataset_1d(
            p=p, template=template_1d, k=k, mode="exhaustive"
        )

        expected_n = p**k
        assert X.shape == (expected_n, k, p)
        assert Y.shape == (expected_n, p)
        assert sequence.shape == (expected_n, k)

    def test_output_shape_return_all_outputs(self, template_1d):
        """Test output shapes with return_all_outputs=True."""
        p = len(template_1d)
        k = 4
        num_samples = 50

        X, Y, sequence = build_modular_addition_sequence_dataset_1d(
            p=p,
            template=template_1d,
            k=k,
            mode="sampled",
            num_samples=num_samples,
            return_all_outputs=True,
        )

        # Y should have k-1 outputs (one after each pair of tokens)
        assert X.shape == (num_samples, k, p)
        assert Y.shape == (num_samples, k - 1, p)
        assert sequence.shape == (num_samples, k)

    def test_rolling_correctness(self, template_1d):
        """Test that X values are rolled versions of template."""
        p = len(template_1d)
        k = 2

        X, Y, sequence = build_modular_addition_sequence_dataset_1d(
            p=p, template=template_1d, k=k, mode="exhaustive"
        )

        # Check first sample
        shift_0 = int(sequence[0, 0])
        expected_x0 = np.roll(template_1d, shift_0)
        np.testing.assert_allclose(X[0, 0, :], expected_x0, rtol=1e-5)


class TestBuildModularAdditionSequenceDataset2D:
    """Tests for build_modular_addition_sequence_dataset_2d."""

    @pytest.fixture
    def template_2d(self):
        """Create a simple 2D template."""
        p1, p2 = 5, 5
        template = np.random.randn(p1, p2).astype(np.float32)
        return template

    def test_output_shape_sampled(self, template_2d):
        """Test output shapes in sampled mode."""
        p1, p2 = template_2d.shape
        k = 3
        num_samples = 100

        X, Y, sequence_xy = build_modular_addition_sequence_dataset_2d(
            p1=p1, p2=p2, template=template_2d, k=k, mode="sampled", num_samples=num_samples
        )

        p_flat = p1 * p2
        assert X.shape == (num_samples, k, p_flat), f"X shape mismatch: {X.shape}"
        assert Y.shape == (num_samples, p_flat), f"Y shape mismatch: {Y.shape}"
        assert sequence_xy.shape == (
            num_samples,
            k,
            2,
        ), f"sequence_xy shape mismatch: {sequence_xy.shape}"

    def test_output_shape_exhaustive(self, template_2d):
        """Test output shapes in exhaustive mode."""
        p1, p2 = 3, 3  # Use small dimensions for exhaustive
        template = np.random.randn(p1, p2).astype(np.float32)
        k = 2

        X, Y, sequence_xy = build_modular_addition_sequence_dataset_2d(
            p1=p1, p2=p2, template=template, k=k, mode="exhaustive"
        )

        expected_n = (p1 * p2) ** k
        p_flat = p1 * p2
        assert X.shape == (expected_n, k, p_flat)
        assert Y.shape == (expected_n, p_flat)
        assert sequence_xy.shape == (expected_n, k, 2)


class TestBuildModularAdditionSequenceDatasetD3:
    """Tests for build_modular_addition_sequence_dataset_D3."""

    @pytest.fixture
    def template_d3(self):
        """Create a template for D3 group (order 6)."""
        group_order = 6
        template = np.random.randn(group_order).astype(np.float32)
        return template

    def test_output_shape_sampled(self, template_d3):
        """Test output shapes in sampled mode."""
        k = 3
        num_samples = 100
        group_order = len(template_d3)

        X, Y, sequence = build_modular_addition_sequence_dataset_D3(
            template=template_d3, k=k, mode="sampled", num_samples=num_samples
        )

        assert X.shape == (num_samples, k, group_order), f"X shape mismatch: {X.shape}"
        assert Y.shape == (num_samples, group_order), f"Y shape mismatch: {Y.shape}"
        assert sequence.shape == (num_samples, k), f"sequence shape mismatch: {sequence.shape}"

    def test_output_shape_exhaustive(self, template_d3):
        """Test output shapes in exhaustive mode."""
        k = 2
        group_order = len(template_d3)
        n_elements = group_order  # D3 has 6 elements

        X, Y, sequence = build_modular_addition_sequence_dataset_D3(
            template=template_d3, k=k, mode="exhaustive"
        )

        expected_n = n_elements**k
        assert X.shape == (expected_n, k, group_order)
        assert Y.shape == (expected_n, group_order)
        assert sequence.shape == (expected_n, k)

    def test_output_shape_return_all_outputs(self, template_d3):
        """Test output shapes with return_all_outputs=True."""
        k = 4
        num_samples = 50
        group_order = len(template_d3)

        X, Y, sequence = build_modular_addition_sequence_dataset_D3(
            template=template_d3,
            k=k,
            mode="sampled",
            num_samples=num_samples,
            return_all_outputs=True,
        )

        assert X.shape == (num_samples, k, group_order)
        assert Y.shape == (num_samples, k - 1, group_order)
        assert sequence.shape == (num_samples, k)


class TestOnlineModularAdditionDataset1D:
    """Tests for OnlineModularAdditionDataset1D."""

    def test_batch_shape(self):
        """Test that batches have correct shapes."""
        p = 7
        k = 3
        batch_size = 16
        template = np.random.randn(p).astype(np.float32)

        dataset = OnlineModularAdditionDataset1D(
            p=p, template=template, k=k, batch_size=batch_size, device="cpu"
        )

        # Get first batch
        iterator = iter(dataset)
        X, Y = next(iterator)

        assert X.shape == (batch_size, k, p), f"X shape mismatch: {X.shape}"
        assert Y.shape == (batch_size, p), f"Y shape mismatch: {Y.shape}"

    def test_batch_shape_return_all_outputs(self):
        """Test batch shapes with return_all_outputs=True."""
        p = 7
        k = 4
        batch_size = 16
        template = np.random.randn(p).astype(np.float32)

        dataset = OnlineModularAdditionDataset1D(
            p=p,
            template=template,
            k=k,
            batch_size=batch_size,
            device="cpu",
            return_all_outputs=True,
        )

        iterator = iter(dataset)
        X, Y = next(iterator)

        assert X.shape == (batch_size, k, p)
        assert Y.shape == (batch_size, k - 1, p)


class TestOnlineModularAdditionDataset2D:
    """Tests for OnlineModularAdditionDataset2D."""

    def test_batch_shape(self):
        """Test that batches have correct shapes."""
        p1, p2 = 5, 5
        k = 3
        batch_size = 16
        template = np.random.randn(p1, p2).astype(np.float32)

        dataset = OnlineModularAdditionDataset2D(
            p1=p1, p2=p2, template=template, k=k, batch_size=batch_size, device="cpu"
        )

        iterator = iter(dataset)
        X, Y = next(iterator)

        p_flat = p1 * p2
        assert X.shape == (batch_size, k, p_flat), f"X shape mismatch: {X.shape}"
        assert Y.shape == (batch_size, p_flat), f"Y shape mismatch: {Y.shape}"

    def test_batch_shape_return_all_outputs(self):
        """Test batch shapes with return_all_outputs=True."""
        p1, p2 = 5, 5
        k = 4
        batch_size = 16
        template = np.random.randn(p1, p2).astype(np.float32)

        dataset = OnlineModularAdditionDataset2D(
            p1=p1,
            p2=p2,
            template=template,
            k=k,
            batch_size=batch_size,
            device="cpu",
            return_all_outputs=True,
        )

        iterator = iter(dataset)
        X, Y = next(iterator)

        p_flat = p1 * p2
        assert X.shape == (batch_size, k, p_flat)
        assert Y.shape == (batch_size, k - 1, p_flat)
