"""Tests for src.model module (QuadraticRNN, SequentialMLP, TwoLayerNet)."""

import pytest
import torch

from src.model import QuadraticRNN, SequentialMLP, TwoLayerNet


class TestQuadraticRNN:
    """Tests for the QuadraticRNN model."""

    @pytest.fixture
    def default_params(self):
        """Default parameters for QuadraticRNN."""
        p = 7
        d = 10
        template = torch.randn(p)
        return {"p": p, "d": d, "template": template}

    def test_output_shape_basic(self, default_params):
        """Test that output shape is correct for basic forward pass."""
        model = QuadraticRNN(**default_params)
        batch_size = 8
        k = 4
        p = default_params["p"]

        x = torch.randn(batch_size, k, p)
        y = model(x)

        assert y.shape == (batch_size, p), f"Expected shape {(batch_size, p)}, got {y.shape}"

    def test_output_shape_return_all_outputs(self, default_params):
        """Test output shape when return_all_outputs=True."""
        params = {**default_params, "return_all_outputs": True}
        model = QuadraticRNN(**params)
        batch_size = 8
        k = 5
        p = default_params["p"]

        x = torch.randn(batch_size, k, p)
        y = model(x)

        # With return_all_outputs=True, we get k-1 outputs (after first two tokens)
        expected_shape = (batch_size, k - 1, p)
        assert y.shape == expected_shape, f"Expected shape {expected_shape}, got {y.shape}"

    def test_output_shape_k_equals_2(self, default_params):
        """Test output shape when k=2 (minimum sequence length)."""
        model = QuadraticRNN(**default_params)
        batch_size = 4
        k = 2
        p = default_params["p"]

        x = torch.randn(batch_size, k, p)
        y = model(x)

        assert y.shape == (batch_size, p)

    def test_quadratic_transform(self, default_params):
        """Test that quadratic transform is applied correctly."""
        params = {**default_params, "transform_type": "quadratic"}
        model = QuadraticRNN(**params)

        batch_size = 2
        k = 3
        p = default_params["p"]

        x = torch.randn(batch_size, k, p)
        y = model(x)

        # Output should be finite
        assert torch.isfinite(y).all(), "Output contains non-finite values"

    def test_multiplicative_transform(self, default_params):
        """Test that multiplicative transform is applied correctly."""
        params = {**default_params, "transform_type": "multiplicative"}
        model = QuadraticRNN(**params)

        batch_size = 2
        k = 3
        p = default_params["p"]

        x = torch.randn(batch_size, k, p)
        y = model(x)

        # Output should be finite
        assert torch.isfinite(y).all(), "Output contains non-finite values"

    def test_invalid_transform_type(self, default_params):
        """Test that invalid transform type raises an error."""
        params = {**default_params, "transform_type": "invalid"}
        model = QuadraticRNN(**params)

        x = torch.randn(2, 3, default_params["p"])

        with pytest.raises(ValueError, match="Invalid transform type"):
            model(x)

    def test_minimum_sequence_length_error(self, default_params):
        """Test that k<2 raises an assertion error."""
        model = QuadraticRNN(**default_params)

        x = torch.randn(2, 1, default_params["p"])  # k=1

        with pytest.raises(AssertionError, match="Sequence length must be at least 2"):
            model(x)

    def test_gradient_flow(self, default_params):
        """Test that gradients flow through the model."""
        model = QuadraticRNN(**default_params)

        x = torch.randn(4, 3, default_params["p"], requires_grad=True)
        y = model(x)
        loss = y.sum()
        loss.backward()

        # Check that gradients exist for all parameters
        for name, param in model.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"
            assert torch.isfinite(param.grad).all(), f"Non-finite gradient for {name}"


class TestSequentialMLP:
    """Tests for the SequentialMLP model."""

    @pytest.fixture
    def default_params(self):
        """Default parameters for SequentialMLP."""
        p = 7
        d = 10
        k = 3
        template = torch.randn(p)
        return {"p": p, "d": d, "k": k, "template": template}

    def test_output_shape(self, default_params):
        """Test that output shape is correct."""
        model = SequentialMLP(**default_params)
        batch_size = 8
        k = default_params["k"]
        p = default_params["p"]

        x = torch.randn(batch_size, k, p)
        y = model(x)

        assert y.shape == (batch_size, p), f"Expected shape {(batch_size, p)}, got {y.shape}"

    def test_k_mismatch_error(self, default_params):
        """Test that mismatched k raises an error."""
        model = SequentialMLP(**default_params)

        wrong_k = default_params["k"] + 1
        x = torch.randn(2, wrong_k, default_params["p"])

        with pytest.raises(AssertionError, match="Expected k="):
            model(x)

    def test_different_k_values(self):
        """Test model with different k values."""
        p = 5
        d = 8
        template = torch.randn(p)

        for k in [2, 3, 4, 5]:
            model = SequentialMLP(p=p, d=d, k=k, template=template)
            x = torch.randn(4, k, p)
            y = model(x)

            assert y.shape == (4, p), f"Failed for k={k}"

    def test_gradient_flow(self, default_params):
        """Test that gradients flow through the model."""
        model = SequentialMLP(**default_params)

        x = torch.randn(4, default_params["k"], default_params["p"], requires_grad=True)
        y = model(x)
        loss = y.sum()
        loss.backward()

        # Check that gradients exist for all parameters
        for name, param in model.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"
            assert torch.isfinite(param.grad).all(), f"Non-finite gradient for {name}"

    def test_k_power_activation(self, default_params):
        """Test that k-th power activation produces finite results."""
        model = SequentialMLP(**default_params)

        # Use small inputs to avoid overflow with k-th power
        x = torch.randn(4, default_params["k"], default_params["p"]) * 0.1
        y = model(x)

        assert torch.isfinite(y).all(), "Output contains non-finite values"


class TestTwoLayerNet:
    """Tests for the TwoLayerNet model."""

    @pytest.fixture
    def default_params(self):
        """Default parameters for TwoLayerNet."""
        return {"group_size": 6, "hidden_size": 20}

    def test_output_shape(self, default_params):
        """Test that output shape is correct."""
        model = TwoLayerNet(**default_params)
        batch_size = 8
        group_size = default_params["group_size"]

        # Input is flattened: (batch, 2 * group_size)
        x = torch.randn(batch_size, 2 * group_size)
        y = model(x)

        assert y.shape == (
            batch_size,
            group_size,
        ), f"Expected shape {(batch_size, group_size)}, got {y.shape}"

    def test_square_nonlinearity(self, default_params):
        """Test that square nonlinearity produces finite results."""
        params = {**default_params, "nonlinearity": "square"}
        model = TwoLayerNet(**params)

        x = torch.randn(4, 2 * default_params["group_size"])
        y = model(x)

        assert torch.isfinite(y).all(), "Output contains non-finite values"

    def test_relu_nonlinearity(self, default_params):
        """Test that relu nonlinearity produces finite results."""
        params = {**default_params, "nonlinearity": "relu"}
        model = TwoLayerNet(**params)

        x = torch.randn(4, 2 * default_params["group_size"])
        y = model(x)

        assert torch.isfinite(y).all(), "Output contains non-finite values"

    def test_tanh_nonlinearity(self, default_params):
        """Test that tanh nonlinearity produces finite results."""
        params = {**default_params, "nonlinearity": "tanh"}
        model = TwoLayerNet(**params)

        x = torch.randn(4, 2 * default_params["group_size"])
        y = model(x)

        assert torch.isfinite(y).all(), "Output contains non-finite values"

    def test_gelu_nonlinearity(self, default_params):
        """Test that gelu nonlinearity produces finite results."""
        params = {**default_params, "nonlinearity": "gelu"}
        model = TwoLayerNet(**params)

        x = torch.randn(4, 2 * default_params["group_size"])
        y = model(x)

        assert torch.isfinite(y).all(), "Output contains non-finite values"

    def test_linear_nonlinearity(self, default_params):
        """Test that linear (no activation) produces finite results."""
        params = {**default_params, "nonlinearity": "linear"}
        model = TwoLayerNet(**params)

        x = torch.randn(4, 2 * default_params["group_size"])
        y = model(x)

        assert torch.isfinite(y).all(), "Output contains non-finite values"

    def test_invalid_nonlinearity(self, default_params):
        """Test that invalid nonlinearity raises an error."""
        params = {**default_params, "nonlinearity": "invalid"}
        model = TwoLayerNet(**params)

        x = torch.randn(4, 2 * default_params["group_size"])

        with pytest.raises(ValueError, match="Invalid nonlinearity"):
            model(x)

    def test_gradient_flow(self, default_params):
        """Test that gradients flow through the model."""
        model = TwoLayerNet(**default_params)

        x = torch.randn(4, 2 * default_params["group_size"], requires_grad=True)
        y = model(x)
        loss = y.sum()
        loss.backward()

        # Check that gradients exist for all parameters
        for name, param in model.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"
            assert torch.isfinite(param.grad).all(), f"Non-finite gradient for {name}"

    def test_default_hidden_size(self):
        """Test that default hidden_size is computed correctly."""
        group_size = 8
        model = TwoLayerNet(group_size=group_size)

        # Default hidden_size should be 50 * group_size
        assert model.hidden_size == 50 * group_size

    def test_output_scale(self, default_params):
        """Test that output_scale affects the output magnitude."""
        scale_small = 0.1
        scale_large = 10.0

        # Same random seed for reproducibility
        torch.manual_seed(42)
        model_small = TwoLayerNet(**default_params, output_scale=scale_small)
        torch.manual_seed(42)
        model_large = TwoLayerNet(**default_params, output_scale=scale_large)

        x = torch.randn(4, 2 * default_params["group_size"])

        y_small = model_small(x)
        y_large = model_large(x)

        # Output with larger scale should have larger absolute values on average
        assert y_large.abs().mean() > y_small.abs().mean()
