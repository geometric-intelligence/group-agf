"""Tests for group_agf.binary_action_learning.models module."""

import pytest
import torch
import numpy as np

from group_agf.binary_action_learning.models import TwoLayerNet


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
        
        assert y.shape == (batch_size, group_size), f"Expected shape {(batch_size, group_size)}, got {y.shape}"

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
        
        model_small = TwoLayerNet(**default_params, output_scale=scale_small)
        model_large = TwoLayerNet(**default_params, output_scale=scale_large)
        
        # Same random seed for reproducibility
        torch.manual_seed(42)
        x = torch.randn(4, 2 * default_params["group_size"])
        
        # Initialize both models with same weights
        torch.manual_seed(42)
        model_small = TwoLayerNet(**default_params, output_scale=scale_small)
        torch.manual_seed(42)
        model_large = TwoLayerNet(**default_params, output_scale=scale_large)
        
        y_small = model_small(x)
        y_large = model_large(x)
        
        # Output with larger scale should have larger absolute values on average
        assert y_large.abs().mean() > y_small.abs().mean()
