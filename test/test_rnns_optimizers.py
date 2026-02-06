"""Tests for gagf.rnns.optimizers module."""

import pytest
import torch

from gagf.rnns.model import QuadraticRNN, SequentialMLP
from gagf.rnns.optimizers import HybridRNNOptimizer, PerNeuronScaledSGD


class TestPerNeuronScaledSGD:
    """Tests for PerNeuronScaledSGD optimizer."""

    @pytest.fixture
    def sequential_mlp(self):
        """Create a SequentialMLP model."""
        p = 5
        d = 10
        k = 3
        template = torch.randn(p)
        return SequentialMLP(p=p, d=d, k=k, template=template)

    def test_step_updates_parameters(self, sequential_mlp):
        """Test that optimizer step updates model parameters."""
        optimizer = PerNeuronScaledSGD(sequential_mlp, lr=0.01)

        # Store initial parameters
        initial_w_in = sequential_mlp.W_in.clone()
        initial_w_out = sequential_mlp.W_out.clone()

        # Forward pass and backward
        x = torch.randn(4, sequential_mlp.k, sequential_mlp.p)
        y = sequential_mlp(x)
        loss = y.sum()
        loss.backward()

        # Optimizer step
        optimizer.step()

        # Parameters should have changed
        assert not torch.allclose(sequential_mlp.W_in, initial_w_in), "W_in not updated"
        assert not torch.allclose(sequential_mlp.W_out, initial_w_out), "W_out not updated"

    def test_degree_inference(self, sequential_mlp):
        """Test that degree is correctly inferred from model."""
        optimizer = PerNeuronScaledSGD(sequential_mlp, lr=0.01)

        # Degree should be k + 1 for SequentialMLP
        expected_degree = sequential_mlp.k + 1
        assert optimizer.defaults["degree"] == expected_degree

    def test_explicit_degree(self, sequential_mlp):
        """Test that explicit degree overrides inference."""
        explicit_degree = 5
        optimizer = PerNeuronScaledSGD(sequential_mlp, lr=0.01, degree=explicit_degree)

        assert optimizer.defaults["degree"] == explicit_degree

    def test_finite_gradients_after_step(self, sequential_mlp):
        """Test that gradients remain finite after optimization step."""
        optimizer = PerNeuronScaledSGD(sequential_mlp, lr=0.01)

        x = torch.randn(4, sequential_mlp.k, sequential_mlp.p)
        y = sequential_mlp(x)
        loss = y.sum()
        loss.backward()

        optimizer.step()

        # All parameters should still be finite
        for name, param in sequential_mlp.named_parameters():
            assert torch.isfinite(param).all(), f"Non-finite values in {name}"


class TestHybridRNNOptimizer:
    """Tests for HybridRNNOptimizer."""

    @pytest.fixture
    def quadratic_rnn(self):
        """Create a QuadraticRNN model."""
        p = 5
        d = 10
        template = torch.randn(p)
        return QuadraticRNN(p=p, d=d, template=template)

    def test_step_updates_all_parameters(self, quadratic_rnn):
        """Test that optimizer step updates all model parameters."""
        optimizer = HybridRNNOptimizer(quadratic_rnn, lr=0.01, adam_lr=0.001)

        # Store initial parameters
        initial_params = {name: param.clone() for name, param in quadratic_rnn.named_parameters()}

        # Forward pass and backward
        x = torch.randn(4, 3, quadratic_rnn.p)
        y = quadratic_rnn(x)
        loss = y.sum()
        loss.backward()

        # Optimizer step
        optimizer.step()

        # All parameters should have changed
        for name, param in quadratic_rnn.named_parameters():
            assert not torch.allclose(param, initial_params[name]), f"{name} not updated"

    def test_scaled_sgd_for_mlp_params(self, quadratic_rnn):
        """Test that W_in, W_drive, W_out use scaled SGD."""
        optimizer = HybridRNNOptimizer(quadratic_rnn, lr=0.01)

        # The optimizer should have two param groups
        assert len(optimizer.param_groups) == 2

        # First group should be scaled_sgd
        assert optimizer.param_groups[0]["type"] == "scaled_sgd"
        # Second group should be adam
        assert optimizer.param_groups[1]["type"] == "adam"

    def test_adam_for_w_mix(self, quadratic_rnn):
        """Test that W_mix uses Adam optimizer."""
        optimizer = HybridRNNOptimizer(quadratic_rnn, lr=0.01, adam_lr=0.001)

        # W_mix should be in the adam group
        adam_params = list(optimizer.param_groups[1]["params"])
        assert len(adam_params) == 1
        assert adam_params[0] is quadratic_rnn.W_mix

    def test_finite_parameters_after_step(self, quadratic_rnn):
        """Test that parameters remain finite after optimization."""
        optimizer = HybridRNNOptimizer(quadratic_rnn, lr=0.01, adam_lr=0.001)

        x = torch.randn(4, 3, quadratic_rnn.p)
        y = quadratic_rnn(x)
        loss = y.sum()
        loss.backward()

        optimizer.step()

        # All parameters should be finite
        for name, param in quadratic_rnn.named_parameters():
            assert torch.isfinite(param).all(), f"Non-finite values in {name}"

    def test_multiple_steps(self, quadratic_rnn):
        """Test that multiple optimization steps work correctly."""
        optimizer = HybridRNNOptimizer(quadratic_rnn, lr=0.01, adam_lr=0.001)

        for _ in range(5):
            optimizer.zero_grad()
            x = torch.randn(4, 3, quadratic_rnn.p)
            y = quadratic_rnn(x)
            loss = y.sum()
            loss.backward()
            optimizer.step()

        # All parameters should still be finite after multiple steps
        for name, param in quadratic_rnn.named_parameters():
            assert torch.isfinite(param).all(), f"Non-finite values in {name}"
