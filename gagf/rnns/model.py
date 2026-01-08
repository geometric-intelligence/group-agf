from torch import nn
import torch


class QuadraticRNN(nn.Module):
    """
    h0 = W_init x1
    h_t = (W_mix h_{t-1} + W_drive x_t)^2   for t=1..k-1
    yhat = W_out h_{k-1}
    Note: This implementation uses k tokens total and applies the recurrence on tokens x1..x_k.
    """

    def __init__(
        self,
        p: int,
        d: int,
        template: torch.Tensor,  # put on device before passing to constructor
        init_scale: float = 1e-2,
        return_all_outputs: bool = False,
        transform_type: str = 'quadratic' # 'quadratic' | 'multiplicative'
    ) -> None:
        """
        Args:
            p: int, dimension of the template (height and width)
            d: int, dimension of the hidden state
            init_scale: float, scale of weights at initialization
            return_all_outputs: bool, whether to return all outputs at each step
            template: torch.Tensor, the template to use for the model
            transform_type: str, the type of transform to use ('quadratic' | 'multiplicative')
            if 'quadratic', uses (h @ W_mix.T + x @ W_drive.T)^2
            if 'multiplicative', uses h @ W_mix.T * x @ W_drive.T
        """
        super().__init__()
        self.p = p
        self.d = d
        self.init_scale = init_scale
        self.template = template
        self.return_all_outputs = return_all_outputs
        self.transform_type = transform_type

        # Params
        self.W_in = nn.Parameter(init_scale * torch.randn(d, p) / torch.sqrt(torch.tensor(p)))
        self.W_mix = nn.Parameter(init_scale * torch.randn(d, d) / torch.sqrt(torch.tensor(d)))
        self.W_drive = nn.Parameter(init_scale * torch.randn(d, p) / torch.sqrt(torch.tensor(p)))
        self.W_out = nn.Parameter(init_scale * torch.randn(p, d) / torch.sqrt(torch.tensor(d)))

    def _apply_transformation(
        self,
        h1: torch.Tensor,
        h2: torch.Tensor
    ) -> torch.Tensor:
        if self.transform_type == 'quadratic':
            return (h1 + h2)**2
        elif self.transform_type == 'multiplicative':
            return h1 * h2
        else:
            raise ValueError(f"Invalid transform type: {self.transform_type}")

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        """
        x_seq: (batch, k, p)
        returns: 
            if return_all_outputs=False: (batch, p) - only final output
            if return_all_outputs=True: (batch, k, p) - output at each step
        
        When k=2, this is equivalent to the 2-layer MLP.
        When k>2, the W_drive is used to drive the recurrence.
        """
        batch_size = x_seq.shape[0]
        k = x_seq.shape[1]
        assert k >= 2, "Sequence length must be at least 2"
        
        # Initialize
        h_0 = x_seq[:, 0, :] @ self.W_in.T  # (B, d)
        h_1 = x_seq[:, 1, :] @ self.W_drive.T  # (B, d)
        h = self._apply_transformation(h_0, h_1)
        
        if self.return_all_outputs:
            # Store outputs at each step
            outputs = []
            
            # Output after first two tokens
            y_1 = h @ self.W_out.T  # (B, p)
            outputs.append(y_1)
            
            # Recurrence
            for t in range(2, k):
                xt = x_seq[:, t, :]  # (B, p)
                h = self._apply_transformation(h @ self.W_mix.T, xt @ self.W_drive.T)  # (B, d)
                
                # Output after this token
                y_t = h @ self.W_out.T  # (B, p)
                outputs.append(y_t)
            
            # Stack all outputs: (B, k-1, p)
            # Note: k-1 because we need at least 2 tokens before first prediction
            return torch.stack(outputs, dim=1)
        else:
            # Original behavior: only return final output
            # Recurrence
            for t in range(2, k):
                xt = x_seq[:, t, :]  # (B, p)
                h = self._apply_transformation(h @ self.W_mix.T, xt @ self.W_drive.T)  # (B, d)
            
            y = h @ self.W_out.T  # (B, p)
            return y

class SequentialMLP(nn.Module):
    """
    MLP that processes k unordered input vectors by concatenating them.
    
    Architecture:
        x_concat = concat(x1, x2, ..., xk)  # shape: (batch, k*p)
        h = x_concat @ W_in.T               # shape: (batch, hidden_size)
        h_activated = h^k                    # k-th power activation
        y = h_activated @ W_out.T           # shape: (batch, p)
    
    Note: The inputs are unordered (not sequential), so the model is permutation-invariant
    with respect to the ordering of the k inputs (ok for commutative groups!).
    """

    def __init__(
        self,
        p: int,
        d: int,
        template: torch.Tensor,
        k: int,
        init_scale: float = 1e-2,
        return_all_outputs: bool = False,
    ) -> None:
        """
        Args:
            p: int, dimension of each input vector
            d: int, hidden dimension (hidden_size)
            template: torch.Tensor, the template (for compatibility with infrastructure)
            k: int, number of input vectors (sequence length)
            init_scale: float, scale of weights at initialization
            return_all_outputs: bool, for compatibility (always False for MLP)
        """
        super().__init__()
        self.p = p
        self.d = d  # Hidden dimension
        self.k = k  # Sequence length
        self.init_scale = init_scale
        self.template = template
        self.return_all_outputs = return_all_outputs
        
        # Parameters
        # W_in: maps concatenated input (k*p) to hidden (d)
        # W_out: maps hidden (d) to output (p)
        self.W_in = nn.Parameter(init_scale * torch.randn(d, k * p) / torch.sqrt(torch.tensor(k * p)))
        self.W_out = nn.Parameter(init_scale * torch.randn(p, d) / torch.sqrt(torch.tensor(d)))

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        """
        Process k unordered input vectors.
        
        Args:
            x_seq: (batch, k, p) - k input vectors of dimension p each
            
        Returns:
            (batch, p) - output vector
            Note: return_all_outputs is ignored for MLP (no intermediate outputs)
        """
        batch_size = x_seq.shape[0]
        k_actual = x_seq.shape[1]
        
        assert k_actual == self.k, f"Expected k={self.k} inputs, got {k_actual}"
        
        # Concatenate all k inputs
        x_concat = x_seq.reshape(batch_size, self.k * self.p)  # (batch, k*p)
        
        # First layer: linear transformation
        h = x_concat @ self.W_in.T  # (batch, d)
        
        # Activation: k-th power
        h_activated = h ** self.k  # (batch, d)
        
        # Second layer: output
        y = h_activated @ self.W_out.T  # (batch, p)
        
        return y