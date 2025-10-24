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
    ) -> None:
        """
        Args:
            p: int, dimension of the template (height and width)
            d: int, dimension of the hidden state
            init_scale: float, scale of weights at initialization
            template
        """
        super().__init__()
        self.p = p
        self.d = d
        self.init_scale = init_scale
        self.template = template

        # Params
        self.W_in = nn.Parameter(init_scale * torch.randn(d, p) / torch.sqrt(torch.tensor(p)))
        self.W_mix = nn.Parameter(init_scale * torch.randn(d, d) / torch.sqrt(torch.tensor(d)))
        self.W_drive = nn.Parameter(init_scale * torch.randn(d, p) / torch.sqrt(torch.tensor(p)))
        self.W_out = nn.Parameter(init_scale * torch.randn(p, d) / torch.sqrt(torch.tensor(d)))

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        """
        x_seq: (batch, k, p)
        returns: (batch, p)
        """
        # x = self.template

        h = x_seq[:,0,:] @ self.W_in.T  # (B, d)

        # Recur for all tokens
        k = x_seq.shape[1]
        for t in range(1, k):
            xt = x_seq[:, t, :]  # (B, p)
            pre = (h @ self.W_mix.T) + (xt @ self.W_drive.T)  # (B, d)
            h = pre**2

        y = h @ self.W_out.T  # (B, p)
        return y