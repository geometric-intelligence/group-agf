import torch


class PerNeuronScaledSGD(torch.optim.Optimizer):
    """SGD with per-neuron learning rate scaling:
        eta_i = ||theta_i||^(1 - k)
    where theta_i = (W_in[i,:], W_drive[i,:], W_out[:,i]).

    Args:
        model: the model to optimize
        lr: the learning rate
        k: the degree of the nonlinearity, for binary composition: square k=2.
    See: Appendix B.5 A neuron-specific adaptive learning rate
    yields instantaneous alignment of AGF paper
    """

    def __init__(self, model, lr=1e-2, k=2):
        params = [model.U, model.V, model.W]
        # Print shape of parameters with their names
        print(f"model.U shape: {model.U.shape}")
        print(f"model.V shape: {model.V.shape}")
        print(f"model.W shape: {model.W.shape}")
        super().__init__([{"params": params, "model": model}], dict(lr=lr, k=k))

    @torch.no_grad()
    def step(self, closure=None):
        group = self.param_groups[0]
        model = group["model"]
        lr = group["lr"]
        k = group["k"]
        U, V, W = model.U, model.V, model.W  # each of shape (hidden_size, group_size)
        g_U, g_V, g_W = (
            U.grad,
            V.grad,
            W.grad,
        )  # each of shape (hidden_size, group_size)
        if g_U is None or g_V is None or g_W is None:
            return
        # per-neuron norms
        # TODO(nina): check if this is correct.
        u2 = (U**2).sum(dim=1)  # shape: (hidden_size,): nb of hidden neurons.
        v2 = (V**2).sum(dim=1)  # shape: (hidden_size,): nb of hidden neurons.
        w2 = (W**2).sum(dim=1)  # shape: (hidden_size,): nb of hidden neurons.
        theta_norm = torch.sqrt(
            u2 + v2 + w2 + 1e-12
        )  # shape: (hidden_size,): nb of hidden neurons.
        # scale = ||theta_i||^(1 - k)
        scale = theta_norm.pow(1 - k)  # shape: (hidden_size,): nb of hidden neurons.
        # scale each neuron's grads
        g_U.mul_(scale.view(-1, 1))
        g_V.mul_(scale.view(-1, 1))
        g_W.mul_(scale.view(-1, 1))
        # SGD update
        U.add_(g_U, alpha=-lr)
        V.add_(g_V, alpha=-lr)
        W.add_(g_W, alpha=-lr)
