import torch

class PerNeuronScaledSGD(torch.optim.Optimizer):
    """SGD with per-neuron learning rate scaling:
        eta_i = ||theta_i||^(1 - k)
    where theta_i = (W_in[i,:], W_drive[i,:], W_out[:,i]).
    """
    def __init__(self, model, lr=1e-2, k=2):
        params = [model.U, model.V, model.W]
        # Print shape of parameters with their names
        print(f"model.U shape: {model.U.shape}")
        print(f"model.V shape: {model.V.shape}")
        print(f"model.W shape: {model.W.shape}")
        super().__init__([{'params': params, 'model': model}], dict(lr=lr, k=k))

    @torch.no_grad()
    def step(self, closure=None):
        group = self.param_groups[0]
        model = group['model']
        lr = group['lr']
        k = group['k']
        U, V, W = model.U, model.V, model.W
        print(f"U shape: {U.shape}")
        print(f"V shape: {V.shape}")
        print(f"W shape: {W.shape}")
        g_U, g_V, g_W = U.grad, V.grad, W.grad
        if g_U is None or g_V is None or g_W is None:
            return
        # per-neuron norms
        # TODO(nina): check if this is correct.
        u2 = (U**2).sum(dim=1)
        v2 = (V**2).sum(dim=1)
        w2 = (W**2).sum(dim=1)
        print(f"u2 shape: {u2.shape}")
        print(f"v2 shape: {v2.shape}")
        print(f"w2 shape: {w2.shape}")
        theta_norm = torch.sqrt(u2 + v2 + w2 + 1e-12)
        print(f"theta_norm shape: {theta_norm.shape}")
        # scale = ||theta_i||^(1 - k)
        scale = theta_norm.pow(1 - k)
        print(f"scale shape: {scale.shape}")
        # scale each neuron's grads
        g_U.mul_(scale.view(-1, 1))
        g_V.mul_(scale.view(-1, 1))
        g_W.mul_(scale.view(-1, 1))
        print(f"g_U shape: {g_U.shape}")
        print(f"g_V shape: {g_V.shape}")
        print(f"g_W shape: {g_W.shape}")
        # SGD update
        U.add_(g_U, alpha=-lr)
        V.add_(g_V, alpha=-lr)
        W.add_(g_W, alpha=-lr)