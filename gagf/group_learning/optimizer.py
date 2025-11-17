import torch

class PerNeuronScaledSGD(torch.optim.Optimizer):
    """SGD with per-neuron learning rate scaling:
        eta_i = ||theta_i||^(1 - k)
    where theta_i = (W_in[i,:], W_drive[i,:], W_out[:,i]).
    """
    def __init__(self, model, lr=1e-2, k=2):
        params = [model.W_in, model.W_drive, model.W_out]
        super().__init__([{'params': params, 'model': model}], dict(lr=lr, k=k))

    @torch.no_grad()
    def step(self, closure=None):
        group = self.param_groups[0]
        model = group['model']
        lr = group['lr']
        k = group['k']
        W_in, W_drive, W_out = model.W_in, model.W_drive, model.W_out
        g_in, g_drive, g_out = W_in.grad, W_drive.grad, W_out.grad
        if g_in is None or g_drive is None or g_out is None:
            return
        # per-neuron norms
        u2 = (W_in**2).sum(dim=1)
        v2 = (W_drive**2).sum(dim=1)
        w2 = (W_out**2).sum(dim=0)
        theta_norm = torch.sqrt(u2 + v2 + w2 + 1e-12)
        # scale = ||theta_i||^(1 - k)
        scale = theta_norm.pow(1 - k)
        # scale each neuron's grads
        g_in.mul_(scale.view(-1, 1))
        g_drive.mul_(scale.view(-1, 1))
        g_out.mul_(scale.view(1, -1))
        # SGD update
        W_in.add_(g_in, alpha=-lr)
        W_drive.add_(g_drive, alpha=-lr)
        W_out.add_(g_out, alpha=-lr)