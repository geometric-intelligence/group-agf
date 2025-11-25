import torch

class PerNeuronScaledSGD(torch.optim.Optimizer):
    """
    Per-neuron scaled SGD optimizer that adapts to different model architectures.
    
    Learning rate scaling per neuron i:
        eta_i = lr * ||theta_i||^scaling_factor
    
    where theta_i comprises all parameters associated with neuron i.
    
    For SequentialMLP:
        theta_i = (W_in[i, :], W_out[:, i])
        scaling_factor = k (degree of homogeneity = activation power)
    
    For TwoLayerNet (k=2):
        theta_i = (U[i, :], V[i, :], W[:, i])
        scaling_factor = 2 (quadratic activation)
    
    The scaling exploits the homogeneity property: if we scale all parameters of
    neuron i by α, the output scales by α^k where k is the degree of homogeneity.
    """

    def __init__(self, 
        model, 
        lr=1.0, 
        scaling_factor=None
    ) -> None:
        """
        Args:
            model: SequentialMLP or compatible model
            lr: base learning rate
            scaling_factor: exponent for norm-based scaling. If None, inferred from model.
                           For SequentialMLP: defaults to model.k
        """
        # Infer scaling_factor from model if not provided
        if scaling_factor is None:
            if hasattr(model, 'k'):
                scaling_factor = model.k  # SequentialMLP
            else:
                scaling_factor = 2  # Default for quadratic models
        
        # Get model parameters
        params = list(model.parameters())
        
        super().__init__(
            [{'params': params, 'model': model, 'model_type': type(model).__name__}], 
            dict(lr=lr, scaling_factor=scaling_factor)
        )

    @torch.no_grad()
    def step(self, closure=None):
        group = self.param_groups[0]
        model = group['model']
        lr = group['lr']
        scaling_factor = group['scaling_factor']
        model_type = group['model_type']
        
        if model_type == 'SequentialMLP':
            # SequentialMLP: W_in (d, k*p), W_out (p, d)
            W_in = model.W_in
            W_out = model.W_out
            g_in = W_in.grad
            g_out = W_out.grad
            
            if g_in is None or g_out is None:
                return
            
            # Per-neuron norms: theta_i = (W_in[i, :], W_out[:, i])
            u2 = (W_in**2).sum(dim=1)  # (d,)
            w2 = (W_out**2).sum(dim=0)  # (d,)
            theta_norm = torch.sqrt(u2 + w2 + 1e-12)  # (d,)
            
            # Scale = ||theta_i||^scaling_factor
            scale = theta_norm.pow(scaling_factor)
            
            # Scale each neuron's gradients
            g_in.mul_(scale.view(-1, 1))
            g_out.mul_(scale.view(1, -1))
            
            # SGD update
            W_in.add_(g_in, alpha=-lr)
            W_out.add_(g_out, alpha=-lr)
            
        else:
            # Generic fallback for other models (e.g., TwoLayerNet with U, V, W)
            # Assume model has U, V, W or similar structure
            params_list = list(model.parameters())
            if len(params_list) == 3:
                # Assume structure like (U, V, W) or (W_in, W_drive, W_out)
                param1, param2, param3 = params_list
                g1, g2, g3 = param1.grad, param2.grad, param3.grad
                
                if g1 is None or g2 is None or g3 is None:
                    return
                
                # Per-neuron norms
                u2 = (param1**2).sum(dim=1)
                v2 = (param2**2).sum(dim=1)
                w2 = (param3**2).sum(dim=0)
                theta_norm = torch.sqrt(u2 + v2 + w2 + 1e-12)
                
                # Scale = ||theta_i||^scaling_factor
                scale = theta_norm.pow(scaling_factor)
                
                # Scale gradients
                g1.mul_(scale.view(-1, 1))
                g2.mul_(scale.view(-1, 1))
                g3.mul_(scale.view(1, -1))
                
                # SGD update
                param1.add_(g1, alpha=-lr)
                param2.add_(g2, alpha=-lr)
                param3.add_(g3, alpha=-lr)
            else:
                raise ValueError(f"PerNeuronScaledSGD: Unsupported model structure with {len(params_list)} parameters")



class HybridRNNOptimizer(torch.optim.Optimizer):
    """
    Hybrid optimizer for QuadraticRNN:
    - Per-neuron scaled SGD for W_in, W_drive, W_out (the MLP-like components)
    - Regular Adam for W_mix (the recurrent component)
    
    The per-neuron scaling is:
        eta_i = lr * ||theta_i||^scaling_factor
    where theta_i = (W_in[i,:], W_drive[i,:], W_out[:,i]).
    """

    def __init__(
        self, 
        model, 
        lr=1e-2, 
        scaling_factor=-1, 
        adam_lr=1e-3, 
        adam_betas=(0.9, 0.999), 
        adam_eps=1e-8
    ):
        """
        Args:
            model: QuadraticRNN model
            lr: learning rate for scaled SGD (W_in, W_drive, W_out)
            scaling_factor: scaling factor for the per-neuron learning rate
            adam_lr: learning rate for Adam (W_mix)
            adam_betas: Adam beta parameters
            adam_eps: Adam epsilon for numerical stability
        """
        # Create parameter groups
        scaled_params = [model.W_in, model.W_drive, model.W_out]
        adam_params = [model.W_mix]
        
        defaults = dict(
            model=model, 
            lr=lr, 
            scaling_factor=scaling_factor, 
            adam_lr=adam_lr,
            adam_betas=adam_betas,
            adam_eps=adam_eps
        )
        
        # Initialize with all params
        super().__init__(
            [
                {'params': scaled_params, 'type': 'scaled_sgd'},
                {'params': adam_params, 'type': 'adam'}
            ], 
            defaults
        )
        
        # Initialize Adam state for W_mix
        self.state['step'] = 0
        for param in adam_params:
            self.state[param] = {
                'exp_avg': torch.zeros_like(param),
                'exp_avg_sq': torch.zeros_like(param)
            }

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step."""
        
        for group in self.param_groups:
            if group['type'] == 'scaled_sgd':
                # Per-neuron scaled SGD for W_in, W_drive, W_out
                model = self.defaults['model']
                lr = self.defaults['lr']
                scaling_factor = self.defaults['scaling_factor']
                
                W_in, W_drive, W_out = model.W_in, model.W_drive, model.W_out
                g_in, g_drive, g_out = W_in.grad, W_drive.grad, W_out.grad
                
                if g_in is None or g_drive is None or g_out is None:
                    continue
                
                # Compute per-neuron norms
                u2 = (W_in**2).sum(dim=1)  # (d,)
                v2 = (W_drive**2).sum(dim=1)  # (d,)
                w2 = (W_out**2).sum(dim=0)  # (d,)
                theta_norm = torch.sqrt(u2 + v2 + w2 + 1e-12)  # (d,)
                
                # Scale = ||theta_i||^scaling_factor
                scale = theta_norm.pow(scaling_factor)
                
                # Scale each neuron's gradients
                g_in_scaled = g_in * scale.view(-1, 1)
                g_drive_scaled = g_drive * scale.view(-1, 1)
                g_out_scaled = g_out * scale.view(1, -1)
                
                # SGD update
                W_in.add_(g_in_scaled, alpha=-lr)
                W_drive.add_(g_drive_scaled, alpha=-lr)
                W_out.add_(g_out_scaled, alpha=-lr)
                
            elif group['type'] == 'adam':
                # Regular Adam for W_mix
                adam_lr = self.defaults['adam_lr']
                beta1, beta2 = self.defaults['adam_betas']
                eps = self.defaults['adam_eps']
                
                self.state['step'] += 1
                step = self.state['step']
                
                for param in group['params']:
                    if param.grad is None:
                        continue
                    
                    grad = param.grad
                    state = self.state[param]
                    
                    exp_avg = state['exp_avg']
                    exp_avg_sq = state['exp_avg_sq']
                    
                    # Update biased first and second moment estimates
                    exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                    exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                    
                    # Bias correction
                    bias_correction1 = 1 - beta1 ** step
                    bias_correction2 = 1 - beta2 ** step
                    
                    # Compute step size
                    step_size = adam_lr / bias_correction1
                    
                    # Compute bias-corrected second moment
                    denom = (exp_avg_sq.sqrt() / (bias_correction2 ** 0.5)).add_(eps)
                    
                    # Update parameters
                    param.addcdiv_(exp_avg, denom, value=-step_size)
        
        return None