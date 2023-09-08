import torch
import torch.optim as optim

def calculate_l2_norm(model):
    l2_norm = 0.0
    for param in model.parameters():
        l2_norm += torch.norm(param, 2).item() ** 2  # L2 norm
    return l2_norm

class AdamW_Perpendicular(optim.AdamW):
    def __init__(self, params, lr=1e-3, weight_decay=0, **kwargs):
        super(AdamW_Perpendicular, self).__init__(params, lr=lr, weight_decay=weight_decay, **kwargs)

    def step(self, closure=None):
        loss = super().step(closure)

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            lr = group['lr']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad.data
                # Compute the squared norm of the gradient
                grad_norm_sq = grad.view(-1).dot(grad.view(-1))

                # Compute the component of the weight that is parallel to the gradient
                parallel_component = (torch.dot(p.data.view(-1), grad.view(-1)) / (grad_norm_sq + 1e-10)) * grad

                # Add the parallel component back, essentially canceling the L2 weight decay on it
                p.data += lr *weight_decay * parallel_component/(1-lr *weight_decay)

        return loss
    
class Adam_Perpendicular(optim.Adam):
    def __init__(self, params, lr=1e-3, weight_decay=0, **kwargs):
        # Initialize the Adam optimizer with weight_decay=0
        super(Adam_Perpendicular, self).__init__(params, lr=lr, weight_decay=0, **kwargs)
        
        # Store the weight decay in another attribute for your modification
        for group in self.param_groups:
            group['weight_decay_perp'] = weight_decay

    def step(self, closure=None):
        loss = super().step(closure)

        for group in self.param_groups:
            weight_decay = group['weight_decay_perp']  # Get weight_decay from the new attribute
            lr = group['lr']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad.data
                # Compute the squared norm of the gradient
                grad_norm_sq = grad.view(-1).dot(grad.view(-1))

                # Compute the component of the weight that is parallel to the gradient
                parallel_component = (torch.dot(p.data.view(-1), grad.view(-1)) / (grad_norm_sq + 1e-10)) * grad

                # Add the parallel component back, essentially canceling the L2 weight decay on it
                p.data*=1 - lr * weight_decay
                p.data += lr * weight_decay * parallel_component 

        return loss
    
    
class AdamSqueeze(optim.Adam):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 threshold=1e-2  # Scaling factor for the soft-thresholding operation
                 ):
        super(AdamSqueeze, self).__init__(params, lr=lr, betas=betas, eps=eps, weight_decay=0)
        self.threshold = threshold

    def step(self, closure=None):
        loss = super().step(closure)

        # Perform custom operations
        for group in self.param_groups:
            lr = group['lr']  # This is where you get the learning rate
            for p in group['params']:
                if p.grad is None:
                    continue

                # Perform the soft-thresholding operation
                p.data = torch.sign(p.data) * torch.clamp(torch.abs(p.data) - self.threshold * lr, min=0)

        return loss

