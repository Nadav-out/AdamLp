import torch
import torch.optim as optim
import numpy as np

def collect_all_weights(model):
    """
    Collect all weights from the PyTorch model and return as a NumPy array.

    Parameters:
        model (torch.nn.Module): PyTorch model

    Returns:
        np.array: All weights from all layers of the model in a flattened form.
    """
    all_weights = []

    # Loop over all named parameters and collect weights
    for name, param in model.named_parameters():
        # Only interested in weights (ignoring biases)
        if 'weight' in name:
            # Convert the weight tensor to a NumPy array and flatten it
            weights = param.detach().cpu().numpy().flatten()
            all_weights.extend(weights)

    # Convert the list to a NumPy array for better performance
    return np.array(all_weights)


def calculate_l2_norm(model):
    l2_norm = 0.0
    for param in model.parameters():
        l2_norm += torch.norm(param, 2).item() ** 2  # L2 norm
    return l2_norm

# class AdamW_Perpendicular(optim.AdamW):
#     def __init__(self, params, lr=1e-3, weight_decay=0, **kwargs):
#         super(AdamW_Perpendicular, self).__init__(params, lr=lr, weight_decay=weight_decay, **kwargs)

#     def step(self, closure=None):
#         loss = super().step(closure)

#         for group in self.param_groups:
#             weight_decay = group['weight_decay']
#             lr = group['lr']

#             for p in group['params']:
#                 if p.grad is None:
#                     continue

#                 grad = p.grad.data
#                 # Compute the squared norm of the gradient
#                 grad_norm_sq = grad.view(-1).dot(grad.view(-1))

#                 # Compute the component of the weight that is parallel to the gradient
#                 parallel_component = (torch.dot(p.data.view(-1), grad.view(-1)) / (grad_norm_sq + 1e-10)) * grad

#                 # Add the parallel component back, essentially canceling the L2 weight decay on it
#                 p.data += lr *weight_decay * parallel_component/(1-lr *weight_decay)

#         return loss
    
# class Adam_Perpendicular(optim.Adam):
#     def __init__(self, params, lr=1e-3, weight_decay=0, **kwargs):
#         # Initialize the Adam optimizer with weight_decay=0
#         super(Adam_Perpendicular, self).__init__(params, lr=lr, weight_decay=0, **kwargs)
        
#         # Store the weight decay in another attribute for your modification
#         for group in self.param_groups:
#             group['weight_decay_perp'] = weight_decay

#     def step(self, closure=None):
#         loss = super().step(closure)

#         for group in self.param_groups:
#             weight_decay = group['weight_decay_perp']  # Get weight_decay from the new attribute
#             lr = group['lr']

#             for p in group['params']:
#                 if p.grad is None:
#                     continue

#                 grad = p.grad.data
#                 # Compute the squared norm of the gradient
#                 grad_norm_sq = grad.view(-1).dot(grad.view(-1))

#                 # Compute the component of the weight that is parallel to the gradient
#                 parallel_component = (torch.dot(p.data.view(-1), grad.view(-1)) / (grad_norm_sq + 1e-10)) * grad

#                 # Add the parallel component back, essentially canceling the L2 weight decay on it
#                 p.data*=1 - lr * weight_decay
#                 p.data += lr * weight_decay * parallel_component 

#         return loss
    
    
# class AdamL1(optim.Adam):
#     def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, threshold=1e-2 ):
#         super(AdamL1, self).__init__(params, lr=lr, betas=betas, eps=eps, weight_decay=0)
#         self.threshold = threshold

#     def step(self, closure=None):
#         loss = super().step(closure)

#         # Perform custom operations
#         for group in self.param_groups:
#             lr = group['lr']  # This is where you get the learning rate
#             for p in group['params']:
#                 if p.grad is None:
#                     continue

#                 # Perform the soft-thresholding operation
#                 p.data = torch.sign(p.data) * torch.clamp(torch.abs(p.data) - self.threshold * lr, min=0)

#         return loss


# class AdamSqueeze(optim.Adam):
#     def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
#                  lambda_=1e-2,   # Scaling factor for the operation
#                  lp_order=1,     # Order for Lp regularization
#                  epsilon=1e-8,   # Numerical stability term
#                  s_initial=1e-2, # The the regularization term is initialized at lambda_/p*s_initial*|w|**p (p is 1 for 'soft' or 2 for 'scale')
#                  type='scale'    # 'scale' or 'soft'
#                  ):
#         super(AdamSqueeze, self).__init__(params, lr=lr, betas=betas, eps=eps, weight_decay=0)
#         self.lambda_ = lambda_
#         self.lp_order = lp_order
#         self.epsilon = epsilon
#         self.type = type
#         self.sigma = {}  
#         initial_sigma_value = torch.log(torch.tensor(s_initial))
#         for group in self.param_groups:
#             for param in group['params']:
#                 self.sigma[param] = torch.full_like(param, initial_sigma_value)

#         # Validate the conditions for 'soft' and 'scale'
#         if self.type == 'soft':
#             if lp_order >= 1 or lp_order < 0:
#                 raise ValueError("For 'soft', lp_order should be in the range 0 < lp_order < 1.")
#             self.order_in_F = 2 * lp_order
#             self.y_func = torch.abs
#             self.update_func = self.soft_threshold
#         elif self.type == 'scale':
#             if lp_order >= 2 or lp_order < 0:
#                 raise ValueError("For 'scale', lp_order should be in the range 0 < lp_order < 2.")
#             self.order_in_F = lp_order
#             self.y_func = lambda x: x ** 2
#             self.update_func = self.scale_param

#     def F_p(self, s, y):
#         # This function implaments the Newton's method update rule for sigma
#         order = self.order_in_F
        
#         # This regulates exp(sigma)<=1/epsilon
#         eps_term = 2*self.epsilon ** (2 / (2 - order)) 
#         y = y + eps_term  # Add epsilon term for numerical stability
#         numerator = s**(order/(order-2)) - s*y
#         denominator = order * s**(order/(order-2)) + (2-order) * s * y
#         return (order-2) / denominator * numerator

#     def soft_threshold(self, param, exp_sigma, lr):
#         return torch.sign(param) * torch.clamp(torch.abs(param) - self.lambda_ * lr * exp_sigma, min=0)

#     def scale_param(self, param, exp_sigma, lr):
#         return param / (1 + self.lambda_ * lr * exp_sigma)

#     def step(self, closure=None):
#         loss = super().step(closure)
        
#         # Perform custom operations
#         for group in self.param_groups:
#             lr = group['lr']  # Access the learning rate for this parameter group
#             for param in group['params']:
#                 if param.grad is None:
#                     continue
                
                
#                 # Compute exp(sigma)
#                 exp_sigma = torch.exp(self.sigma[param])
                
#                 # Update the parameter
#                 param.data = self.update_func(param.data, exp_sigma, lr)  # Pass lr to the update function

#                 # Update sigma using the F_p function
#                 y = self.y_func(param.data)
#                 self.sigma[param] -= 0.005*self.F_p(exp_sigma, y)
        
#         return loss
