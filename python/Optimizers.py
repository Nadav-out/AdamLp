import torch
import torch.optim as optim


class Adam_Perpendicular(torch.optim.AdamW):
    def __init__(self, *args, **kwargs):
        super(Adam_Perpendicular, self).__init__(*args, **kwargs)

    def step(self, closure=None):
        for group in self.param_groups:
            weight_decay = group['weight_decay']
            lr = group['lr']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad.data
                grad_norm_sq = grad.view(-1).dot(grad.view(-1))
                parallel_component = (torch.dot(p.data.view(-1), grad.view(-1)) / (grad_norm_sq + 1e-10)) * grad

                # Adjust the weight for the parallel component
                p.data += lr * weight_decay * parallel_component / (1 - lr * weight_decay)

        # Finally, let the super().step() apply the AdamW updates (including its weight decay)
        loss = super().step(closure)

        return loss


class AdamSqueeze(optim.Adam):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 lambda_=1e-2,  # Scaling factor for the operation
                 lp_order=1,    # Order for Lp regularization
                 epsilon=1e-8,  # Numerical stability term
                 s_initial=1,   # The the regularization term is initialized at lambda_/p*s_initial*|w|**p (p is 1 for 'soft' or 2 for 'scale')
                 s_rate=0.005,  # The rate at which sigma is updated
                 type='scale'   # 'scale' or 'soft'
                 ):
        super(AdamSqueeze, self).__init__(params, lr=lr, betas=betas, eps=eps, weight_decay=0)
        self.lambda_ = lambda_
        self.lp_order = lp_order
        self.epsilon = epsilon
        self.type = type
        self.s_rate = s_rate

        # Initialize sigma
        # s=exp(sigma) is our new learnable parameter, which pushes L_2/L_1 to L_p.
        self.sigma = {}  
        initial_sigma_value = torch.log(torch.tensor(s_initial))
        for group in self.param_groups:
            for param in group['params']:
                self.sigma[param] = torch.full_like(param, initial_sigma_value)

        # Validate the conditions for 'soft' and 'scale'
        if self.type == 'soft':
            # Soft thresholding, we start at L_1 and flow twards L_p.
            # Only valid for p<1
            if lp_order >= 1 or lp_order < 0:
                raise ValueError("For 'soft', lp_order should be in the range 0 < lp_order < 1.")
            self.order_in_F = 2 * lp_order
            self.y_func = torch.abs
            self.update_func = self.soft_threshold
        elif self.type == 'scale':
            # Weight decay, simlar to AdamW. We start at L_2 and flow twards L_p.
            # Only valid for p<2
            if lp_order >= 2 or lp_order < 0:
                raise ValueError("For 'scale', lp_order should be in the range 0 < lp_order < 2.")
            self.order_in_F = lp_order
            self.y_func = lambda x: x ** 2
            self.update_func = self.scale_param

    def F_p(self, s, y):
        # This function implaments the Newton's method update rule for sigma
        # y=|param| for 'soft' and y=parm**2 for 'scale'.
        order = self.order_in_F
        
        # Scale epsilon term to regulates exp(sigma)=s<=1/epsilon
        eps_term = 2*self.epsilon ** (2 / (2 - order)) 
        y = y + eps_term  # Add epsilon term for numerical stability
        
        numerator = s**(order/(order-2)) - s*y
        denominator = order * s**(order/(order-2)) + (2-order) * s * y
        return (order-2) / denominator * numerator

    def soft_threshold(self, param, exp_sigma, lr):
        return torch.sign(param) * torch.relu(torch.abs(param) - self.lambda_ * self.order_in_F * lr * exp_sigma)

    def scale_param(self, param, exp_sigma, lr):
        return param / (1 + self.lambda_ * self.order_in_F * lr * exp_sigma)

    def step(self, closure=None):
        loss = super().step(closure)
        
        # Perform custom operations
        for group in self.param_groups:
            lr = group['lr']  # Access the learning rate for this parameter group
            for param in group['params']:
                if param.grad is None:
                    continue
                
                
                # Compute exp(sigma)
                exp_sigma = torch.exp(self.sigma[param])
                
                # Update the parameter
                param.data = self.update_func(param.data, exp_sigma, lr)  # Pass lr to the update function

                # Update sigma using the F_p function
                y = self.y_func(param.data)
                self.sigma[param] -= self.s_rate*self.F_p(exp_sigma, y)
        
        return loss
