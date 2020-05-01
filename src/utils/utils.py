import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np 

# Math

def jacobian(y, x, create_graph=False):
    jac = []
    flat_y = y.reshape(-1)     
    grad_y = torch.zeros_like(flat_y)     
    
    for i in range(len(flat_y)):         
        grad_y[i] = 1.
        grad_x, = torch.autograd.grad(flat_y, x, grad_y, retain_graph=True, create_graph=create_graph)
        jac.append(grad_x.reshape(x.shape))
        grad_y[i] = 0.
    return torch.stack(jac).reshape(y.shape + x.shape)           
                                                                                                      
def hessian(y, x):
    return jacobian(jacobian(y, x, create_graph=True), x)
