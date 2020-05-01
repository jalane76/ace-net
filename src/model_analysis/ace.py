import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

def ie_full(x, f, cov):
    y = f(x)
    h = hessian(y, x)
    result = torch.zeros_like(y)
    for i in range(len(y)):
        result[i] = y[i] + 0.5 * torch.trace(torch.matmul(h[i], cov))
    return result

def ie_approx(x, f, cov, epsilon=0.000001):
    e, v = torch.symeig(cov, eigenvectors=True)
    
    v = epsilon * e.sqrt() * v
    v1 = x - v
    v2 = x + v
    
    y = f(x)
    
    y_v1 = f(v1) - y
    y_v1 = torch.sum(y_v1, dim=0)
    
    y_v2 = f(v2) - y
    y_v2 = torch.sum(y_v2, dim=0)
    
    return y + 0.5 * (y_v1 + y_v2)