import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

def interventional_expectation(model, mean, cov, interventions, epsilon=0.000001, method='hessian_diag'):
    # TODO: improve and extend documentation
    ''' Calculates the interventional expectations on a given model and statistics.  (insert link?)

        :param model: A function or trained PyTorch model
        :type model: torch.nn.Module
        :param mean: Mean of a data distribution relevant to the model, e.g., the training data
        :type mean: torch.Tensor with shape (number_of_inputs_to_model)
        :param cov: Covariance matrix of a data distribution relevant to the model, e.g., the training data
        :type cov: torch.Tensor with shape (number_of_inputs_to_model, number_of_inputs_to_model)
        :param interventions: Interventional values that will be used to perturb the model
        :type interventions: torch.Tensor with shape (number_of_interventional_values)
        :param epsilon: Small value used in approximating the interventional expectation
        :type epsilon: float, optional
        :param method: Method for calculating the interventional expectation, can be one of ['hessian_full', 'hessian_diag', 'approximate']
        :type method" string, optional

        :return: A tensor containing the interventional expectations
        :rtype: Named torch.Tensor with shape (number_of_inputs_to_model, number_of_outputs_from_model, number_of_interventional_values) and dimension names ('X', 'Y', 'Alpha')
    '''

    if method == 'hessian_full':
        return __ie_hessian_full(x, f, cov)
    elif method == 'approximate':
        return __ie_approx(x, f, cov, epsilon)
    else:
        return __ie_hessian_diag(x, f, cov)

def __ie_hessian_full(model, mean, cov, interventions):
    out_shape = model(mean).shape

    y = f(x)
    h = hessian(y, x)
    result = torch.zeros_like(y)
    for i in range(len(y)):
        result[i] = y[i] + 0.5 * torch.trace(torch.matmul(h[i], cov))
    return result

def __ie_hessian_diag(model, mean, cov, interventions):
    # TODO: this is not actually the right way to calculate this
    # TODO: return a proper tensor
    y = f(x)
    h = hessian(y, x)
    result = torch.zeros_like(y)
    for i in range(len(y)):
        result[i] = y[i] + 0.5 * torch.trace(torch.matmul(h[i], cov))
    return result

def __ie_approx(model, mean, cov, interventions, epsilon=0.000001):
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