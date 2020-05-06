import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from tqdm import tqdm, trange

from utils.utils import hessian

def average_causal_effect(interventional_expectation):
    ''' Calculates the average causal effect from the interventional expectations.

        :param interventional_expectation: The interventional expectations on a given model and statistics.
        :type interventional_expectation: Named torch.Tensor with dimension names ('X', 'Y', 'I')

        :return: A tensor containing the average causal effects.
        :rtype: Named torch.Tensor with the same shape as the input tensor and dimension names ('X', 'Y', 'I')
    '''
    size_x = interventional_expectation.size('X')
    size_y = interventional_expectation.size('Y')
    names = interventional_expectation.names
    interventional_expectation = interventional_expectation.rename(None)

    result = torch.zeros_like(interventional_expectation)
    for x in range(size_x):
        for y in range(size_y):
            result[x, y, :] = interventional_expectation[x, y, :] - torch.mean(interventional_expectation[x, y, :])

    interventional_expectation = interventional_expectation.rename(*names)

    return result

def interventional_expectation(model, mean, cov, interventions, epsilon=0.000001, method='hessian_diag', progress=False):
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
        :rtype: Named torch.Tensor with shape (number_of_inputs_to_model, number_of_outputs_from_model, number_of_interventional_values) and dimension names ('X', 'Y', 'I')
    '''

    result_shape = mean.shape + model(mean).shape + interventions.shape
    result = torch.zeros(result_shape, names=('X', 'Y', 'I'))

    if method == 'hessian_full':
        return __ie_hessian_full(model, mean, cov, interventions, result, progress=progress)
    elif method == 'approximate':
        return __ie_approx(model, mean, cov, interventions, result, epsilon=epsilon, progress=progress)
    else:
        return __ie_hessian_diag(model, mean, cov, interventions, result, progress=progress)

def __ie_hessian_full(model, mean, cov, interventions, result, progress=False):

    with tqdm(total=result.size('X') * result.size('Y') * result.size('I'), disable=not progress) as pbar:
        for x in range(result.size('X')):
            cov_row = cov[x, :].clone()  # hold out the covariance row we'll intervene upon
            cov_col = cov[:, x].clone()  # hold out the covariance col we'll intervene upon
            cov[x, :] = 0.0  # zero covariances for intervened input value
            cov[:, x] = 0.0
            
            
            for i in range(result.size('I')):
                inp = mean.clone().detach()
                inp[x] = interventions[i]
                inp.requires_grad = True

                output = model(inp)
                
                h = hessian(output, inp)
                
                for y in range(result.size('Y')):
                    result[x, y, i] = output[y] + 0.5 * torch.trace(torch.matmul(h[y], cov))
                    pbar.update(1)
            
            cov[x, :] = cov_row  # restore covariances
            cov[:, x] = cov_col

    return result

def __ie_hessian_diag(model, mean, cov, interventions, result, progress=False):

    with tqdm(total=result.size('X') * result.size('Y') * result.size('I'), disable=not progress) as pbar:
        for y in range(result.size('Y')):
            for x in range(result.size('X')):
                for i in range(result.size('I')):
                    inp = mean.clone().detach()
                    inp[x] = interventions[i]
                    inp.requires_grad = True
                    
                    output = model(inp)

                    result[x, y, i] = output[y]

                    grad_mask = torch.zeros_like(output)
                    grad_mask[y] = 1.0

                    grads = autograd.grad(output, inp, grad_outputs=grad_mask, retain_graph=True, create_graph=True)

                    for xx in range(result.size('X')):
                        if xx == x:
                            continue
                    
                        cov_val = cov[xx, x].clone()  # hold out the covariance value we'll intervene upon
                        cov[xx, x] = 0.0  # zero covariances for intervened input value

                        hess_mask = torch.zeros_like(inp)
                        hess_mask[xx] = 1.0

                        h, = autograd.grad(grads, inp, grad_outputs=hess_mask, retain_graph=True, create_graph=False)
                        result[x, y, i] = result[x, y, i] + torch.sum(0.5 * h * cov[xx])

                        cov[xx, x] = cov_val  # restore held out covariance value
                    pbar.update(1)

    return result

def __ie_approx(model, mean, cov, interventions, result, epsilon=0.000001, progress=False):
    with tqdm(total=result.size('X') * result.size('I'), disable=not progress) as pbar:
        for x in range(result.size('X')):
            mean_x = mean[x].clone()    # hold out the mean value we'll intervene upon
            
            cov_row = cov[x, :].clone()  # hold out the covariance row we'll intervene upon
            cov_col = cov[:, x].clone()  # hold out the covariance col we'll intervene upon
            cov[x, :] = 0.0  # zero covariances for intervened input value
            cov[:, x] = 0.0

            e, v = torch.symeig(cov, eigenvectors=True)
            e[e < 0.0] = 0.0    # Just in case there are numerical shenanigans

            v = epsilon * e.sqrt() * v
            
            for i in range(result.size('I')):
                mean[x] = interventions[i]

                v1 = mean - v
                v2 = mean + v

                output_mean = model(mean)

                output_v1 = model(v1)
                output_v1 -= output_mean
                output_v1 = output_v1.sum(dim=0)
                
                output_v2 = model(v2)
                output_v2 -= output_mean
                output_v2 = output_v2.sum(dim=0)

                result[x, :, i] = output_mean + 0.5 * (output_v1 + output_v2)

                pbar.update(1)

            mean[x] = mean_x  # restore held out mean value

            cov[x, :] = cov_row  # restore covariances
            cov[:, x] = cov_col

    return result