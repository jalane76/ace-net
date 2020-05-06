#! /usr/bin/env python3

from model_analysis.ace import interventional_expectation

import torch
import numpy as np

def f(x):                                                                                             
    return x ** 2

np.random.seed(435537698)

num_dims = 4
num_interventions = 2
num_samples = 3
val_range = (0.0, 1.0)

X = [np.random.uniform(*val_range, size=num_dims) for i in range(num_samples)]

mean = torch.Tensor(np.mean(X, axis=0))
print('mean: \n{}\n\n'.format(mean))

cov = torch.Tensor(np.cov(X, rowvar=False))
print('cov: \n{}\n\n'.format(cov))

interventions = torch.Tensor(np.linspace(*val_range, num=num_interventions, endpoint=True))
print('interventions: \n{}\n\n'.format(interventions))

hessian_full = interventional_expectation(f, mean, cov, interventions, method='hessian_full', progress=False)
print('hessian_full: \n{}\n\n'.format(hessian_full))
print('*****************************************************************************************************')

hessian_diag = interventional_expectation(f, mean, cov, interventions, method='hessian_diag', progress=False)
print('hessian_diag: \n{}\n\n'.format(hessian_diag))
print('*****************************************************************************************************')

approximation = interventional_expectation(f, mean, cov, interventions, method='approximate', progress=False)
print('approximation: \n{}\n\n'.format(approximation))
print('*****************************************************************************************************')

print('hessian_full == hessian_diag: {}\n'.format(torch.allclose(hessian_full.rename(None), hessian_diag.rename(None))))
print('hessian_diag == approximation: {}\n'.format(torch.allclose(hessian_diag.rename(None), approximation.rename(None))))
print('approximation == hessian_full: {}\n'.format(torch.allclose(approximation.rename(None), hessian_full.rename(None))))