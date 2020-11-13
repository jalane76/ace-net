#! /usr/bin/env python3

from common.ace import interventional_expectation

import torch
import numpy as np


def f(x):
    return x * x * x


np.random.seed(435537698)

num_dims = 4
num_interventions = 2
num_samples = 3
val_range = (0.0, 1.0)

X = [np.random.uniform(*val_range, size=num_dims) for i in range(num_samples)]

mean = torch.Tensor(np.mean(X, axis=0))
print(f"mean: \n{mean}\n\n")

cov = torch.Tensor(np.cov(X, rowvar=False))
print(f"cov: \n{cov}\n\n")

interventions = torch.Tensor(
    np.linspace(*val_range, num=num_interventions, endpoint=True)
)
print(f"interventions: \n{interventions}\n\n")

hessian_full = interventional_expectation(
    f, mean, cov, interventions, method="hessian_full", progress=False
)
print(f"hessian_full: \n{hessian_full}\n\n")
print("*" * 100)

hessian_diag = interventional_expectation(
    f, mean, cov, interventions, method="hessian_diag", progress=False
)
print(f"hessian_diag: \n{hessian_diag}\n\n")
print("*" * 100)

approximation = interventional_expectation(
    f, mean, cov, interventions, method="approximate", progress=False
)
print(f"approximation: \n{approximation}\n\n")
print("*" * 100)

print(f"hessian_full == hessian_diag: {torch.allclose(hessian_full, hessian_diag)}\n")
print(f"hessian_diag == approximation: {torch.allclose(hessian_diag, approximation)}\n")
print(f"approximation == hessian_full: {torch.allclose(approximation, hessian_full)}\n")
