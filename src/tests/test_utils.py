#! /usr/bin/env python3

# TODO: remove eventually when the package has been made installable.
import sys
sys.path.insert(0, '../utils')
# END TODO

from utils import jacobian
from utils import hessian

import torch
import numpy as np
import numdifftools as nd

def f(x):
    return x ** 2

# TODO: do a much better job of testing, perhaps with a real unit testing package.  Also, output test results to data.

dims = (2)

# Set up numdifftools values
np_zeros = np.zeros(dims)
np_ones = np.ones(dims)

nd_jacobian = nd.Jacobian(f)
nd_jacobian_at_zeros = nd_jacobian(np_zeros)
print(nd_jacobian_at_zeros)
nd_jacobian_at_ones = nd_jacobian(np_ones)
print(nd_jacobian_at_ones)

# Set up pytorch values
torch_zeros = torch.zeros(dims)
torch_zeros.requires_grad = True

torch_ones = torch.ones(dims)
torch_ones.requires_grad = True

pt_jacobian_at_zeros = jacobian(f(torch_zeros) * 1.0, torch_zeros)
print(pt_jacobian_at_zeros)
pt_jacobian_at_ones = jacobian(f(torch_ones) * 1.0, torch_ones)
print(pt_jacobian_at_ones)