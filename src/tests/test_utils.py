#! /usr/bin/env python3

from common.utils import jacobian
from common.utils import hessian

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
nd_jacobian_at_ones = nd_jacobian(np_ones)

#nd_hessian = nd.Hessian(f)

#nd_hessian_at_zeros = []
#for i in range(dims):
#    nd_hessian_at_zeros.append(nd_hessian(np_zeros[i]))

#nd_hessian_at_ones = []
#for i in range(dims):
#    nd_hessian_at_ones.append(nd_hessian(np_ones))

# Set up pytorch values
torch_zeros = torch.zeros(dims)
torch_zeros.requires_grad = True

torch_ones = torch.ones(dims)
torch_ones.requires_grad = True

pt_jacobian_at_zeros = jacobian(f(torch_zeros), torch_zeros)
pt_jacobian_at_ones = jacobian(f(torch_ones), torch_ones)

print('Jacobian at zeros equal: {}'.format(np.allclose(pt_jacobian_at_zeros.numpy(), nd_jacobian_at_zeros)))
print('Jacobian at ones equal: {}'.format(np.allclose(pt_jacobian_at_ones.numpy(), nd_jacobian_at_ones)))

pt_hessian_at_zeros = hessian(f(torch_zeros), torch_zeros)
pt_hessian_at_ones = hessian(f(torch_ones), torch_ones)
#print('Hessian at zeros equal: {}'.format(np.allclose(pt_hessian_at_zeros.numpy(), nd_hessian_at_zeros)))
#print('Hessian at ones equal: {}'.format(np.allclose(pt_hessian_at_ones.numpy(), nd_hessian_at_ones)))

torch_zeros[0] = 10.0
pt_hessian_one_hot = hessian(f(torch_zeros), torch_zeros)
print(pt_hessian_one_hot)