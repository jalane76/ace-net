import inspect
import torch

# Math

def jacobian(y, x, create_graph=False):
    jac = []
    flat_y = y.reshape(-1)
    for i in range(len(flat_y)):
        grad_y = torch.zeros_like(flat_y)
        grad_y[i] = 1.0
        grad_x, = torch.autograd.grad(flat_y, x, grad_outputs=grad_y, retain_graph=True, create_graph=create_graph)
        jac.append(grad_x.reshape(x.shape))
    return torch.stack(jac).reshape(y.shape + x.shape)
                                                                                                      
def hessian(y, x):
    return jacobian(jacobian(y, x, create_graph=True), x)


# Module manipulation

def get_model_from_module(module, model_name):
    model = None
    for name, cls in inspect.getmembers(module, inspect.isclass):
        if name == model_name:
            model = cls()
            break
    return model