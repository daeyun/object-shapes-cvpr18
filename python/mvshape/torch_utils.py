"""
Generatic torch utils
"""
import numpy as np
import torch


def recursive_numpy_to_torch(data, cuda: bool = False):
    if isinstance(data, (list, tuple)):
        return [recursive_numpy_to_torch(item, cuda=cuda) for item in data]
    elif isinstance(data, dict):
        return {k: recursive_numpy_to_torch(v, cuda=cuda) for k, v in data.items()}
    elif isinstance(data, torch._TensorBase):
        if cuda:
            ret = data.cuda()
        else:
            ret = data.cpu()
        return torch.autograd.Variable(ret)
    elif isinstance(data, np.ndarray):
        return recursive_numpy_to_torch(torch.from_numpy(data), cuda=cuda)
    else:
        raise RuntimeError('unknown data type {}'.format(data))


def recursive_torch_to_numpy(data):
    if isinstance(data, (list, tuple)):
        return [recursive_torch_to_numpy(item) for item in data]
    elif isinstance(data, dict):
        return {k: recursive_torch_to_numpy(v) for k, v in data.items()}
    elif isinstance(data, torch._TensorBase):
        return data.cpu().numpy()
    elif isinstance(data, torch.autograd.Variable):
        return recursive_torch_to_numpy(data.data)
    elif isinstance(data, np.ndarray):
        return data
    else:
        raise RuntimeError('unknown data type {}'.format(data))


def recursive_module_apply(models, func):
    if isinstance(models, (list, tuple)):
        return [recursive_module_apply(item, func=func) for item in models]
    elif isinstance(models, dict):
        return {
            key: recursive_module_apply(value, func=func) for key, value in models.items()
        }
    elif isinstance(models, torch.nn.Module):
        return func(models)
    else:
        raise RuntimeError('unknown model type {}'.format(models))
