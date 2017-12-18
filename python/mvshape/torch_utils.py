"""
Generatic torch utils
"""
import numpy as np
import torch
import torch.nn
import torch.autograd


def to_python_scalar(gpu_var):
    d = gpu_var.cpu().data.numpy()
    return d.item()


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


def recursive_variable_apply(data, func):
    if isinstance(data, (list, tuple)):
        return [recursive_variable_apply(item, func) for item in data]
    elif isinstance(data, dict):
        return {k: recursive_variable_apply(v, func) for k, v in data.items()}
    elif isinstance(data, torch.autograd.Variable):
        return func(data)
    elif isinstance(data, np.ndarray):
        raise RuntimeError('expected torch Variables, not numpy arrays')
    else:
        raise RuntimeError('unknown data type {}. {}'.format(type(data), data))


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
        # raise RuntimeError('unknown model type {}'.format(models))
        return models


def recursive_train_setter(module, is_training: bool):
    if isinstance(module, (list, tuple)):
        for item in module:
            recursive_train_setter(item, is_training)
    elif isinstance(module, dict):
        for item in module.values():
            recursive_train_setter(item, is_training)
    elif isinstance(module, torch.nn.Module):
        if is_training:
            module.train()
        else:
            module.eval()
    elif isinstance(module, (np.ndarray, int, float, bool, str)):
        pass
    else:
        raise RuntimeError('unknown module {}'.format(module))


def recursive_scalar_dict_merge(results_dict: dict, accum_dict: dict):
    """
    modifies `accum_dict` in-place. returns nothing.
    """
    if isinstance(results_dict, dict):
        assert isinstance(accum_dict, dict), accum_dict
        for k in results_dict.keys():
            if k not in accum_dict:
                if isinstance(results_dict[k], dict):
                    accum_dict[k] = {}
                else:
                    accum_dict[k] = []
            recursive_scalar_dict_merge(results_dict[k], accum_dict[k])
    elif isinstance(results_dict, torch.autograd.Variable):
        recursive_scalar_dict_merge(to_python_scalar(results_dict), accum_dict)
    elif isinstance(results_dict, np.ndarray):
        recursive_scalar_dict_merge(results_dict.item(), accum_dict)
    elif isinstance(results_dict, (float, int, bool)):
        assert isinstance(accum_dict, (list, tuple))
        accum_dict.append(results_dict)


def recursive_dict_of_lists_mean(accum_dict: dict):
    """
    no side effects. returns a new nested dict of scalars.
    :return:
    """
    if isinstance(accum_dict, (list, tuple, np.ndarray)):
        return np.mean(accum_dict)
    elif isinstance(accum_dict, dict):
        return {k: recursive_dict_of_lists_mean(v) for k, v in accum_dict.items()}
    else:
        raise RuntimeError('unknown accum_dict type {}'.format(accum_dict))


def reduce_sum(vars: list) -> torch.autograd.Variable:
    assert isinstance(vars, (list, tuple))
    assert isinstance(vars[0], (torch.autograd.Variable))
    # Be careful not to modify in-place.
    ret = vars[0].clone()
    for item in vars[1:]:
        ret += item
    assert isinstance(ret, torch.autograd.Variable)
    return ret


def reduce_mean(vars: list) -> torch.autograd.Variable:
    ret = reduce_sum(vars).div(len(vars))
    assert isinstance(ret, torch.autograd.Variable)
    return ret
