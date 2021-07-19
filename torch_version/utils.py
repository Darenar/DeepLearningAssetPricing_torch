from typing import Callable, Iterable, List, Union

import numpy as np
import torch

from .config_reader import Config


def try_to_cuda(func: Callable):
    def to_cuda(*args):
        cuda_args = [x.cuda() for x in args]
        if len(cuda_args) == 1:
            return cuda_args[0]
        return cuda_args

    def wrapper(*args, **kwargs):
        output_tensor = func(*args, **kwargs)
        if torch.cuda.is_available():
            if isinstance(output_tensor, tuple):
                return to_cuda(*output_tensor)
            return to_cuda(output_tensor)
        return output_tensor

    return wrapper


def get_optimizer_from_config(config: Config, trainable_params: Iterable[torch.Tensor], momentum: float = 0.9,
                              eps: float = 1e-08, rho: float = 0.95):
    optimizer_name = config['optimizer']
    optimizer_params = {'params': trainable_params, 'lr': config['learning_rate']}
    if optimizer_name == 'Momentum':
        optimizer_params.update({'momentum': momentum})
        return torch.optim.SGD(**optimizer_params)
    elif optimizer_name == 'AdaDelta':
        optimizer_params.update({'eps': eps, 'rho': rho})
        return torch.optim.Adadelta(**optimizer_params)
    elif optimizer_name == 'Adam':
        return torch.optim.Adam(**optimizer_params)
    else:
        raise NotImplementedError(f"Optimizer {optimizer_name} is not yet implemented.")


def get_scheduler_from_config(config: Config, optimizer: torch.optim.Optimizer):
    if config['use_decay']:
        return torch.optim.lr_scheduler.ExponentialLR(
            optimizer,
            gamma=config['decay_rate'],
            last_epoch=config['decay_steps'])
    return None


def to_numpy(*args) -> Union[np.ndarray, List[np.ndarray]]:
    if len(args) == 1 and isinstance(args[0], tuple):
        args = args[0]
    numpy_args = [x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else x for x in args]
    if len(numpy_args) == 1:
        return numpy_args[0]
    return numpy_args
