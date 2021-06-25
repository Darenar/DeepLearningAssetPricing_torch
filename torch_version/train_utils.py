from typing import Iterable

import torch


def get_optimizer_from_config(config: dict, trainable_params: Iterable[torch.Tensor],
                              momentum: float = 0.9, eps: float = 1e-08, rho: float = 0.95):
    optimizer_name = config['optimizer']
    optimizer_params = {
        'params': trainable_params,
        'lr': config['learning_rate']
    }
    if optimizer_name == 'Momentum':
        optimizer_params.update({'momentum': momentum})
        return torch.optim.SGD(**optimizer_params)
    elif optimizer_name == 'AdaDelta':
        optimizer_params.update({'eps': eps,
                                 'rho': rho})
        return torch.optim.Adadelta(**optimizer_params)
    elif optimizer_name == 'Adam':
        return torch.optim.Adam(**optimizer_params)
    else:
        raise NotImplementedError(f"Optimizer {optimizer_name} is not yet implemented.")


def get_scheduler_from_config(config: dict, optimizer: torch.optim.Optimizer):
    if config.get('use_decay', False):
        return torch.optim.lr_scheduler.ExponentialLR(
            optimizer,
            gamma=config['decay_rate'],
            last_epoch=config['decay_steps'])
    return None
