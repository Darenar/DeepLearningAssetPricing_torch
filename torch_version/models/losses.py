from typing import Optional

import torch
import numpy as np


class BaseLoss:
    def __init__(self, to_weight: bool):
        self.to_weight = to_weight

    def calculate_loss(self, returns: torch.Tensor, masks: torch.Tensor,
                       sdf: torch.Tensor, hidden_weights: Optional[torch.Tensor] = None, *args, **kwargs):
        weights = torch.sum(masks, dim=0)
        multiplied_tensor = returns * masks * sdf
        if hidden_weights is not None:
            multiplied_tensor *= hidden_weights
        empirical_mean = torch.sum(multiplied_tensor, dim=1) / weights
        if self.to_weight:
            weights_normalized = weights / torch.max(weights)
            return torch.mean(torch.square(empirical_mean) * weights_normalized)
        return torch.mean(torch.square(empirical_mean))


class UnconditionalLoss(BaseLoss):
    def __init__(self, to_weight: bool):
        super().__init__(to_weight=to_weight)
    def __call__(self, returns: torch.Tensor, masks: torch.Tensor,
                 sdf: torch.Tensor, *args, **kwargs):
        return self.calculate_loss(returns, masks, sdf)


class ConditionalLoss(BaseLoss):
    def __init__(self, to_weight: bool):
        super().__init__(to_weight=to_weight)

    def __call__(self, returns: torch.Tensor, masks: torch.Tensor,
                 sdf: torch.Tensor, hidden_weights: torch.Tensor, *args, **kwargs):
        return self.calculate_loss(returns, masks, sdf, hidden_weights)


class ResidualLoss:
    def __call__(self, returns_tensor: torch.Tensor, masks: torch.Tensor, weights: torch.Tensor, *args, **kwargs):
        num_val_per_time = torch.sum(torch.squeeze(masks, -1), dim=1)
        returns_masked = torch.masked_select(returns_tensor, masks).reshape(
            (1, torch.sum(masks), returns_tensor.shape[-1]))
        returns_split = torch.split(torch.squeeze(returns_masked, 0), num_val_per_time.tolist())
        weights_split = torch.split(weights, num_val_per_time.tolist())
        residual_square_list = list()
        r_square_list = list()
        for r_t, w_t in zip(returns_split, weights_split):
            r_t_hat = torch.sum(r_t * w_t) / torch.sum(w_t * w_t) * w_t
            residual_square_list.append(torch.mean(torch.square(r_t - r_t_hat)).detach().cpu().numpy())
            r_square_list.append(torch.mean(torch.square(r_t)).detach().cpu().numpy())
        return np.mean(residual_square_list) / np.mean(r_square_list)
