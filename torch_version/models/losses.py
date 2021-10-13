from typing import Optional

import torch


class MinimizeFlag:
    @property
    def minimize(self):
        if hasattr(self, '_minimize'):
            return self._minimize
        return True


class BaseLoss:
    def __init__(self, to_weight: bool):
        self.to_weight = to_weight

    def _calculate_loss(self, returns_tensor: torch.Tensor, masks: torch.Tensor,
                        sdf: torch.Tensor, moments: Optional[torch.Tensor] = None, *args, **kwargs):
        weights = self.get_weights_from_mask(masks)
        if len(returns_tensor.shape) > 3:
            returns_tensor = torch.squeeze(returns_tensor, 0)
        multiplied_tensor = (torch.squeeze(returns_tensor * masks, -1) * sdf)
        if moments is not None:
            if len(moments.shape) > 3:
                moments = torch.squeeze(moments, 0)
            multiplied_tensor = moments * multiplied_tensor
        else:
            multiplied_tensor = torch.unsqueeze(multiplied_tensor, 0)
        empirical_mean = torch.div(torch.sum(multiplied_tensor, dim=1), weights)
        return self.mean_square(empirical_mean, weights if self.to_weight else None)

    @staticmethod
    def get_weights_from_mask(masks: torch.Tensor) -> torch.Tensor:
        weights = torch.sum(masks, dim=0)
        weights = torch.squeeze(weights, -1)
        return weights

    @staticmethod
    def mean_square(output_tensor: torch.Tensor, weights: torch.Tensor = None, normalize_by_max: bool = True):
        squared_output = torch.square(output_tensor)
        if weights is not None:
            denominator = torch.max(weights) if normalize_by_max else torch.sum(weights)
            weights = weights / denominator
            squared_output *= weights
            return torch.mean(squared_output) if normalize_by_max else torch.sum(squared_output)
        return torch.mean(squared_output)


class UnconditionalLoss(BaseLoss):
    def __call__(self, returns_tensor: torch.Tensor, masks: torch.Tensor,
                 sdf: torch.Tensor, *args, **kwargs):
        return self._calculate_loss(returns_tensor, masks, sdf)


class ConditionalLoss(BaseLoss):
    def __call__(self, returns_tensor: torch.Tensor, masks: torch.Tensor,
                 sdf: torch.Tensor, moments: torch.Tensor, *args, **kwargs):
        return self._calculate_loss(returns_tensor, masks, sdf, moments)


class WeightedLSLoss(BaseLoss, MinimizeFlag):
    def __call__(self, returns_tensor: torch.Tensor, masks: torch.Tensor, *args, **kwargs):
        weights = self.get_weights_from_mask(masks)
        return self.mean_square(returns_tensor, weights if self.to_weight else None, normalize_by_max=False)


class ResidualLoss:
    def __call__(self, returns_tensor: torch.Tensor, masks: torch.Tensor, sdf_weights: torch.Tensor, *args, **kwargs):
        num_val_per_time = torch.sum(torch.squeeze(masks, -1), dim=1)
        returns_masked = torch.masked_select(returns_tensor, masks).reshape(
            (torch.sum(masks), returns_tensor.shape[-1]))
        sdf_weights = torch.squeeze(sdf_weights, 0)
        returns_split = torch.split(returns_masked, num_val_per_time.tolist())
        sdf_weights_split = torch.split(sdf_weights, num_val_per_time.tolist())
        residual_square_list = list()
        r_square_list = list()
        for r_t, w_t in zip(returns_split, sdf_weights_split):
            r_t_hat = torch.sum(r_t * w_t) / torch.sum(w_t * w_t) * w_t
            residual_square_list.append(torch.mean(torch.square(r_t - r_t_hat)))
            r_square_list.append(torch.mean(torch.square(r_t)))

        return torch.mean(torch.tensor(residual_square_list)) / torch.mean(torch.tensor(r_square_list))


class LossCompose(MinimizeFlag):
    def __init__(self, minimize: bool, to_weight: bool, main_loss_conditional: bool,
                 residual_loss_factor: float = 0.):
        self.residual_loss_factor = residual_loss_factor
        if main_loss_conditional:
            self.main_loss = ConditionalLoss(to_weight=to_weight)
        else:
            self.main_loss = UnconditionalLoss(to_weight=to_weight)
        self.residual_loss = ResidualLoss()
        self._minimize = minimize
        if not self.minimize and self.residual_loss_factor:
            raise NotImplementedError('Can not maximize with residual loss')

    def __call__(self, returns_tensor: torch.Tensor, masks: torch.Tensor, sdf: torch.Tensor,
                 sdf_weights: torch.Tensor = None, moments: Optional[torch.Tensor] = None, *args, **kwargs):
        main_loss_factor = 1 if self.minimize else -1
        loss = main_loss_factor * self.main_loss(
            returns_tensor=returns_tensor,
            masks=masks,
            sdf=sdf,
            moments=moments)
        if self.residual_loss_factor:
            loss += self.residual_loss(returns_tensor, masks, sdf_weights) * self.residual_loss_factor
        return loss
