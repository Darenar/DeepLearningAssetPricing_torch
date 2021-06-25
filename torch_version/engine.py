from typing import Union, Tuple

import torch
import numpy as np

from .models.models import GANModel, ReturnsModel
from .models.losses import LossCompose, WeightedLSLoss
from .train_utils import get_optimizer_from_config, get_scheduler_from_config
from .data_loader import FinanceDataset
from .portfolio_utils import sharpe, construct_long_short_portfolio


HIDDEN_STATE_TYPE = Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]


def evaluate(model: Union[GANModel, ReturnsModel], dataset: FinanceDataset, loss: Union[LossCompose, WeightedLSLoss],
             hidden_state: HIDDEN_STATE_TYPE, normalize_sdf: bool = False):
    model.eval()
    main_loss_value = None
    residual_loss_value = None
    sharpe_val = None
    for macro_tensor, ind_feat_tensor, return_tensor, mask_tensor in dataset.iterator():
        if isinstance(model, GANModel):
            sdf, weights, hidden_weights = model(
                macro_tensor, ind_feat_tensor, mask_tensor,
                return_tensor, hidden_state=hidden_state)
            main_loss_value = loss.main_loss(
                returns_tensor=return_tensor,
                masks=mask_tensor,
                sdf=sdf,
                hidden_weights=hidden_weights).detach().cpu().numpy()
            residual_loss_value = loss.residual_loss(
                returns_tensor=return_tensor,
                masks=mask_tensor,
                weights=weights
            ).detach().cpu().numpy()
            sharpe_val = evaluate_sharpe_from_sdf(sdf, weights, return_tensor, mask_tensor, normalize=normalize_sdf)

        elif isinstance(model, ReturnsModel):
            residual_returns = model(
                macro_tensor, ind_feat_tensor, mask_tensor,
                return_tensor, hidden_state=hidden_state)
            main_loss_value = loss(returns_tensor=residual_returns, masks=mask_tensor)
            sharpe_val = evaluate_sharpe_from_residual_returns(return_tensor, mask_tensor, residual_returns)

    return main_loss_value, residual_loss_value, sharpe_val


def get_normalized_sdf(sdf: torch.Tensor, weights: torch.Tensor, returns: torch.Tensor,
                       mask: torch.Tensor, normalize: bool):
    weights_array = weights.detach().cpu().numpy()
    sdf_array = sdf.detach().cpu().numpy()
    mask_array = mask.detach().cpu().numpy()
    returns_array = returns.detach().cpu().numpy()
    if normalize:
        splits = np.sum(mask_array, axis=1).cumsum()[:-1]
        weights_list = np.split(weights_array, splits)
        weights_array = np.concatenate([item / np.absolute(item).sum() for item in weights_list])
        if len(returns_array.shape) > 3:
            returns_array = returns_array[0, :]
        weighted_returns_list = np.split(returns_array[mask_array] * weights_array.flatten(), splits)
        sdf_array = np.array([[item.sum()] for item in weighted_returns_list]) + 1
    return sdf_array


def predict_returns(model: ReturnsModel, dataset: FinanceDataset):
    for macro_tensor, ind_feat_tensor, return_tensor, mask_tensor in dataset.iterator():
        residual_returns = model(macro_tensor,
                                 ind_feat_tensor,
                                 mask_tensor,
                                 return_tensor,
                                 hidden_state=None)
        predicted_return = get_predicted_returns(
            returns_tensor=return_tensor,
            mask_tensor=mask_tensor,
            residual_returns=residual_returns)
        return predicted_return


def get_predicted_returns(returns_tensor: torch.Tensor, mask_tensor: torch.Tensor, residual_returns: torch.Tensor):
    mask_array = mask_tensor.detach().cpu().numpy()
    returns_array = returns_tensor.detach().cpu().numpy()
    residual_returns_array = residual_returns.detach().cpu().numpy()
    returns_array = returns_array[mask_array]
    return returns_array - residual_returns_array


def predict_normalized_sdf(model: GANModel, dataset: FinanceDataset, hidden_state: torch.Tensor = None):
    model.eval()
    normalized_sdf = None
    output_hidden_state = None
    if hidden_state is None:
        hidden_state = model.initialize_sdf_hidden_state(
            dataset.get_sequence_length())
    for macro_tensor, ind_feat_tensor, return_tensor, mask_tensor in dataset.iterator():
        sdf, weights, hidden_weights = model(macro_tensor,
                                             ind_feat_tensor,
                                             mask_tensor,
                                             return_tensor,
                                             hidden_state=hidden_state)
        normalized_sdf = get_normalized_sdf(sdf, weights, return_tensor, mask_tensor, normalize=True)
        output_hidden_state = model.get_last_hidden_state()
    return normalized_sdf, output_hidden_state


def evaluate_sharpe_from_sdf(sdf: torch.Tensor, weights: torch.Tensor, returns: torch.Tensor,
                             mask: torch.Tensor, normalize: bool):
    sdf_array = get_normalized_sdf(sdf, weights, returns, mask, normalize)
    sdf_array = 1 - sdf_array
    return sharpe(sdf_array)


def evaluate_sharpe_from_residual_returns(returns: torch.Tensor, mask: torch.Tensor, residual_returns: torch.Tensor):
    mask_array = mask.detach().cpu().numpy()
    returns_array = returns.detach().cpu().numpy()
    if len(returns_array.shape) > 3:
        returns_array = returns_array[0, :]
    residual_returns_array = residual_returns.detach().cpu().numpy()
    returns_array = returns_array[mask_array]
    sdf = returns_array - residual_returns_array
    portfolio = construct_long_short_portfolio(sdf, returns_array, mask_array)
    return sharpe(portfolio)


def train_model(config: dict, epochs: int, model: Union[GANModel, ReturnsModel],
                loss: Union[LossCompose, WeightedLSLoss], dataset_train: FinanceDataset,
                dataset_valid: FinanceDataset, dataset_test: FinanceDataset = None):
    optimizer = get_optimizer_from_config(config, model.parameters())
    scheduler = get_scheduler_from_config(config, optimizer)
    train_initial_hidden_state = None
    if config.get('use_rnn', False):
        train_initial_hidden_state = model.initialize_sdf_hidden_state(
            dataset_train.get_sequence_length())
    for epoch in range(epochs):
        model.train()
        # for macro_tensor, ind_feat_tensor, return_tensor, mask_tensor in dataset_train.iterator(config['sub_epoch']):
        for macro_tensor, ind_feat_tensor, return_tensor, mask_tensor in dataset_train.iterator():
            if isinstance(model, GANModel):
                sdf, weights, hidden_weights = model(macro_features=macro_tensor,
                                                     individual_features=ind_feat_tensor,
                                                     masks=mask_tensor,
                                                     returns_tensor=return_tensor,
                                                     hidden_state=train_initial_hidden_state)

                loss_tensor = loss(
                    returns_tensor=return_tensor,
                    masks=mask_tensor,
                    sdf=sdf,
                    weights=weights,
                    hidden_weights=hidden_weights)
            elif isinstance(model, ReturnsModel):
                print('RETURNS MODEL')
                residual_returns = model(
                    macro_tensor, ind_feat_tensor, mask_tensor,
                    return_tensor, hidden_state=None)
                loss_tensor = loss(returns_tensor=residual_returns, masks=mask_tensor)
            else:
                raise NotImplementedError

            optimizer.zero_grad()
            loss_tensor.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
        train_main_loss, train_residual_loss, train_sharpe = evaluate(
            model=model,
            dataset=dataset_train,
            loss=loss,
            hidden_state=train_initial_hidden_state)
        train_hidden_state = model.get_last_hidden_state()
        valid_main_loss, valid_residual_loss, valid_sharpe = evaluate(
            model=model,
            dataset=dataset_valid,
            loss=loss,
            hidden_state=train_hidden_state)
        valid_hidden_state = model.get_last_hidden_state()
        test_main_loss, test_residual_loss, test_sharpe = evaluate(
            model=model,
            dataset=dataset_test,
            loss=loss,
            hidden_state=valid_hidden_state)
        print(f"Train main loss: {train_main_loss} "
              f"and residual loss: {train_residual_loss}"
              f" and sharpe {train_sharpe}")
        print(f"Valid main loss: {valid_main_loss} "
              f"and residual loss: {valid_residual_loss}"
              f" and sharpe {valid_sharpe}")
        print(f"Test main loss: {test_main_loss} "
              f"and residual loss: {test_residual_loss}" 
              f"and sharpe {test_sharpe}")


def train_gan(config: dict, model: GANModel, path_to_dump: str,
              dataset_train: FinanceDataset, dataset_valid: FinanceDataset,
              dataset_test: FinanceDataset = None):
    train_inputs = {
        'config': config,
        'model': model,
        'dataset_train': dataset_train,
        'dataset_valid': dataset_valid,
        'dataset_test': dataset_test
    }
    unconditional_loss = LossCompose(minimize=True,
                                     to_weight=config['weighted_loss'],
                                     main_loss_conditional=False,
                                     residual_loss_factor=config.get('residual_loss_factor', 0.)
                                     )
    train_model(epochs=1,  # config['num_epochs_unc'],
                loss=unconditional_loss,
                **train_inputs)

    moment_conditional_loss = LossCompose(minimize=False,
                                          to_weight=config['weighted_loss'],
                                          main_loss_conditional=True
                                          )
    train_model(epochs=1,  # config['num_epochs_unc'],
                loss=moment_conditional_loss,
                **train_inputs)

    conditional_loss = LossCompose(minimize=True,
                                   to_weight=config['weighted_loss'],
                                   main_loss_conditional=True,
                                   residual_loss_factor=config.get('residual_loss_factor', 0.))
    train_model(epochs=1,  # config['num_epochs_unc'],
                loss=conditional_loss,
                **train_inputs)
    torch.save(model.state_dict(), f"{path_to_dump}/model_dump.pth")
