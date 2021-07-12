from typing import Union, Tuple, Iterable

import torch
import numpy as np

from .models.models import GANModel, ReturnsModel
from .models.losses import LossCompose, WeightedLSLoss
from .utils import get_scheduler_from_config, get_optimizer_from_config, to_numpy
from .data_loader import FinanceDataset
from .portfolio_utils import sharpe, construct_long_short_portfolio
from .callbacks import EarlyStopping
from .config_reader import Config


HIDDEN_STATE_TYPE = Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]
MODEL_TYPE = Union[GANModel, ReturnsModel]
LOSS_TYPE = Union[LossCompose, WeightedLSLoss]


def forward_with_dataset(model: MODEL_TYPE, dataset: FinanceDataset,
                         hidden_state: torch.Tensor = None) -> Union[torch.Tensor, Iterable[torch.Tensor]]:
    return model(
            macro_features=dataset.macro_feat_tensor,
            individual_features=dataset.ind_feat_tensor,
            masks=dataset.masks_tensor,
            returns_tensor=dataset.returns_tensor,
            hidden_state=hidden_state)


def evaluate(model: MODEL_TYPE, dataset: FinanceDataset, loss: LOSS_TYPE,
             hidden_state: HIDDEN_STATE_TYPE, normalize_sdf: bool = False) -> Iterable[float]:

    model.eval()
    residual_loss_value = None

    if isinstance(model, GANModel):
        sdf, weights, hidden_weights = forward_with_dataset(model, dataset, hidden_state)
        main_loss_value = loss.main_loss(
            returns_tensor=dataset.returns_tensor,
            masks=dataset.masks_tensor,
            sdf=sdf,
            hidden_weights=hidden_weights)
        residual_loss_value = loss.residual_loss(
            returns_tensor=dataset.returns_tensor,
            masks=dataset.masks_tensor,
            weights=weights
        )
        sharpe_val = evaluate_sharpe_from_sdf(
            sdf, weights,
            dataset.returns_tensor,
            dataset.masks_tensor,
            normalize=normalize_sdf)

    elif isinstance(model, ReturnsModel):
        residual_returns = forward_with_dataset(model, dataset, hidden_state)
        main_loss_value = loss(returns_tensor=residual_returns, masks=dataset.masks_tensor)
        sharpe_val = evaluate_sharpe_from_residual_returns(
            dataset.returns_tensor,
            dataset.masks_tensor,
            residual_returns)
    else:
        raise ValueError(f"Model of type {type(model)} is not supported.")

    main_loss_value, residual_loss_value, sharpe_val = to_numpy(main_loss_value, residual_loss_value, sharpe_val)
    return main_loss_value, residual_loss_value, sharpe_val


def get_normalized_sdf(sdf: np.ndarray, weights: np.ndarray, returns: np.ndarray,
                       mask: np.ndarray, normalize: bool) -> np.ndarray:
    if normalize:
        splits = np.sum(mask, axis=1).cumsum()[:-1]
        weights_list = np.split(weights, splits)
        weights_array = np.concatenate([item / np.absolute(item).sum() for item in weights_list])
        if len(returns.shape) > 3:
            returns = returns[0, :]
        weighted_returns_list = np.split(returns[mask] * weights_array.flatten(), splits)
        sdf = np.array([[item.sum()] for item in weighted_returns_list]) + 1
    return sdf


def predict_returns(model: ReturnsModel, dataset: FinanceDataset):
    residual_returns = to_numpy(forward_with_dataset(model, dataset))
    predicted_return = get_predicted_returns(
        returns=dataset.individual_data.return_array,
        mask=dataset.individual_data.mask,
        residual_returns=residual_returns)
    return predicted_return


def get_predicted_returns(returns: np.ndarray,
                          mask: np.ndarray,
                          residual_returns: np.ndarray) -> np.ndarray:
    if len(returns.shape) > 3:
        returns = returns[0, :]
    returns = returns[mask]
    return returns - residual_returns.flatten()


def predict_normalized_sdf(model: GANModel, dataset: FinanceDataset, hidden_state: torch.Tensor = None) -> \
        Tuple[np.ndarray, HIDDEN_STATE_TYPE]:
    model.eval()
    if hidden_state is None:
        hidden_state = model.initialize_sdf_hidden_state()

    sdf, weights, hidden_weights = to_numpy(forward_with_dataset(model, dataset, hidden_state))
    normalized_sdf = get_normalized_sdf(sdf, weights,
                                        dataset.individual_data.return_array,
                                        dataset.individual_data.mask, normalize=True)
    output_hidden_state = model.get_last_hidden_state()
    return normalized_sdf, output_hidden_state


def evaluate_sharpe_from_sdf(sdf: torch.Tensor, weights: torch.Tensor, returns: torch.Tensor,
                             mask: torch.Tensor, normalize: bool) -> float:
    sdf, weights, returns, mask = to_numpy(sdf, weights, returns, mask)
    sdf_array = get_normalized_sdf(sdf, weights, returns, mask, normalize)
    sdf_array = 1 - sdf_array
    return sharpe(sdf_array)


def evaluate_sharpe_from_residual_returns(returns: torch.Tensor, mask: torch.Tensor,
                                          residual_returns: torch.Tensor) -> float:
    returns, mask, residual_returns = to_numpy(returns, mask, residual_returns)
    if len(returns.shape) > 3:
        returns = returns[0, :]
    returns = returns[mask]
    sdf = returns - residual_returns.flatten()
    portfolio = construct_long_short_portfolio(sdf, returns, mask[:, :, 0])
    return sharpe(portfolio)


def train_model(config: Config, epochs: int, model: MODEL_TYPE,
                loss: LOSS_TYPE, dataset_train: FinanceDataset,
                dataset_valid: FinanceDataset, dataset_test: FinanceDataset, path_to_save: str):
    # Initialize early stopper
    early_stopping = EarlyStopping()
    # Get optimizer and scheduler if applicable
    optimizer = get_optimizer_from_config(config, model.parameters())
    scheduler = get_scheduler_from_config(config, optimizer)

    # If RNN should be used - initialize random hidden state
    train_initial_hidden_state = model.initialize_sdf_hidden_state() if config['use_rnn'] else None

    last_metric_value = 1e10
    for epoch in range(epochs):
        model.train()
        for sub_epoch in range(config['sub_epoch']):
            if isinstance(model, GANModel):
                sdf, weights, hidden_weights = forward_with_dataset(model, dataset_train, train_initial_hidden_state)
                loss_tensor = loss(
                    returns_tensor=dataset_train.returns_tensor,
                    masks=dataset_train.masks_tensor,
                    sdf=sdf,
                    weights=weights,
                    hidden_weights=hidden_weights)
            elif isinstance(model, ReturnsModel):
                residual_returns = forward_with_dataset(model, dataset_train)
                loss_tensor = loss(returns_tensor=residual_returns, masks=dataset_train.masks_tensor)
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
              f" and sharpe {test_sharpe}")

        if early_stopping(valid_main_loss):
            return
        if last_metric_value > valid_main_loss:
            last_metric_value = valid_main_loss
            torch.save(model.state_dict(), path_to_save)


def train_gan(config: Config, path_to_dump: str,
              dataset_train: FinanceDataset, dataset_valid: FinanceDataset,
              dataset_test: FinanceDataset = None) -> GANModel:

    # Initialize GAN Model
    gan_model = GANModel(config=config)

    # Prepare inputs for training
    train_inputs = {
        'config': config,
        'model': gan_model,
        'dataset_train': dataset_train,
        'dataset_valid': dataset_valid,
        'dataset_test': dataset_test
    }
    # Firstly, Initialize and optimize unconditional loss
    unconditional_loss = LossCompose(minimize=True,
                                     to_weight=config['weighted_loss'],
                                     main_loss_conditional=False,
                                     residual_loss_factor=config['residual_loss_factor']
                                     )
    train_model(epochs=config['num_epochs_unc'],
                loss=unconditional_loss,
                path_to_save=f'{path_to_dump}/unconditional_model.pth',
                **train_inputs)

    # Secondly, Initialize and optimize moment conditional loss
    moment_conditional_loss = LossCompose(minimize=False,
                                          to_weight=config['weighted_loss'],
                                          main_loss_conditional=True
                                          )
    train_model(epochs=config['num_epochs_moment'],
                loss=moment_conditional_loss,
                path_to_save=f'{path_to_dump}/moment_model.pth',
                **train_inputs)

    # Lastly, initialize and optimize conditional loss
    conditional_loss = LossCompose(minimize=True,
                                   to_weight=config['weighted_loss'],
                                   main_loss_conditional=True,
                                   residual_loss_factor=config['residual_loss_factor'])
    train_model(epochs=config['num_epochs'],
                loss=conditional_loss,
                path_to_save=f'{path_to_dump}/condition_model.pth',
                **train_inputs)

    # Dump trained model
    torch.save(gan_model.state_dict(), f"{path_to_dump}/model_dump.pth")
    return gan_model


def train_returns_model(config: Config, path_to_dump: str,
                        dataset_train: FinanceDataset, dataset_valid: FinanceDataset,
                        dataset_test: FinanceDataset = None) -> ReturnsModel:

    # Initialize weighted loss
    least_squares_loss = WeightedLSLoss(to_weight=config['weighted_loss'])
    # Initialize returns model
    returns_model = ReturnsModel(config=config)

    train_model(config,
                epochs=config['num_epochs'],
                model=returns_model,
                loss=least_squares_loss,
                dataset_train=dataset_train,
                dataset_valid=dataset_valid,
                dataset_test=dataset_test,
                path_to_save=f'{path_to_dump}/returns_model.pth')
    return returns_model
