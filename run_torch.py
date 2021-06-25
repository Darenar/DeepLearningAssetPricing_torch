import argparse
import os
import json
import logging

from torch_version.data_loader import FinanceDataset
from torch_version.models.models import GANModel, ReturnsModel
from torch_version.models.losses import WeightedLSLoss
from torch_version.engine import train_gan, predict_normalized_sdf, train_model, predict_returns
from torch_version.portfolio_utils import calculate_statistics


def read_config(path_to_config: str) -> dict:
    with open(path_to_config, 'r') as f:
        config = json.load(f)
    if 'macro_idx' not in config:
        config['macro_idx'] = None
    return config


def main(args):
    # logging.basicConfig(filename=args.path_to_output,
    #                     format='%(asctime)s %(message)s',
    #                     datefmt='%m/%d/%Y %I:%M:%S %p',
    #                     level=logging.INFO)
    # logging.info(f"Read config {args.path_to_config}")
    config = read_config(args.path_to_config)
    # logging.info("Creating datasets")

    train_dataset = FinanceDataset.from_config(config=config, path_to_firm_chars=args.path_to_firm_chars)
    train_macro_mean, train_macro_std = train_dataset.macro_features_mean_std
    val_dataset = FinanceDataset.from_config(config,
                                             suffix='valid',
                                             path_to_firm_chars=args.path_to_firm_chars,
                                             macro_feature_mean_val=train_macro_mean,
                                             macro_feature_std_val=train_macro_std)
    test_dataset = FinanceDataset.from_config(config,
                                              suffix='test',
                                              path_to_firm_chars=args.path_to_firm_chars,
                                              macro_feature_mean_val=train_macro_mean,
                                              macro_feature_std_val=train_macro_std
                                              )
    logging.info("Initializing model")
    model = GANModel(config=config)
    train_gan(config,
              model=model,
              path_to_dump=args.path_to_output,
              dataset_train=train_dataset,
              dataset_valid=val_dataset,
              dataset_test=test_dataset)

    train_sdf, train_hidden_state = predict_normalized_sdf(model, train_dataset)
    val_sdf, val_hidden_state = predict_normalized_sdf(model, val_dataset, train_hidden_state)
    test_sdf, _ = predict_normalized_sdf(model, test_dataset, val_hidden_state)

    print(train_dataset.individual_data.return_array.shape)
    train_dataset.individual_data.multiply_returns_on_sdf(train_sdf, factor=50)
    print(train_dataset.individual_data.return_array.shape)
    val_dataset.individual_data.multiply_returns_on_sdf(val_sdf, factor=50)
    test_dataset.individual_data.multiply_returns_on_sdf(test_sdf, factor=50)
    print('STARTING RESIDUAL')
    # Training prediction model
    config_rf = read_config(args.path_to_rf_config)
    config_rf['macro_feature_dim'] = train_dataset.macro_data.features.shape[-1]
    least_squares_loss = WeightedLSLoss(to_weight=config_rf['weighted_loss'])
    returns_model = ReturnsModel(config=config_rf)
    train_model(config_rf,
                epochs=config_rf['num_epochs'],
                model=returns_model,
                loss=least_squares_loss,
                dataset_train=train_dataset,
                dataset_valid=val_dataset,
                dataset_test=test_dataset)

    train_pred_returns = predict_returns(returns_model, train_dataset)
    val_pred_returns = predict_returns(returns_model, val_dataset)
    test_pred_returns = predict_returns(returns_model, test_dataset)
    print(calculate_statistics(train_pred_returns, train_dataset))
    print(calculate_statistics(val_pred_returns, val_dataset))
    print(calculate_statistics(test_pred_returns, test_dataset))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Torch Version Asset Pricing')
    parser.add_argument('--path_to_config', help='Path to config file.')
    parser.add_argument('--path_to_rf_config', help='Path to rf config file.')
    parser.add_argument('--path_to_firm_chars', help='Path to firm chars json file.')
    parser.add_argument('--path_to_output', default='torch_output', help='Path to output folder')
    # parser.add_argument('--path_to_output', help='Path to output folder to save logs and checkpoints into.')
    # parser.add_argument('--save_freq', default=-1, help='Frequency to save checkpoints')
    # parser.add_argument("--save_log", dest="save_log", help="Save logs in output folder", action="store_true")
    # parser.add_argument("--print", dest="show", help="Show logs in the console", action="store_true")
    # parser.add_argument('--print_freq', default=128, type=int, metavar='N', help='Frequency of logs in console')
    # parser.add_argument('--warmup_steps', default=64, type=int, metavar='N', help='First epochs to ignore for printing')
    args = parser.parse_args()
    os.makedirs(args.path_to_output, exist_ok=True)
    main(args)
