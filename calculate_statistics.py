import argparse
import os

import torch


from torch_version.config_reader import Config
from torch_version.data_loader import FinanceDataset
from torch_version.models.models import GANModel, ReturnsModel
from torch_version.engine import predict_normalized_sdf, predict_returns
from torch_version.portfolio_utils import calculate_statistics


PATH_TO_GAN_CONFIG = 'configs/config.json'
PATH_TO_RETURNS_CONFIG = 'configs/config_RF_1.json'


def main(args):
    config = Config.from_json(args.path_to_config)
    train_dataset = FinanceDataset.from_config(config)
    train_macro_mean, train_macro_std = train_dataset.macro_features_mean_std
    val_dataset = FinanceDataset.from_config(config,
                                             suffix='valid',
                                             macro_feature_mean_val=train_macro_mean,
                                             macro_feature_std_val=train_macro_std)
    test_dataset = FinanceDataset.from_config(config,
                                              suffix='test',
                                              macro_feature_mean_val=train_macro_mean,
                                              macro_feature_std_val=train_macro_std
                                              )
    gan_model = GANModel(config=config)
    if not os.path.exists(f"{args.path_to_models}/model_dump.pth"):
        raise ValueError(f"Could not find pretrained GAN model in {args.path_to_models}")
    gan_model.load_state_dict(torch.load(f"{args.path_to_models}/model_dump.pth"))

    train_sdf, train_hidden_state = predict_normalized_sdf(gan_model, train_dataset)
    val_sdf, val_hidden_state = predict_normalized_sdf(gan_model, val_dataset, train_hidden_state)
    test_sdf, _ = predict_normalized_sdf(gan_model, test_dataset, val_hidden_state)

    train_dataset.individual_data.multiply_returns_on_sdf(train_sdf, factor=args.sdf_factor)
    val_dataset.individual_data.multiply_returns_on_sdf(val_sdf, factor=args.sdf_factor)
    test_dataset.individual_data.multiply_returns_on_sdf(test_sdf, factor=args.sdf_factor)

    config_rf = Config.from_json(args.path_to_rf_config)
    # Filter macro data by the config if necessary
    train_dataset.macro_data.filter(config_rf['macro_idx'])
    val_dataset.macro_data.filter(config_rf['macro_idx'])
    test_dataset.macro_data.filter(config_rf['macro_idx'])

    returns_model = ReturnsModel(config=config_rf)
    if not os.path.exists(f"{args.path_to_models}/returns_model.pth"):
        raise ValueError(f"Could not find pretrained returns model in {args.path_to_models}")
    returns_model.load_state_dict(torch.load(f"{args.path_to_models}/returns_model.pth"))
    train_pred_returns = predict_returns(returns_model, train_dataset)
    val_pred_returns = predict_returns(returns_model, val_dataset)
    test_pred_returns = predict_returns(returns_model, test_dataset)
    print(f"Calculated train statistics: {calculate_statistics(train_pred_returns, train_dataset)}")
    print(f"Calculated valid statistics: {calculate_statistics(val_pred_returns, val_dataset)}")
    print(f"Calculated test statistics: {calculate_statistics(test_pred_returns, test_dataset)}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Calculate statistics with pre-trained models')
    parser.add_argument('--path_to_config', default=PATH_TO_GAN_CONFIG, help='Path to config file.')
    parser.add_argument('--path_to_rf_config', default=PATH_TO_RETURNS_CONFIG, help='Path to rf config file.')
    parser.add_argument('--path_to_models', default='torch_output', help='Path to folder with model dumps')
    parser.add_argument('--sdf_factor', default=50, help='Value by which to increase obtained SDF values.')
    args = parser.parse_args()
    os.makedirs(args.path_to_output, exist_ok=True)
    main(args)
