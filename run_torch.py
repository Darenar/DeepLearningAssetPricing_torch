import argparse
import os
import logging

from torch_version.config_reader import Config
from torch_version.data_loader import FinanceDataset
from torch_version.engine import train_gan, train_returns_model, predict_normalized_sdf, predict_returns
from torch_version.portfolio_utils import calculate_statistics


PATH_TO_GAN_CONFIG = 'configs/config.json'
PATH_TO_RETURNS_CONFIG = 'configs/config_RF_1.json'


def main(args):
    logging.basicConfig(filename=f"{args.path_to_output}/logger.txt",
                        format='%(asctime)s %(message)s',
                        datefmt='%m/%d/%Y %I:%M:%S %p',
                        level=logging.INFO)

    logging.info(f"Read config {args.path_to_config}")
    config = Config.from_json(args.path_to_config)
    logging.info("Creating datasets")

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

    logging.info("Initialize and train GAN model")
    gan_model = train_gan(config,
                          path_to_dump=args.path_to_output,
                          dataset_train=train_dataset,
                          dataset_valid=val_dataset,
                          dataset_test=test_dataset)
    logging.info("Predict normalized SDF values with trained GAN model")
    train_sdf, train_hidden_state = predict_normalized_sdf(gan_model, train_dataset)
    val_sdf, val_hidden_state = predict_normalized_sdf(gan_model, val_dataset, train_hidden_state)
    test_sdf, _ = predict_normalized_sdf(gan_model, test_dataset, val_hidden_state)

    logging.info("Multiply individual returns by obtained SDF vector")
    train_dataset.individual_data.multiply_returns_on_sdf(train_sdf, factor=args.sdf_factor)
    val_dataset.individual_data.multiply_returns_on_sdf(val_sdf, factor=args.sdf_factor)
    test_dataset.individual_data.multiply_returns_on_sdf(test_sdf, factor=args.sdf_factor)

    logging.info("Start modelling residual returns")
    config_rf = Config.from_json(args.path_to_rf_config)
    # Filter macro data by the config if necessary
    train_dataset.macro_data.filter(config_rf['macro_idx'])
    val_dataset.macro_data.filter(config_rf['macro_idx'])
    test_dataset.macro_data.filter(config_rf['macro_idx'])

    returns_model = train_returns_model(config=config_rf,
                                        path_to_dump=args.path_to_output,
                                        dataset_train=train_dataset,
                                        dataset_valid=val_dataset,
                                        dataset_test=test_dataset)
    logging.info("Predicting returns with trained Returns Model")
    train_pred_returns = predict_returns(returns_model, train_dataset)
    val_pred_returns = predict_returns(returns_model, val_dataset)
    test_pred_returns = predict_returns(returns_model, test_dataset)
    logging.info(f"Calculated train statistics: {calculate_statistics(train_pred_returns, train_dataset)}")
    logging.info(f"Calculated valid statistics: {calculate_statistics(val_pred_returns, val_dataset)}")
    logging.info(f"Calculated test statistics: {calculate_statistics(test_pred_returns, test_dataset)}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Torch Version Asset Pricing')
    parser.add_argument('--path_to_config', default=PATH_TO_GAN_CONFIG, help='Path to config file.')
    parser.add_argument('--path_to_rf_config', default=PATH_TO_RETURNS_CONFIG, help='Path to rf config file.')
    parser.add_argument('--path_to_output', default='torch_output', help='Path to output folder')
    parser.add_argument('--sdf_factor', default=50, help='Value by which to increase obtained SDF values.')
    args = parser.parse_args()
    os.makedirs(args.path_to_output, exist_ok=True)
    main(args)
