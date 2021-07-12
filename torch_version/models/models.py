from typing import Union, Tuple, Optional

import torch

from .sub_networks import RecurrentNetwork, DenseNetwork
from ..utils import try_to_cuda
from ..config_reader import Config


HIDDEN_STATE_TYPE = Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]


class FeatureExtractionModel(torch.nn.Module):
    SUFFIX = ''

    def __init__(self, config: Config, input_dropout: float = 0., output_dropout: float = 0.,
                 output_size: int = 1, dense_dropout: float = 0.,
                 hidden_activation: str = 'relu', output_activation: str = None
                 ):
        super().__init__()
        self.recurrent_net = None

        if config.use_rnn:
            self.recurrent_net = RecurrentNetwork.from_config(
                config, suffix=self.SUFFIX,
                input_dropout=input_dropout, output_dropout=output_dropout)
        self.dense_net = DenseNetwork.from_config(
            config, suffix=self.SUFFIX,
            output_size=output_size,
            dropout=dense_dropout,
            hidden_activation=hidden_activation,
            output_activation=output_activation
        )
        self.normalize = config.normalize_w

    def extract_features(self, macro_features: torch.Tensor, individual_features: torch.Tensor,
                         masks: torch.Tensor = None,
                         hidden_state: HIDDEN_STATE_TYPE = None,
                         *args, **kwargs):
        if self.recurrent_net is not None:
            macro_processed_tensor = self.recurrent_net(macro_features, hidden_state)
        else:
            macro_processed_tensor = macro_features
        # Expand second dim (1, time, feature) -> (1, time, 1, feature)
        macro_dense_input = torch.unsqueeze(
            macro_processed_tensor, 2)
        # Tile tensor for each n users (1, time, 1, feature) -> (1, time, n_users, feature)
        macro_dense_input = torch.tile(
            macro_dense_input, (1, 1, individual_features.shape[2], 1))
        dense_inputs = torch.cat([individual_features, macro_dense_input], dim=-1)

        # Mask invalid elements if applicable
        if masks is not None:
            dense_inputs = torch.masked_select(dense_inputs, masks).reshape(
                (1, torch.sum(masks), dense_inputs.shape[-1]))
        dense_output = self.dense_net(dense_inputs)

        return dense_output

    def forward(self, macro_features: torch.Tensor, individual_features: torch.Tensor,
                masks: torch.Tensor, returns_tensor: torch.Tensor, hidden_state: HIDDEN_STATE_TYPE, *args, **kwargs):
        return self.extract_features(macro_features, individual_features, masks, hidden_state)


class SDFModel(FeatureExtractionModel):
    SUFFIX = ''

    def forward(self, macro_features: torch.Tensor, individual_features: torch.Tensor,
                masks: torch.Tensor, returns_tensor: torch.Tensor, hidden_state: HIDDEN_STATE_TYPE, *args, **kwargs):
        weights = self.extract_features(macro_features, individual_features, masks, hidden_state)
        returns_masked = torch.masked_select(returns_tensor, masks).reshape(
                (1, torch.sum(masks), returns_tensor.shape[-1]))

        weighted_returns = returns_masked * weights

        num_val_per_time = torch.sum(torch.squeeze(masks, -1), dim=1)
        weighted_returns_split = torch.split(torch.squeeze(weighted_returns, 0), num_val_per_time.tolist())
        sum_of_returns_per_time = torch.cat([
            torch.unsqueeze(torch.unsqueeze(torch.sum(x), 0), 1) for x in weighted_returns_split
        ], dim=0)

        if self.normalize:
            mean_num_val_per_time = torch.mean(num_val_per_time)
            sum_of_returns_per_time = sum_of_returns_per_time / num_val_per_time * mean_num_val_per_time

        sdf = sum_of_returns_per_time + 1
        return sdf, weights


class MomentsModel(FeatureExtractionModel):
    SUFFIX = 'moment'

    def forward(self, macro_features: torch.Tensor, individual_features: torch.Tensor, *args, **kwargs):
        weights = self.extract_features(macro_features, individual_features, masks=None)
        weights = weights.permute((0, 3, 1, 2))
        return weights


class ReturnsModel(FeatureExtractionModel):
    SUFFIX = ''

    def forward(self, macro_features: torch.Tensor, individual_features: torch.Tensor,
                masks: torch.Tensor, returns_tensor: torch.Tensor, *args, **kwargs):
        returns_pred = self.extract_features(macro_features, individual_features, masks)
        returns_masked = torch.masked_select(returns_tensor, masks).reshape(
            (1, torch.sum(masks), returns_tensor.shape[-1]))
        return returns_masked - returns_pred

    @try_to_cuda
    def initialize_sdf_hidden_state(self) -> Optional[torch.Tensor]:
        if self.recurrent_net is not None:
            return self.recurrent_net.initialize_hidden_state()
        return None

    def get_last_hidden_state(self):
        if self.recurrent_net is not None:
            return self.recurrent_net.last_hidden_state
        return None


class GANModel(torch.nn.Module):
    def __init__(self, config: Config, input_dropout: float = 0., output_dropout: float = 0.,
                 sdf_output_size: int = 1, dense_dropout: float = 0.,
                 hidden_activation: str = 'relu', moment_output_activation: str = 'tanh'):
        super().__init__()
        model_params = {
            'config': config,
            'input_dropout': input_dropout,
            'output_dropout': output_dropout,
            'dense_dropout': dense_dropout,
            'hidden_activation': hidden_activation,
        }
        self.sdf_net = SDFModel(
            output_size=sdf_output_size, **model_params)
        self.moment_net = MomentsModel(
            output_size=config.num_condition_moment,
            output_activation=moment_output_activation, **model_params)

    def forward(self, macro_features: torch.Tensor, individual_features: torch.Tensor,
                masks: torch.Tensor, returns_tensor: torch.Tensor, hidden_state: HIDDEN_STATE_TYPE, *args, **kwargs):
        sdf, weights = self.sdf_net(macro_features, individual_features, masks, returns_tensor, hidden_state)
        hidden_weights = self.moment_net(macro_features, individual_features)
        return sdf, weights, hidden_weights

    @try_to_cuda
    def initialize_sdf_hidden_state(self) -> Optional[torch.Tensor]:
        if self.sdf_net.recurrent_net is not None:
            return self.sdf_net.recurrent_net.initialize_hidden_state()
        return None

    def get_last_hidden_state(self):
        if self.sdf_net.recurrent_net is not None:
            return self.sdf_net.recurrent_net.last_hidden_state
        return None



