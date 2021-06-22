import torch

from .sub_networks import RecurrentNetwork, DenseNetwork


class FeatureExtractionModel:
    def __init__(self, config: dict, suffix: str, input_dropout: float = 0., output_dropout: float = 0.,
                 output_size: int = 1, dense_dropout: float = 0.,
                 hidden_activation: str = 'relu', output_activation: str = None
                 ):
        self.recurrent_net = None
        self.use_rnn = config['use_rnn']

        if self.use_rnn:
            self.recurrent_net = RecurrentNetwork.from_config(
                config, suffix=suffix,
                input_dropout=input_dropout, output_dropout=output_dropout)
        self.dense_net = DenseNetwork.from_config(
            config, suffix=suffix,
            output_size=output_size,
            dropout=dense_dropout,
            hidden_activation=hidden_activation,
            output_activation=output_activation
        )

    def extract_features(self, macro_features: torch.Tensor, individual_features: torch.Tensor,
                         masks: torch.Tensor = None, *args, **kwargs):
        if self.use_rnn:
            macro_processed_tensor = self.recurrent_net(macro_features)
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


class SDFModel(torch.nn.Module, FeatureExtractionModel):
    SUFFIX = ''

    def __init__(self, config: dict, input_dropout: float = 0., output_dropout: float = 0.,
                 output_size: int = 1, dense_dropout: float = 0.,
                 hidden_activation: str = 'relu'):
        torch.nn.Module.__init__(self)
        FeatureExtractionModel.__init__(
            self,
            config=config,
            suffix=SDFModel.SUFFIX,
            input_dropout=input_dropout,
            output_dropout=output_dropout,
            output_size=output_size,
            dense_dropout=dense_dropout,
            hidden_activation=hidden_activation,
            output_activation=None
        )
        self.normalize = config.get('normalize_w', False)

    def forward(self, macro_features: torch.Tensor, individual_features: torch.Tensor,
                masks: torch.Tensor, returns_tensor: torch.Tensor, *args, **kwargs):
        weights = self.extract_features(macro_features, individual_features, masks)
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

    def get_last_hidden_state(self):
        if self.recurrent_net is not None:
            return self.recurrent_net.last_hidden_state
        return None


class MomentsModel(torch.nn.Module, FeatureExtractionModel):
    SUFFIX = 'moment'

    def __init__(self, config: dict, input_dropout: float = 0., output_dropout: float = 0.,
                 dense_dropout: float = 0., hidden_activation: str = 'relu', output_activation: str = 'tanh'):
        torch.nn.Module.__init__(self)
        output_size = config['num_condition_moment']
        FeatureExtractionModel.__init__(
            self,
            config=config,
            suffix=MomentsModel.SUFFIX,
            input_dropout=input_dropout,
            output_dropout=output_dropout,
            output_size=output_size,
            dense_dropout=dense_dropout,
            hidden_activation=hidden_activation,
            output_activation=output_activation
        )
        self.normalize = config.get('normalize_w', False)

    def forward(self, macro_features: torch.Tensor, individual_features: torch.Tensor, *args, **kwargs):
        weights = self.extract_features(macro_features, individual_features, masks=None)
        weights = weights.permute((0, 3, 1, 2))
        return weights
