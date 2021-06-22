from typing import List, Union, Tuple

import torch


class RecurrentNetwork(torch.nn.Module):
    def __init__(self, model_name: str, input_size: int, hidden_size: int, num_layers: int = 1,
                 input_dropout: float = 0., output_dropout: float = 0.):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(RecurrentNetwork, self).__init__()
        self._input_dropout_val = input_dropout
        self.input_dropout = torch.nn.Dropout(input_dropout)
        self.last_hidden_state = None
        parameters = {
            "input_size": input_size,
            "hidden_size": hidden_size,
            "num_layers": num_layers,
            "dropout": output_dropout
        }
        if model_name == 'rnn':
            self.recurrent_layer = torch.nn.RNN(**parameters)
        elif model_name == 'gru':
            self.recurrent_layer = torch.nn.GRU(**parameters)
        elif model_name == 'lstm':
            self.recurrent_layer = torch.nn.LSTM(**parameters)
        else:
            raise ValueError(f"Model name is unknown {model_name}")

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        if self._input_dropout_val:
            input_tensor = self.input_dropout(input_tensor)
        input_tensor, hidden_state = self.recurrent_layer(input_tensor)
        self.save_hidden_state(hidden_state)
        return input_tensor

    @classmethod
    def from_config(cls, config: dict, suffix: str = '', input_dropout: float = 0., output_dropout: float = 0.):
        if suffix:
            suffix = f"_{suffix}"

        hidden_size = config[f'num_units_rnn{suffix}']
        if isinstance(hidden_size, list):
            hidden_size = hidden_size[0]

        return cls(
            model_name=config[f'cell_type_rnn{suffix}'],
            input_size=config['macro_feature_dim'],
            hidden_size=hidden_size,
            num_layers=config[f"num_layers_rnn{suffix}"],
            input_dropout=input_dropout,
            output_dropout=output_dropout
        )

    def save_hidden_state(self, hidden_state: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]):
        if isinstance(hidden_state, tuple):
            self.last_hidden_state = torch.cat(hidden_state, dim=-1)
            return
        self.last_hidden_state = hidden_state


class DenseNetwork(torch.nn.Module):
    def __init__(self, input_size: int, hidden_dims: List[int], dropout: float,
                 output_size: int, hidden_activation: str, output_activation: str = None):
        super(DenseNetwork, self).__init__()
        layers_list = list()
        last_layer_dim = input_size
        for index, dim in enumerate(hidden_dims):
            layers_list.extend([
                torch.nn.Linear(in_features=last_layer_dim, out_features=dim),
                self.get_activation_by_name(hidden_activation),
                torch.nn.Dropout(dropout)
            ])
            last_layer_dim = dim
        layers_list.append(
            torch.nn.Linear(last_layer_dim, output_size)
        )
        if output_activation:
            layers_list.append(
                self.get_activation_by_name(output_activation)
            )
        self.stacked_dense_layers = torch.nn.Sequential(*layers_list)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        return self.stacked_dense_layers(input_tensor)

    @staticmethod
    def get_activation_by_name(activation_name: str):
        if activation_name == 'tanh':
            return torch.nn.Tanh()
        elif activation_name == 'relu':
            return torch.nn.ReLU()
        else:
            raise NotImplementedError(f"Activation {activation_name} "
                                      f"for DenseNetwork is not implemented. ")

    @classmethod
    def from_config(cls, config: dict, suffix: str = '', output_size: int = 1., dropout: float = 0.,
                    hidden_activation: str = 'relu', output_activation: str = None):
        if suffix:
            suffix = f"_{suffix}"
        input_size = config['individual_feature_dim']

        if config['use_rnn']:
            rnn_output_size = config[f"num_units_rnn{suffix}"]
            if isinstance(rnn_output_size, list):
                rnn_output_size = rnn_output_size[0]
            input_size += rnn_output_size
        else:
            input_size += config['macro_feature_dim']

        return cls(
            input_size=input_size,
            hidden_dims=config[f"hidden_dim{suffix}"],
            dropout=dropout,
            output_size=output_size,
            hidden_activation=hidden_activation,
            output_activation=output_activation
        )





