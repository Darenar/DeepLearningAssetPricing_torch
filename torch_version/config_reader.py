from typing import List, Union

import json
from dataclasses import dataclass, field


@dataclass
class Config:
    learning_rate: float = 0.001
    optimizer: str = 'Adam'
    weighted_loss: bool = False
    loss_factor: float = 1.0
    dropout: float = 0.95
    sub_epoch: Union[int, bool] = False
    normalize_w: bool = False
    residual_loss_factor: float = 0.
    use_decay: bool = False
    decay_rate: float = 0.
    decay_steps: int = 0.

    mode: str = 'bool'
    macro_idx: Union[str, List[int]] = None
    macro_feature_file_test: str = 'datasets/macro/macro_test.npz'
    macro_feature_file: str = 'datasets/macro/macro_train.npz'
    macro_feature_file_valid: str = 'datasets/macro/macro_valid.npz'
    individual_feature_file_valid: str = 'datasets/char/Char_valid.npz'
    individual_feature_file: str = 'datasets/char/Char_train.npz'
    individual_feature_file_test: str = 'datasets/char/Char_test.npz'
    tSize_test: int = 300
    macro_feature_dim: int = 178
    tSize: int = 240
    individual_feature_dim: int = 46
    tSize_valid: int = 60

    use_rnn: bool = False
    cell_type_rnn: str = 'lstm'
    num_units_rnn: List[int] = field(default_factory=list)
    hidden_dim: List[int] = field(default_factory=list)
    num_layers_rnn: int = 1
    num_layers: int = 2
    num_epochs: int = 1024
    num_epochs_unc: int = 256

    num_layers_moment: int = 0
    cell_type_rnn_moment: str = 'lstm'
    num_epochs_moment: int = 64
    num_condition_moment: int = 8
    num_layers_rnn_moment: int = 1
    num_units_rnn_moment: List[int] = field(default_factory=list)
    hidden_dim_moment: List[int] = field(default_factory=list)

    @classmethod
    def from_json(cls, path_to_json: str):
        with open(path_to_json, 'r') as f:
            json_file = json.load(f)
        return cls(**json_file)

    def __getitem__(self, item: str):
        return getattr(self, item)
