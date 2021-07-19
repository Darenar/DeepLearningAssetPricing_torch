from typing import Union, Optional, List, Any, Dict
import os

import numpy as np
from pathlib import Path
import torch
import torch.utils.data

from .config_reader import Config
from .utils import try_to_cuda
from .constants import FIRM_CHARS_DICT


PATH_TYPE = Union[Path, str]
UNKNOWN_VAL = -99.99


def to_tensor(input_array: np.ndarray, tensor_type: torch.dtype = torch.float32) -> torch.Tensor:
    return torch.as_tensor(input_array, dtype=tensor_type)


class BaseMapping:
    def __init__(self, list_of_values: List[Any]):
        self.list_of_values = list_of_values
        self.__idx2var = None
        self.__var2idx = None

    def filter(self, filter_indexes: np.ndarray):
        self.list_of_values = self.list_of_values[filter_indexes]
        self.__idx2var = None
        self.__var2idx = None

    @property
    def idx2var(self):
        if self.__idx2var is None:
            self.__idx2var = {idx: var for idx, var in enumerate(self.list_of_values)}
        return self.__idx2var

    @property
    def var2idx(self):
        if self.__var2idx is None:
            self.__var2idx = {var: idx for idx, var in enumerate(self.list_of_values)}
        return self.__var2idx


class StoredData:
    def __init__(self, data: np.ndarray, date_list: List[str],
                 variable_names: List[str], return_feature_num: int):
        self.__data = data
        self.__return_feature_num = return_feature_num
        self.mask = (self.return_array != UNKNOWN_VAL)
        self.dates = BaseMapping(date_list)
        self.variables = BaseMapping(variable_names)

    @property
    def return_array(self):
        return self.__data[:, :, self.__return_feature_num]

    @property
    def features(self):
        return self.__data[:, :, self.__return_feature_num+1:]

    def multiply_returns_on_sdf(self, sdf: np.ndarray, factor: int):
        multiplied_sdf = np.tile(sdf, self.mask.shape[1]) * factor
        self.__data[:, :, self.__return_feature_num][self.mask] *= multiplied_sdf[self.mask]

    def save_data_to_numpy(self, path_to_file: str):
        np.savez(path_to_file,
                 date=self.dates.list_of_values,
                 variable=self.variables.list_of_values,
                 permno=self.permno_count,
                 data=self.__data)

    @classmethod
    def from_numpy_file(cls, path_to_numpy_file: PATH_TYPE) -> 'StoredData':
        if not os.path.exists(str(path_to_numpy_file)):
            raise ValueError(f'No such file {path_to_numpy_file}')
        dataset = np.load(path_to_numpy_file)
        return cls(
            data=dataset['data'],
            date_list=dataset['date'],
            variable_names=dataset['variable'],
            return_feature_num=0,
        )

    @property
    def date_count(self) -> int:
        return self.__data.shape[0]

    @property
    def permno_count(self) -> int:
        return self.__data.shape[1]

    @property
    def var_count(self) -> int:
        return self.__data.shape[2] - 1


class MacroDataset:
    def __init__(self, macro_features_data: np.ndarray, variable_names: List[str]):
        self.features = macro_features_data
        self.variables = BaseMapping(variable_names)
        self.mean_features = None
        self.std_features = None

    def filter(self, macro_idx: List[str]):
        if isinstance(macro_idx, list):
            macro_idx = np.sort(np.array(macro_idx, dtype=int))
        elif macro_idx == '178':
            macro_idx = np.sort(np.concatenate((np.arange(124), np.arange(284, 338))))
        else:
            return
        self.features = self.features[:, macro_idx]
        self.variables.filter(macro_idx)

    @classmethod
    def from_numpy_file(cls, path_to_numpy_file: PATH_TYPE) -> 'MacroDataset':
        if not os.path.exists(str(path_to_numpy_file)):
            raise ValueError(f'No such file {path_to_numpy_file}')
        dataset = np.load(path_to_numpy_file)
        return cls(macro_features_data=dataset['data'],
                   variable_names=dataset['variable'])

    def normalize(self, macro_feature_mean_val: np.ndarray,
                  macro_feature_std_val: np.ndarray):
        if macro_feature_mean_val is None or macro_feature_std_val is None:
            self.mean_features = self.features.mean(axis=0)
            self.std_features = self.features.std(axis=0)
        else:
            self.mean_features = macro_feature_mean_val
            self.std_features = macro_feature_std_val
        self.features -= self.mean_features
        self.features /= self.std_features

    @property
    def var_count(self) -> int:
        return self.features.shape[1]


class BaseCategory:
    def __init__(self, category_name: str, variables: List[str], color: str):
        self.name = category_name
        self.variables = variables
        self.color = color

    def get_variable_to_color_map(self) -> Dict[str, str]:
        return {var: self.color for var in self.variables}


class FirmChar:
    def __init__(self, categories_list: List[dict]):
        self.categories = list()
        for cat_dict in categories_list:
            self.categories.append(BaseCategory(
                category_name=cat_dict['name'],
                variables=cat_dict['variables'],
                color=cat_dict['color']
            ))

    @classmethod
    def from_dict(cls, firm_chars_dict: dict):
        return cls(firm_chars_dict['categories'])

    def get_color_label_map(self):
        variables_to_color_map = dict()
        for cat in self.categories:
            variables_to_color_map.update(cat.get_variable_to_color_map())
        return variables_to_color_map


class FinanceDataset:
    def __init__(self, individual_data: np.ndarray, individual_date_list: List[str],
                 individual_variable_names: List[str], return_feature_num: int,
                 macro_features_data: np.ndarray, macro_variable_names: List[str],
                 macro_indices: Optional[Union[str, List[str]]] = None, *args, **kwargs):
        self.individual_data = StoredData(
            data=individual_data,
            date_list=individual_date_list,
            variable_names=individual_variable_names,
            return_feature_num=return_feature_num,
        )
        self.macro_data = MacroDataset(
            macro_features_data=macro_features_data,
            variable_names=macro_variable_names
        )
        self.macro_data.filter(macro_indices)
        self.firm_char = FirmChar.from_dict(FIRM_CHARS_DICT)

    def normalize_macro_features(self, macro_feature_mean_val: np.ndarray = None,
                                 macro_feature_std_val: np.ndarray = None):
        self.macro_data.normalize(macro_feature_mean_val, macro_feature_std_val)

    @classmethod
    def from_config(cls, config: Config, suffix: str = '',
                    macro_feature_mean_val: np.ndarray = None,
                    macro_feature_std_val: np.ndarray = None):
        if suffix:
            suffix = f'_{suffix}'

        individual_dataset = np.load(config[f'individual_feature_file{suffix}'])
        macro_dataset = np.load(config[f'macro_feature_file{suffix}'])
        dataset = cls(
            individual_data=individual_dataset['data'],
            individual_date_list=individual_dataset['date'],
            individual_variable_names=individual_dataset['variable'],
            return_feature_num=0,
            macro_features_data=macro_dataset['data'],
            macro_variable_names=macro_dataset['variable'],
            macro_indices=config['macro_idx']
        )
        dataset.normalize_macro_features(macro_feature_mean_val, macro_feature_std_val)
        return dataset

    @property
    def date_count_list(self):
        return np.sum(self.individual_data.mask, axis=0)

    @property
    def macro_features_mean_std(self):
        return self.macro_data.mean_features, self.macro_data.std_features

    def get_individual_feature_by_idx(self, idx: int):
        return self.individual_data.variables.idx2var[idx]

    def get_sequence_length(self):
        return self.individual_data.features.shape[0]

    def save_individual_data_to_numpy(self, path_to_file: str):
        self.individual_data.save_data_to_numpy(path_to_file)

    @property
    @try_to_cuda
    def macro_feat_tensor(self):
        return torch.unsqueeze(to_tensor(self.macro_data.features), 0)

    @property
    @try_to_cuda
    def ind_feat_tensor(self):
        return torch.unsqueeze(to_tensor(self.individual_data.features), 0)

    @property
    @try_to_cuda
    def returns_tensor(self):
        return torch.unsqueeze(
            torch.unsqueeze(
                to_tensor(self.individual_data.return_array), 0), -1)

    @property
    @try_to_cuda
    def masks_tensor(self):
        return torch.unsqueeze(to_tensor(self.individual_data.mask, tensor_type=torch.bool), -1)