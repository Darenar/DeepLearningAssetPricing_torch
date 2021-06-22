from typing import Union, Optional, List, Any
import os
import json

import numpy as np
from pathlib import Path
import torch
import torch.utils.data


PATH_TYPE = Union[Path, str]
UNKNOWN_VAL = -99.99


class BaseMapping:
    def __init__(self, list_of_values: List[Any]):
        self.__list_of_values = list_of_values
        self.__idx2var = None
        self.__var2idx = None

    def filter(self, filter_indexes: np.ndarray):
        self.__list_of_values = self.__list_of_values[filter_indexes]
        self.__idx2var = None
        self.__var2idx = None

    @property
    def idx2var(self):
        if self.__idx2var is None:
            self.__idx2var = {idx: var for idx, var in enumerate(self.__list_of_values)}
        return self.__idx2var

    @property
    def var2idx(self):
        if self.__var2idx is None:
            self.__var2idx = {var: idx for idx, var in enumerate(self.__list_of_values)}
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

    @classmethod
    def from_numpy_file(cls, path_to_numpy_file: PATH_TYPE):
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
    def from_numpy_file(cls, path_to_numpy_file: PATH_TYPE):
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

    def get_variable_to_color_map(self):
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
    def from_json_file(cls, path_to_json: str):
        with open(path_to_json, 'r') as f:
            firm_chars_json = json.load(f)
        return cls(firm_chars_json['categories'])

    def get_color_label_map(self):
        variables_to_color_map = dict()
        for cat in self.categories:
            variables_to_color_map.update(cat.get_variable_to_color_map())
        return variables_to_color_map


class FinanceDataset(torch.utils.data.Dataset):
    def __init__(self, path_to_individual_features: PATH_TYPE, path_to_macro_features: Optional[PATH_TYPE] = None,
                 macro_indices: Optional[Union[str, List[str]]] = None):
        self.individual_data = StoredData.from_numpy_file(path_to_individual_features)
        self.macro_data = MacroDataset.from_numpy_file(path_to_macro_features)
        self.macro_data.filter(macro_indices)
        self.firm_char = None

    def normalize_macro_features(self, macro_feature_mean_val: np.ndarray = None,
                                 macro_feature_std_val: np.ndarray = None):
        self.macro_data.normalize(macro_feature_mean_val, macro_feature_std_val)

    def add_firm_char(self, path_to_firm_char_json: str):
        self.firm_char = FirmChar.from_json_file(path_to_firm_char_json)

    @classmethod
    def from_config(cls, config: dict, path_to_firm_chars: str, suffix: str = '',
                    macro_feature_mean_val: np.ndarray = None, macro_feature_std_val: np.ndarray = None):
        if suffix:
            suffix = f'_{suffix}'
        dataset = cls(
            path_to_individual_features=config[f'individual_feature_file{suffix}'],
            path_to_macro_features=config[f'macro_feature_file{suffix}'],
            macro_indices=config['macro_idx']
        )
        dataset.add_firm_char(path_to_firm_chars)
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

    def iterate_one_epoch(self, sub_epoch: int = 1):
        for _ in range(sub_epoch):
            yield self.macro_data.features, self.individual_data.features, \
                  self.individual_data.return_array, self.individual_data.mask