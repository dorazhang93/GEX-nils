import dataclasses as dc
import os.path
import pickle
import typing as ty
import warnings
from pathlib import Path
import numpy as np
import torch
from sklearn.preprocessing import OneHotEncoder

import env, util
TRAIN = 'train'
VAL = 'val'
TEST = 'test'
PARTS = [TRAIN, VAL, TEST]

ArrayDict = ty.Dict[str, np.ndarray]


@dc.dataclass
class Dataset:
    test: ty.Optional[ArrayDict]
    val: ty.Optional[ArrayDict]
    train: ty.Optional[ArrayDict]
    info: ty.Optional[ArrayDict]
    folder: ty.Optional[Path]
    endpoint: str

    @classmethod
    def from_dir(cls, dir_: ty.Union[Path, str], ep_: str) -> 'Dataset':
        dir_ = Path(dir_)

        with open(dir_ / "build_X_Y.pickle", 'rb') as f:
            data = pickle.load(f)

        return Dataset(
            data['test'],
            data['val'],
            data['train'],
            data['info'],
            dir_,
            ep_
        )

    def task_type(self):
        y_info=self.info['Y']
        if self.endpoint in y_info['regression'].keys():
            self.info['task_type'] = util.REGRESSION
            self.info['task_idx'] = y_info['regression'][self.endpoint]
        elif self.endpoint in y_info['binary_cls']:
            self.info['task_type'] = util.BINCLASS
            self.info['task_idx'] = y_info['binary_cls'][self.endpoint]
        elif self.endpoint in y_info['multi_cls']:
            self.info['task_type'] = util.MULTICLASS
            self.info['task_idx'] = y_info['multi_cls'][self.endpoint]
        else:
            raise ValueError(f"Invalid task type")

    def ohe_cat(self):
        if self.test['C']['data'] is None:
            return
        ohe = OneHotEncoder(handle_unknown='ignore', sparse=False, dtype='float32')
        ohe.fit(self.train['C']['data'])
        self.test['C']['data']=ohe.transform(self.test['C']['data'])
        self.val['C']['data'] = ohe.transform(self.val['C']['data'])
        self.train['C']['data'] = ohe.transform(self.train['C']['data'])

    @property
    def is_binclass(self) -> bool:
        return self.info['task_type'] == util.BINCLASS

    @property
    def is_multiclass(self) -> bool:
        return self.info['task_type'] == util.MULTICLASS

    @property
    def is_regression(self) -> bool:
        return self.info['task_type'] == util.REGRESSION

    @property
    def n_num_features(self) -> int:
        return len(self.info['N'].keys())

    @property
    def n_cat_features(self) -> int:
        return len(self.info['C'].keys())

    @property
    def n_features(self) -> int:
        return self.n_num_features + self.n_cat_features

    def filter_train_nan(self):
        y_idx = self.info['task_idx']
        nan_mask = {'train': np.isnan(self.train['Y']['data'][:, y_idx]),}
        self.train = {k: {kk: vv[~nan_mask['train']] if vv is not None else None for kk, vv in v.items()} for k, v in self.train.items()}
        print(f"Filtered missing values for ground truth label--{self.endpoint} !! \n "
              f"Train| {nan_mask['train'].sum()},")

    def size(self, part: str) -> int:
        X = self.train if part=='train' else self.val if  part=='val' else self.test
        X = X['C']['data'] if X['C']['data'] is not None else X['N']['data']
        return len(X)


def to_tensors(data: ArrayDict) -> ty.Dict[str, torch.Tensor]:
    return {k: torch.as_tensor(v) if v is not None else None for k, v in data.items()}


