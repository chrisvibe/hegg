import numpy as np
from benchmark.utils.parameter import DatasetParameters
from math import floor

_DATA_KEYS = ['x', 'y', 'x_dev', 'y_dev', 'x_test', 'y_test']


class BaseDataset:
    def __init__(self, D: DatasetParameters):
        self.D = D
        self.x = None
        self.y = None
        self.x_dev = None
        self.y_dev = None
        self.x_test = None
        self.y_test = None
        self.normalizer_x = None
        self.normalizer_y = None
        self.encoder_x = None

    @property
    def data_path(self):
        return self.D.path.with_suffix('.npz')

    @property
    def data(self):
        return {k: getattr(self, k, None) for k in _DATA_KEYS}

    def set_data(self, data_dict):
        for key, array in data_dict.items():
            setattr(self, key, array)

    def split_dataset(self, split=[0.8, 0.1, 0.1]):
        split_train = split[0] if self.D.split is None else self.D.split.train
        split_dev   = split[1] if self.D.split is None else self.D.split.dev
        split_test  = split[2] if self.D.split is None else self.D.split.test
        assert float(sum((split_train, split_dev, split_test))) == 1.0, "Split ratios must sum to 1."

        x, y = self.x, self.y
        n = x.shape[0]
        train_end = floor(split_train * n)
        dev_end   = floor((split_train + split_dev) * n)

        self.x      = x[:train_end]
        self.y      = y[:train_end]
        self.x_dev  = x[train_end:dev_end]
        self.y_dev  = y[train_end:dev_end]
        self.x_test = x[dev_end:]
        self.y_test = y[dev_end:]

    def save_data(self):
        self.data_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(self.data_path, **{k: v for k, v in self.data.items() if v is not None})

    def load_data(self):
        loaded = np.load(self.data_path)
        self.set_data(dict(loaded))

    def set_normalizer_x(self, normalizer_x):
        self.normalizer_x = normalizer_x

    def set_normalizer_y(self, normalizer_y):
        self.normalizer_y = normalizer_y

    def set_encoder_x(self, encoder_x):
        self.encoder_x = encoder_x

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

    def normalize(self):
        self.x = self.normalizer_x(self.x)
        self.y = self.normalizer_y(self.y)

    def inverse_normalize_x(self, x):
        return self.normalizer_x.inverse(x)

    def inverse_normalize_y(self, y):
        return self.normalizer_y.inverse(y)

    def shuffle_data(self):
        perm = np.random.permutation(self.x.shape[0])
        self.x = self.x[perm]
        self.y = self.y[perm]

    def encode_x(self):
        self.x = self.encoder_x(self.x).astype(np.uint8)
