from abc import ABC, abstractmethod
import copy
import numpy as np
import psutil
from project.boolean_reservoir.code.utils.utils import set_seed
from project.boolean_reservoir.code.reservoir import BooleanReservoir
from project.boolean_reservoir.code.parameter import *
from project.boolean_reservoir.code.visualization import *
from benchmark.utils.parameter import KQGRDatasetParams

_KQGR_CACHE_EXCLUDE = {'evaluation'}


# ---------------------------------------------------------------------------
# Accuracy functions
# ---------------------------------------------------------------------------

class AccuracyFunction(ABC):
    @abstractmethod
    def accuracy(self, y_hat, y, threshold, normalize=True):
        pass


class EuclideanDistanceAccuracy(AccuracyFunction):
    def accuracy(self, y_hat, y, threshold, normalize=True):
        distances = np.sqrt(np.sum((y_hat - y) ** 2, axis=1))
        correct = (distances < threshold).sum()
        return correct / len(y) if normalize else int(correct)


class BooleanAccuracy(AccuracyFunction):
    def accuracy(self, y_hat, y, threshold, normalize=True):
        correct = ((y_hat > threshold) == y.astype(bool)).sum()
        return correct / len(y) if normalize else int(correct)


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------

def _mse_loss(y_hat, y):
    return float(np.mean((y_hat - y) ** 2))


def _bce_loss(y_hat, y):
    eps = 1e-7
    yh = np.clip(y_hat, eps, 1 - eps)
    return float(-np.mean(y * np.log(yh) + (1 - y) * np.log(1 - yh)))


def criterion_fn(strategy: str):
    return {'MSE': _mse_loss, 'BCE': _bce_loss}[strategy]


# ---------------------------------------------------------------------------
# Dataset cache (for KQ/GR)
# ---------------------------------------------------------------------------

class DatasetCache:
    """Memory-aware dataset cache with OS-style aging (exponential fading LRU+LFU)."""
    _kq_cache = {}
    _kq_cache_current_bytes = 0
    _kq_cache_max_bytes = 1 * 1024 * 1024 * 1024  # 1 GB per worker
    _ram_threshold_percent = 90.0
    _AGE_MSB = 1 << 15

    def _get_dataset_size_bytes(self, dataset):
        total = 0
        if hasattr(dataset, 'data') and isinstance(dataset.data, dict):
            for v in dataset.data.values():
                if isinstance(v, np.ndarray):
                    total += v.nbytes
        return total

    def _age_cache(self):
        for entry in self._kq_cache.values():
            entry[2] >>= 1

    def _evict_smart(self, needed_bytes):
        if psutil.virtual_memory().percent > self._ram_threshold_percent:
            self._kq_cache.clear()
            self._kq_cache_current_bytes = 0
            return
        while self._kq_cache_current_bytes + needed_bytes > self._kq_cache_max_bytes and self._kq_cache:
            target_key = min(self._kq_cache, key=lambda k: self._kq_cache[k][2])
            _, size, _ = self._kq_cache.pop(target_key)
            self._kq_cache_current_bytes -= size

    def _cache_dataset(self, cache_key: str, create_fn):
        if cache_key not in self._kq_cache:
            if psutil.virtual_memory().percent > 80.0:
                self._kq_cache.clear()
                self._kq_cache_current_bytes = 0
            dataset = create_fn()
            ds_size = self._get_dataset_size_bytes(dataset)
            self._age_cache()
            self._evict_smart(ds_size)
            self._kq_cache[cache_key] = [dataset, ds_size, self._AGE_MSB]
            self._kq_cache_current_bytes += ds_size
        else:
            self._kq_cache[cache_key][2] |= self._AGE_MSB
        return copy.deepcopy(self._kq_cache[cache_key][0])


class KQGRInit(DatasetCache):
    """KQGR capacity measurement (KQ / GR) built on DatasetCache."""

    def _get_kq_cached_dataset(self, P: Params, gr_tau: int = 0):
        exclude_fields = _KQGR_CACHE_EXCLUDE if isinstance(P.D, KQGRDatasetParams) else set()
        cache_key = str((
            self.__class__.__name__,
            P.D.model_dump(exclude=exclude_fields),
            P.M.I.model_dump(),
            gr_tau,
        ))

        def _build():
            P_raw = copy.deepcopy(P)
            P_raw.dataset.split.train = 1
            P_raw.dataset.split.dev = 0
            P_raw.dataset.split.test = 0
            if hasattr(P_raw.dataset, 'tau'):
                P_raw.dataset.tau = 0
            augment_extra = P.D.tau if P.D.tau_mode == 'augment' else 0
            if P.D.tau_axis == 'resolution':
                P_raw.dataset.samples = P.M.R.n_nodes
                P_raw.M.I.resolution = P.M.I.resolution + augment_extra - gr_tau
                if hasattr(P_raw.dataset, 'bits'):
                    P_raw.dataset.bits = P.M.I.features * P_raw.M.I.resolution
            elif P.D.tau_axis == 'steps' and getattr(P_raw.dataset, 'sampling_mode', '') == 'exhaustive':
                P_raw.dataset.bits = P.D.bits + augment_extra - gr_tau
                P_raw.dataset.samples = 2 ** P_raw.dataset.bits
            else:
                P_raw.dataset.samples = P.M.R.n_nodes
                if P.D.tau_axis == 'steps':
                    P_raw.dataset.bits = P.D.bits + augment_extra - gr_tau
            return self._create_raw_dataset(P_raw)

        return self._cache_dataset(cache_key, _build)

    def kqgr(self, P: Params, kq: bool):
        # Deepcopy and zero tau BEFORE building the dataset so _get_kq_cached_dataset
        # sees tau=0 for KQ — prevents double-counting in augment+resolution mode.
        P = copy.deepcopy(P)
        if kq and hasattr(P.dataset, 'tau'):
            P.dataset.tau = 0
        gr_tau = 0 if kq else (P.D.tau if hasattr(P.D, 'tau') else 0)
        dataset = self._get_kq_cached_dataset(P, gr_tau=gr_tau)
        dataset = self._process_dataset(dataset, P)
        n = P.M.R.n_nodes
        x = dataset.data['x']
        if x.shape[0] < n:
            import warnings
            warnings.warn(
                f"Only {x.shape[0]} distinct free-bit patterns available for {n} samples "
                f"— inputs will repeat. Reduce tau or increase bits.",
                UserWarning,
            )
        elif x.shape[0] > n:
            idx = np.random.permutation(x.shape[0])[:n]
            dataset.set_data({
                k: (v[idx] if isinstance(v, np.ndarray) and v.shape[0] == x.shape[0] else v)
                for k, v in dataset.data.items()
            })
        return dataset


class DatasetInit(ABC):
    @abstractmethod
    def _create_raw_dataset(self, P: Params): pass

    @abstractmethod
    def _process_dataset(self, dataset, P: Params): pass

    def train(self, P: Params):
        dataset = self._create_raw_dataset(P)
        if hasattr(P.dataset, 'tau') and P.dataset.tau != 0:
            P = copy.deepcopy(P)
            P.dataset.tau = 0
        return self._process_dataset(dataset, P)


# ---------------------------------------------------------------------------
# Ridge regression training
# ---------------------------------------------------------------------------

def _collect_ridge_matrices(model: BooleanReservoir, dataset, batch_size: int) -> tuple:
    """Accumulate X^T X and X^T y in one forward pass using model.get_states().

    For time-series mode, get_states() returns summed states over steps and the step
    count s; the bias column is set to s so bias scaling matches forward():
    output = sum_s(W @ o_s + b) = W @ X_eff + s*b, where X_eff = sum_s(o_s).
    """
    n_r  = model.R.n_nodes
    n_out = model.O.n_nodes
    XtX = np.zeros((n_r + 1, n_r + 1), dtype=np.float64)
    Xty = np.zeros((n_r + 1, n_out),   dtype=np.float64)

    model.eval()
    x_all = dataset.data['x']
    y_all = dataset.data['y']
    n = len(x_all)

    for start in range(0, n, batch_size):
        end     = min(start + batch_size, n)
        x_batch = x_all[start:end]
        y_batch = y_all[start:end]

        states, bias_scale = model.get_states(x_batch)         # (m, n_r), scalar
        m = len(states)
        bias_col = np.full((m, 1), float(bias_scale), dtype=np.float64)
        X_aug = np.concatenate([states.astype(np.float64), bias_col], axis=1)  # (m, n_r+1)
        XtX += X_aug.T @ X_aug
        Xty += X_aug.T @ y_batch.astype(np.float64)

    return XtX, Xty


def train_readout_ridge(model: BooleanReservoir, dataset, alpha: float, accuracy: AccuracyFunction) -> tuple:
    """Fit linear readout via closed-form Ridge regression, push weights to Julia engine."""
    T = model.P.M.T

    XtX, Xty = _collect_ridge_matrices(model, dataset, T.batch_size)
    n_r = model.R.n_nodes
    reg = alpha * np.eye(n_r + 1, dtype=np.float64)
    sol = np.linalg.solve(XtX + reg, Xty)           # (n_r+1, n_out)
    W = sol[:-1].T.astype(np.float32)               # (n_out, n_r)
    b = sol[-1].astype(np.float32)                  # (n_out,)
    model.set_readout_weights(W, b)

    model.eval()
    x_eval = dataset.data['x_' + T.evaluation]
    y_eval = dataset.data['y_' + T.evaluation]
    y_hat  = model(x_eval)
    loss   = criterion_fn(T.criterion)(y_hat, y_eval)
    acc    = accuracy(y_hat, y_eval, T.accuracy_threshold)

    best_stats = {'epoch': 1, 'accuracy': acc, 'loss': loss}
    model.P.L.train = TrainLog(**best_stats)
    return best_stats, model, []


def train_and_evaluate(model: BooleanReservoir, dataset, record_stats=False, verbose=False,
                       accuracy_fn: AccuracyFunction = None):
    if accuracy_fn is None:
        accuracy_fn = EuclideanDistanceAccuracy().accuracy
    T = model.P.M.T
    if T.optim.name == 'ridge':
        alpha = getattr(T.optim.params, 'alpha', 1e-3)
        return train_readout_ridge(model, dataset, alpha=alpha, accuracy=accuracy_fn)
    raise ValueError(
        f"Unsupported optimizer '{T.optim.name}' — only 'ridge' is supported in this branch. "
        "Online training was intentionally removed; see torch-removal branch notes."
    )


def train_single_model(yaml_or_checkpoint_path='', parameter_override: Params = None,
                       model=None, save_model=True):
    if model is None:
        load_path = yaml_or_checkpoint_path if yaml_or_checkpoint_path else None
        model = BooleanReservoir(params=parameter_override, load_path=load_path)
    P = model.P
    dataset = P.dataset_init_obj.train(P)
    _, model, train_history = train_and_evaluate(
        model, dataset, record_stats=True, verbose=True, accuracy_fn=P.accuracy_obj
    )
    if save_model:
        model.save()
    return P, model, dataset, train_history
