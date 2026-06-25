import numpy as np
from project.boolean_reservoir.code.parameter import Params, InputParams
from project.boolean_reservoir.code.utils.primes import PRIMES


class BooleanTransformer:
    def __init__(self, P: Params, apply_redundancy=True):
        self.P = P
        self.I: InputParams = P.M.I
        self.redundancy = self.I.redundancy
        self.interleaving = self.I.interleaving
        self.apply_redundancy = apply_redundancy
        self._tau_gen = self._make_tau_generator(P)

        if self.I.encoding == 'binary_embedding':
            self.binary_encoder = BinaryEmbedding(b=self.I.resolution, n=self.I.redundancy)
        else:
            self.binary_encoder = None

    @staticmethod
    def _make_tau_generator(P):
        try:
            seed = P.D.seed
            if seed is not None:
                return np.random.default_rng(int(seed))
        except AttributeError:
            pass
        return None

    def __call__(self, bin_values):
        if self._has_tau():
            bin_values = self._apply_tau(bin_values)

        if self.apply_redundancy and self.redundancy > 1:
            perm_str = self.I.permute
            if perm_str and not isinstance(perm_str, list):
                m, s, f, b = bin_values.shape
                K = self.redundancy
                perm = [int(c) for c in perm_str]
                expanded = np.broadcast_to(bin_values[:, :, :, np.newaxis, :], (m, s, f, K, b)).copy()
                full_perm = [0, 1] + [2 + p for p in perm]
                bin_values = np.transpose(expanded, full_perm).reshape(m, s, f, -1)
            else:
                bin_values = np.repeat(bin_values, self.redundancy, axis=-1)
        elif self.binary_encoder is not None:
            bin_values = self.binary_encoder.encode_boolean(bin_values)

        if self.interleaving:
            bin_values = interleave_features(bin_values, group_size=self.interleaving)

        return bin_values.astype(np.uint8)

    def _has_tau(self):
        try:
            return self.P.D.tau > 0
        except AttributeError:
            return False

    def _apply_tau(self, x):
        kqgr = self.P.D
        if kqgr.tau == 0:
            return x
        if kqgr.tau_axis == 'steps':
            if x.shape[1] == 1:
                m, _, f, b = x.shape
                result = self._apply_tau_steps(x.reshape(m, b, f, 1), kqgr)
                return result.reshape(m, 1, f, -1)
            return self._apply_tau_steps(x, kqgr)
        return self._apply_tau_bits(x, kqgr)

    def _apply_tau_bits(self, x, kqgr):
        m, s, f, b = x.shape
        tau = kqgr.tau
        total_b = b + tau
        max_distinct = 2 ** (s * f * b)
        if m > max_distinct:
            import warnings
            warnings.warn(
                f"tau={tau}: only {max_distinct} distinct {s*f*b}-bit free patterns available "
                f"for {m} samples — inputs will repeat. Reduce tau or increase resolution.",
                UserWarning, stacklevel=3,
            )

        rng = self._tau_gen if self._tau_gen is not None else np.random.default_rng()
        ref_idx = int(rng.integers(0, m))
        ref_tau = np.zeros((m, s, f, tau), dtype=x.dtype)
        ref_tau[:] = x[ref_idx:ref_idx+1, :, :, :min(tau, b)]

        if kqgr.evaluation_mode == 'last':
            return np.concatenate([x, ref_tau], axis=-1)

        elif kqgr.evaluation_mode == 'first':
            return np.concatenate([ref_tau, x], axis=-1)

        elif kqgr.evaluation_mode == 'random':
            out = np.empty((m, s, f, total_b), dtype=x.dtype)
            perm = np.array([rng.permutation(total_b)[:tau] for _ in range(f)])  # (f, tau)
            free_pos = np.array([
                [i for i in range(total_b) if i not in perm[fi]] for fi in range(f)
            ])  # (f, b)
            for fi in range(f):
                out[:, :, fi, :][:, :, perm[fi]]    = ref_tau[:, :, fi, :]
                out[:, :, fi, :][:, :, free_pos[fi]] = x[:, :, fi, :]
            return out

        return x

    def _apply_tau_steps(self, x, kqgr):
        m, s, f, b = x.shape
        tau = kqgr.tau
        total_s = s + tau
        max_distinct = 2 ** (s * f * b)
        if m > max_distinct:
            import warnings
            warnings.warn(
                f"tau={tau}: only {max_distinct} distinct {s*f*b}-bit free patterns available "
                f"for {m} samples — inputs will repeat.",
                UserWarning, stacklevel=3,
            )

        rng = self._tau_gen if self._tau_gen is not None else np.random.default_rng()
        ref_idx = int(rng.integers(0, m))
        ref_step = x[ref_idx:ref_idx+1, :1, :, :]          # (1, 1, f, b)
        ref_steps = np.broadcast_to(ref_step, (m, tau, f, b)).copy()  # (m, tau, f, b)

        if kqgr.evaluation_mode == 'last':
            return np.concatenate([x, ref_steps], axis=1)

        elif kqgr.evaluation_mode == 'first':
            return np.concatenate([ref_steps, x], axis=1)

        elif kqgr.evaluation_mode == 'random':
            out = np.empty((m, total_s, f, b), dtype=x.dtype)
            tau_pos = rng.permutation(total_s)[:tau]
            free_pos = np.array([i for i in range(total_s) if i not in tau_pos])
            out[:, tau_pos, :, :]  = ref_steps
            out[:, free_pos, :, :] = x
            return out

        return x


class BooleanEncoder:
    def __init__(self, P):
        self.P = P
        self.I: InputParams = P.M.I
        apply_redundancy = (self.I.encoding != 'binary_embedding')
        self.transformer = BooleanTransformer(P, apply_redundancy=apply_redundancy)

    def __call__(self, x):
        bin_values = self._encode(x)
        return self.transformer(bin_values)

    def _encode(self, values):
        assert np.issubdtype(values.dtype, np.floating), (
            f"BooleanEncoder expects float input (normalized [0,1]). "
            f"Got dtype={values.dtype}. For boolean/integer inputs use BooleanTransformer directly."
        )
        assert values.max() <= 1
        assert values.min() >= 0
        fn = _ENCODERS.get(self.I.encoding)
        if fn is None:
            raise ValueError(f"encoding {self.I.encoding!r} is not an option! Valid: {list(_ENCODERS)}")
        return fn(values, self.I.resolution)


def dec2bin(x, bits):
    x_int = (x * (2**bits - 1)).astype(np.int64)
    mask = (1 << np.arange(bits - 1, -1, -1, dtype=np.int64))
    return (x_int[..., None] & mask) != 0


def dec2tally(x, bits):
    d = (bits * x).round()[..., None]
    bit_range = np.arange(bits, dtype=np.float32)
    return bit_range < d


def dec2gray(x, bits):
    x_int = (x * (2**bits - 1)).astype(np.int64)
    gray = x_int ^ (x_int >> 1)
    mask = (1 << np.arange(bits - 1, -1, -1, dtype=np.int64))
    return (gray[..., None] & mask) != 0


# NOTE: dec2rate produces a different binary pattern on every call for the same
# input value (inherently stochastic). With n_samples=1 and single-shot ridge
# regression, this adds irreducible noise — the readout sees inconsistent
# representations of identical inputs. Rate coding only works reliably when
# averaged over many trials (as in biological rate-coding theory). Needs either
# a seeded generator per sample or multi-trial averaging before use in grid search.
def dec2rate(x, bits):
    p = x[..., None] * np.ones(bits, dtype=np.float32)
    return np.random.random(p.shape) < p


def dec2delta_sigma(x, bits):
    result = np.zeros(x.shape + (bits,), dtype=bool)
    integrator = np.zeros_like(x, dtype=np.float32)
    for i in range(bits):
        integrator = integrator + x
        bit = integrator >= 0.5
        result[..., i] = bit
        integrator = integrator - bit.astype(np.float32)
    return result


def load_primes(n):
    if n > 1000:
        raise ValueError(f"Cannot load more than 1000 primes, got {n}")
    return PRIMES[:n]


def dec2primes(x, bits):
    primes = load_primes(bits - 1)
    weights = np.array(primes[::-1] + [1], dtype=np.int64)
    max_val = weights.sum()
    x = (x * max_val).round().astype(np.int64)
    result = np.zeros(x.shape + (bits,), dtype=bool)
    for i in range(bits):
        result[..., i] = x >= weights[i]
        x = x - result[..., i].astype(np.int64) * weights[i]
    return result


_ENCODERS: dict[str, callable] = {
    'base2':            dec2bin,
    'primes':           dec2primes,
    'tally':            dec2tally,
    'gray':             dec2gray,
    'rate':             dec2rate,
    'delta_sigma':      dec2delta_sigma,
    'binary_embedding': dec2bin,
}


def bin2dec(x, bits, small_endian=False):
    if small_endian:
        mask = (1 << np.arange(bits, dtype=np.int64))
    else:
        mask = (1 << np.arange(bits - 1, -1, -1, dtype=np.int64))
    return (mask * x).sum(axis=-1) / (2**bits - 1)


def interleave_features(x, group_size=1):
    shape = x.shape
    n, m, d, b = shape
    gs = group_size
    num_full = b // gs
    rem = b % gs
    parts = []
    for i in range(num_full):
        for j in range(d):
            parts.append(x[:, :, j:j+1, i*gs:(i+1)*gs])
    if rem > 0:
        for j in range(d):
            parts.append(x[:, :, j:j+1, -rem:])
    return np.concatenate(parts, axis=-1).reshape(shape)


class MinMaxNormalization:
    def __call__(self, data):
        data = data.astype(np.float32)
        self.min_ = data.min(axis=(0, 1), keepdims=True)
        self.max_ = data.max(axis=(0, 1), keepdims=True)
        denom = self.max_ - self.min_
        denom = np.where(denom == 0, 1.0, denom)
        return (data - self.min_) / denom

    def inverse(self, data):
        return data * (self.max_ - self.min_) + self.min_


class StandardScaler:
    def __call__(self, data):
        data = data.astype(np.float32)
        self.mean_ = data.mean(axis=(0, 1), keepdims=True)
        self.std_  = data.std(axis=(0, 1), keepdims=True)
        self.std_  = np.maximum(self.std_, 1e-8)
        return (data - self.mean_) / self.std_

    def inverse(self, data):
        return data * self.std_ + self.mean_


class BinaryEmbedding:
    def __init__(self, b, n):
        self.b = None
        self.n = None
        self.random_boolean_keys = None
        self.set_random_boolean_keys(b, n)

    def _encode_bits(self, bits: np.ndarray):
        batch_size, seq_length, features, b = bits.shape
        assert b == self.b, "Input bit resolution does not match encoder configuration."
        bits_expanded = bits[:, :, :, np.newaxis, :]          # (B, T, F, 1, b)
        encoded = bits_expanded ^ self.random_boolean_keys     # broadcast to (B, T, F, n, b)
        return encoded.reshape(batch_size, seq_length, features, -1)

    def encode_float(self, data: np.ndarray):
        bits = dec2bin(data, self.b).astype(np.uint8)
        return self._encode_bits(bits)

    def encode_boolean(self, data: np.ndarray):
        return self._encode_bits(data)

    def set_random_boolean_keys(self, new_b=None, new_n=None):
        self.n = new_n if new_n is not None else self.n
        self.b = new_b if new_b is not None else self.b
        self.random_boolean_keys = np.random.randint(
            0, 2, (1, 1, 1, self.n, self.b), dtype=np.uint8
        )
