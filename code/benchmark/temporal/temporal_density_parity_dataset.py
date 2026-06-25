import numpy as np
from benchmark.temporal.parameter import TemporalDatasetParams
from project.boolean_reservoir.code.utils.utils import set_seed
from benchmark.utils.base_dataset import BaseDataset


class TemporalDatasetBase(BaseDataset):
    def __init__(self, D: TemporalDatasetParams, task):
        # bits i.e. 12: 101010111010
        # window (in the bit stream) i.e. 4: 10101011[1010]
        # delay (for the window) i.e. 1: 1010101[1101]0
        super().__init__(D)
        set_seed(D.seed)
        self.task = task

        if self.data_path.exists() and not D.generate_data:
            self.load_data()
        else:
            if D.bits < D.window + D.delay:
                print('Warning: bits < delay + window, overriding bits...')
                D.bits = D.window + D.delay
            raw_data = self.generate_data(D.samples, D.bits, D.delay, D.window)
            self.set_data(raw_data)
            self.save_data()
        if D.shuffle:
            self.shuffle_data()

    @staticmethod
    def gen_integer_samples(samples, stream_length, sampling_mode, shuffle=False):
        """Generate integer samples based on sampling mode."""
        if sampling_mode == 'exhaustive':
            max_possible = 2**stream_length
            if stream_length > 20:
                raise ValueError(
                    f"Exhaustive mode with stream_length={stream_length} would generate "
                    f"{max_possible} samples, which is too large. Consider random mode or smaller stream_length."
                )
            int_samples = np.arange(max_possible)
            if shuffle:
                np.random.shuffle(int_samples)
            if samples > len(int_samples):
                int_samples = np.tile(int_samples, samples // len(int_samples) + 1)
            int_samples = int_samples[:samples]
        else:  # 'random'
            if stream_length > 63:
                raise ValueError(
                    f"stream_length={stream_length} is too large for random integer generation. "
                    f"Maximum is 63 bits."
                )
            int_samples = np.random.randint(0, 2**stream_length, size=samples)
        return int_samples

    def generate_data(self, samples, stream_length, delay, window):
        """Generate dataset from integer samples."""
        total_streams = samples * self.D.dimensions
        int_samples = self.gen_integer_samples(total_streams, stream_length, self.D.sampling_mode, shuffle=self.D.shuffle)

        # Vectorised bit unpacking: (total_streams, stream_length), dtype bool
        mask = (1 << np.arange(stream_length - 1, -1, -1)).astype(np.int64)
        arrays = (int_samples[:, None] & mask[None, :]) != 0

        labels = [self.task(arr, window, delay) for arr in arrays]

        x = arrays.reshape(samples, self.D.dimensions, stream_length)
        y = np.array(labels, dtype=np.float32).reshape(samples, self.D.dimensions)
        x = x[:, np.newaxis]  # m x 1 x d x b
        return {'x': x, 'y': y}

    @staticmethod
    def gen_boolean_array(n):
        return np.random.randint(0, 2, size=n, dtype=bool)


class TemporalDensityDataset(TemporalDatasetBase):
    def __init__(self, p: TemporalDatasetParams):
        TemporalDatasetBase.__init__(self, p, self.density_task)

    @staticmethod
    def density_task(bits, window, delay):
        b = len(bits)
        if delay + window > b:
            raise ValueError("delay + window must be less than or equal to the bits in the input stream")
        window_bits = bits[b - window - delay:b - delay]
        count_ones = window_bits.sum()
        return 2 * count_ones > window


class TemporalParityDataset(TemporalDatasetBase):
    def __init__(self, p: TemporalDatasetParams):
        TemporalDatasetBase.__init__(self, p, self.parity_task)

    @staticmethod
    def parity_task(bits, window, delay):
        b = len(bits)
        if delay + window > b:
            raise ValueError("delay + window must be less than or equal to the bits in the input stream")
        window_bits = bits[b - window - delay:b - delay]
        count_ones = window_bits.sum()
        return count_ones % 2 != 0


class TemporalDataset:
    def __new__(cls, D: TemporalDatasetParams):
        assert isinstance(D, TemporalDatasetParams)
        if D.task == 'density':
            return TemporalDensityDataset(D)
        elif D.task == 'parity':
            return TemporalParityDataset(D)
        else:
            raise ValueError(f"Unknown task: {D.task}")
