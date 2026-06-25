import pytest
from benchmark.temporal.temporal_density_parity_dataset import (
    TemporalDatasetParams,
    TemporalDataset,
)
from project.boolean_reservoir.code.utils.utils import set_seed
import numpy as np

def color_the_stream(bits, window, delay):
    # Prepare the stream for printing with the window in red
    start_index = len(bits) - window - delay
    end_index = len(bits) - delay
    colored_stream = ""
    for i in range(len(bits)):
        if start_index <= i < end_index:
            colored_stream += f"\033[91m{bits[i].astype(np.uint8)}\033[0m"  # Red color
        else:
            colored_stream += f"{bits[i].astype(np.uint8)}"
    return colored_stream

def format_task_result(label, task_name):
    """Format a single task result as human-readable text."""
    if task_name.lower() == "density":
        return "more 1's than 0's" if label else "more 0's than 1's"
    elif task_name.lower() == "parity":
        return "odd number of 1's" if label else "even number of 1's"
    else:
        return str(label)

def verify_sample(bits, window, delay, task_func, expected_label):
    """Verify that a bit stream produces the expected label."""
    computed_label = task_func(bits, window, delay)
    return bool(computed_label) == bool(expected_label)

@pytest.fixture
def density_dataset_1d():
    set_seed(0)
    D = TemporalDatasetParams(
        path="/tmp/test_density_1d",
        task="density",
        bits=10,
        window=5,
        delay=2,
        dimensions=1,
        samples=10,
        sampling_mode='random',
        generate_data=True,
    )
    return TemporalDataset(D)


@pytest.fixture
def parity_dataset_1d():
    set_seed(0)
    D = TemporalDatasetParams(
        path="/tmp/test_parity_1d",
        task="parity",
        bits=10,
        window=5,
        delay=2,
        dimensions=1,
        samples=10,
        sampling_mode='random',
        generate_data=True,
    )
    return TemporalDataset(D)


@pytest.fixture
def density_dataset_2d():
    set_seed(0)
    D = TemporalDatasetParams(
        path="/tmp/test_density_2d",
        task="density",
        bits=8,
        window=3,
        delay=1,
        dimensions=3,
        samples=10,
        sampling_mode='random',
        generate_data=True,
    )
    return TemporalDataset(D)


def test_dataset_shapes_1d(density_dataset_1d):
    """Test that 1D dataset has correct shapes."""
    dataset = density_dataset_1d
    assert dataset.x.shape == (10, 1, 1, 10)  # samples, 1, dimensions, bits
    assert dataset.y.shape == (10, 1)  # samples, dimensions


def test_dataset_shapes_2d(density_dataset_2d):
    """Test that multi-D dataset has correct shapes."""
    dataset = density_dataset_2d
    assert dataset.x.shape == (10, 1, 3, 8)  # samples, 1, dimensions, bits
    assert dataset.y.shape == (10, 3)  # samples, dimensions


def test_density_labels_correct(density_dataset_1d):
    """Verify all density labels are computed correctly."""
    dataset = density_dataset_1d
    
    for i in range(len(dataset.x)):
        sample_x = dataset.x[i, 0]  # d x b
        sample_y = dataset.y[i]      # d
        
        for d in range(dataset.D.dimensions):
            bits = sample_x[d]
            label = sample_y[d].item()
            
            assert verify_sample(
                bits, 
                dataset.D.window, 
                dataset.D.delay, 
                dataset.density_task,
                label
            ), f"Sample {i}, dimension {d} has incorrect label"


def test_parity_labels_correct(parity_dataset_1d):
    """Verify all parity labels are computed correctly."""
    dataset = parity_dataset_1d
    
    for i in range(len(dataset.x)):
        sample_x = dataset.x[i, 0]  # d x b
        sample_y = dataset.y[i]      # d
        
        for d in range(dataset.D.dimensions):
            bits = sample_x[d]
            label = sample_y[d].item()
            
            assert verify_sample(
                bits,
                dataset.D.window,
                dataset.D.delay,
                dataset.parity_task,
                label
            ), f"Sample {i}, dimension {d} has incorrect label"


def test_density_labels_correct_multid(density_dataset_2d):
    """Verify all density labels are correct in multi-dimensional dataset."""
    dataset = density_dataset_2d
    
    for i in range(len(dataset.x)):
        sample_x = dataset.x[i, 0]  # d x b
        sample_y = dataset.y[i]      # d
        
        for d in range(dataset.D.dimensions):
            bits = sample_x[d]
            label = sample_y[d].item()
            
            assert verify_sample(
                bits,
                dataset.D.window,
                dataset.D.delay,
                dataset.density_task,
                label
            ), f"Sample {i}, dimension {d} has incorrect label"


def test_exhaustive_mode():
    """Test exhaustive sampling mode generates unique patterns."""
    set_seed(0)
    D = TemporalDatasetParams(
        path="/tmp/test_exhaustive",
        task="density",
        bits=4,  # 2^4 = 16 possible patterns
        window=2,
        delay=1,
        dimensions=1,
        samples=16,
        sampling_mode='exhaustive',
        generate_data=True,
    )
    dataset = TemporalDataset(D)
    
    # Check we have all unique patterns
    patterns = dataset.x[:, 0, 0, :]  # samples x bits
    unique_patterns = set(tuple(p) for p in patterns)
    assert len(unique_patterns) == 16, "Exhaustive mode should generate all unique patterns"


def test_exhaustive_mode_with_repetition():
    """Test exhaustive mode cycles when samples > 2^bits.

    shuffle=False is required: the cycling-order assertions (sample[i] == sample[i+8])
    only hold when dataset-level shuffling is disabled.  With shuffle=True the full
    dataset is globally randomised by shuffle_data(), which destroys cycle boundaries.
    """
    set_seed(0)
    D = TemporalDatasetParams(
        path="/tmp/test_exhaustive_repeat",
        task="density",
        bits=3,  # 2^3 = 8 possible patterns
        window=2,
        delay=0,
        dimensions=1,
        samples=20,  # More than 8
        sampling_mode='exhaustive',
        shuffle=False,          # preserve cycle order for the assertions below
        generate_data=True,
    )
    dataset = TemporalDataset(D)
    
    # First 8 should be unique
    patterns = dataset.x[:8, 0, 0, :]
    unique_patterns = set(tuple(p) for p in patterns)
    assert len(unique_patterns) == 8
    
    # Pattern 0 should equal pattern 8, pattern 1 should equal pattern 9, etc.
    for i in range(8):
        assert tuple(dataset.x[i, 0, 0, :]) == tuple(dataset.x[i+8, 0, 0, :])