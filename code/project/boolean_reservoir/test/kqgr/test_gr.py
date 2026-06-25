import pytest
import numpy as np
from project.boolean_reservoir.code.encoding import BooleanTransformer
from project.boolean_reservoir.code.parameter import (
    Params, ModelParams, InputParams, ReservoirParams,
    OutputParams, TrainingParams
)

def create_test_params(tau, eval_mode, features=2, resolution=10):
    """Create minimal params for testing, with kqgr universe overrides."""
    return Params(
        model=ModelParams(
            input_layer=InputParams(features=features, resolution=resolution),
            reservoir_layer=ReservoirParams(n_nodes=10),
            output_layer=OutputParams(n_nodes=1),
            training=TrainingParams()
        ),
        multiverse_overrides={
            'kqgr': {
                'dataset': {'tau': tau, 'evaluation': eval_mode}
            }
        }
    )

def create_free_bit_data(m=10, s=1, f=2, free_b=7):
    """Create synthetic free-bit data (resolution - tau bits per sample)."""
    return np.random.randint(0, 2, (m, s, f, free_b), dtype=np.uint8)


@pytest.mark.parametrize("eval_mode", ['first', 'last', 'random'])
def test_apply_tau_output_shape(eval_mode):
    """_apply_tau(free_b_input) must return (m, s, f, free_b + tau) tensor."""
    tau = 3
    free_b = 7
    p = create_test_params(tau, eval_mode)
    transformer = BooleanTransformer(p.U.kqgr, apply_redundancy=False)

    x = create_free_bit_data(free_b=free_b)
    m, s, f, _ = x.shape
    out = transformer._apply_tau(x)

    assert out.shape == (m, s, f, free_b + tau), \
        f"Expected shape {(m, s, f, free_b + tau)}, got {out.shape}"


@pytest.mark.parametrize("eval_mode", ['first', 'last', 'random'])
def test_apply_tau_identical_bits_across_samples(eval_mode):
    """The tau constrained bits must be identical for all m samples."""
    tau = 3
    free_b = 7
    p = create_test_params(tau, eval_mode)
    transformer = BooleanTransformer(p.U.kqgr, apply_redundancy=False)

    x = create_free_bit_data(free_b=free_b)
    out = transformer._apply_tau(x)  # (m, s, f, 10)

    # Find positions where bits are identical across all samples
    # (identical = all samples have the same value at that position)
    identical_mask = (out == out[0:1]).all(axis=0)  # (s, f, b+tau)
    n_identical = identical_mask.sum().item()

    assert n_identical >= tau * x.shape[2], \
        f"Expected at least {tau * x.shape[2]} identical bit positions, got {n_identical}"


def test_apply_tau_last_positions_identical():
    """'last' mode: last tau bits of each feature must be identical across samples."""
    tau = 3
    free_b = 7
    p = create_test_params(tau, 'last')
    transformer = BooleanTransformer(p.U.kqgr, apply_redundancy=False)

    out = transformer._apply_tau(create_free_bit_data(free_b=free_b))  # (m, s, 2, 10)

    for feat in range(2):
        tail = out[:, :, feat, -tau:]  # (m, s, tau)
        assert (tail == tail[0:1]).all(), \
            f"Feature {feat}: last {tau} bits must be identical across samples"


def test_apply_tau_first_positions_identical():
    """'first' mode: first tau bits of each feature must be identical across samples."""
    tau = 3
    free_b = 7
    p = create_test_params(tau, 'first')
    transformer = BooleanTransformer(p.U.kqgr, apply_redundancy=False)

    out = transformer._apply_tau(create_free_bit_data(free_b=free_b))

    for feat in range(2):
        head = out[:, :, feat, :tau]
        assert (head == head[0:1]).all(), \
            f"Feature {feat}: first {tau} bits must be identical across samples"


def test_tau_per_feature():
    """'last' mode: tau applies per feature — each feature's last tau bits identical."""
    tau = 4
    free_b = 6
    p = create_test_params(tau, 'last', features=3)
    transformer = BooleanTransformer(p.U.kqgr, apply_redundancy=False)

    out = transformer._apply_tau(create_free_bit_data(f=3, free_b=free_b))

    for feat in range(3):
        tail = out[:, :, feat, -tau:]
        assert (tail == tail[0:1]).all(), \
            f"Feature {feat}: last {tau} bits should be identical"


def test_tau_zero_is_noop():
    """tau=0 must return x unchanged."""
    p = create_test_params(tau=0, eval_mode='last')
    transformer = BooleanTransformer(p.U.kqgr, apply_redundancy=False)

    x = create_free_bit_data(free_b=10)
    out = transformer._apply_tau(x.copy())

    assert (out == x).all(), "tau=0 should not modify any bits"


def test_apply_tau_free_bits_unique():
    """All m samples must have distinct full output vectors (unique free bits guaranteed)."""
    tau = 3
    free_b = 7
    m = 10
    p = create_test_params(tau, 'last')
    transformer = BooleanTransformer(p.U.kqgr, apply_redundancy=False)

    x = create_free_bit_data(m=m, free_b=free_b)
    # Ensure all free-bit inputs are distinct (test precondition)
    x_flat = x.reshape(m, -1)
    if len(np.unique(x_flat, axis=0)) < m:
        pytest.skip("Random free-bit data happened to collide; rerun")

    out = transformer._apply_tau(x)
    out_flat = out.reshape(m, -1)
    assert len(np.unique(out_flat, axis=0)) == m, \
        "Distinct free inputs must produce distinct full output vectors"
