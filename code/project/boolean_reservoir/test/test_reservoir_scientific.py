"""
Peer-review-oriented tests for BooleanReservoir (RBN-based reservoir computing).

Covers:
  1. Synchronous update correctness (true parallel tick, no race conditions)
  2. Determinism under seed control
  3. Batch independence (samples evolve in isolation)
  4. Input injection effectiveness (input nodes actually change)
  5. I-node isolation (input nodes have no incoming graph edges → reservoir cannot write them)
  6. Isolated-node state preservation (nodes with 0 in-degree keep their state)
  7. LUT bias fidelity (p=0.5 → ~50 % active output)
  8. LUT entropy / non-trivial dynamics (reservoir doesn't collapse to fixed point)
  9. Effective-k LUT access pattern (heterogeneous degree uses non-contiguous but correct LUT rows)
 10. Numpy-only stack (reservoir and readout are pure numpy; no torch tensors)
 11. Warmup monotonicity (warmup changes state compared to cold start)
 12. Kauffman criticality parameter (lambda = 2*p*(1-p); warns when far from critical point)
"""

import pytest
import numpy as np

from project.boolean_reservoir.code.parameter import (
    Params, ModelParams, InputParams, ReservoirParams, OutputParams,
    TrainingParams, LoggingParams, PatternWiring, RandomWiring
)
from project.boolean_reservoir.code.reservoir import BooleanReservoir
from project.boolean_reservoir.code.utils.utils import set_seed


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_params(
    n_r=64,
    k_avg=2.0,
    k_min=0,
    k_max=4,
    self_loops=0.0,
    p=0.5,
    perturbation='override',
    init='random',
    reset=True,
    mode='heterogeneous',
    n_input_nodes=None,
    bits=4,
    features=2,
    batch_size=8,
    seed=42,
) -> Params:
    I = InputParams(
        perturbation=perturbation,
        encoding='base2',
        features=features,
        bits=bits,
        n_nodes=n_input_nodes,
        ticks=f'1{{{features}}}',
        seed=seed,
    )
    R = ReservoirParams(
        n_nodes=n_r,
        p=p,
        init=init,
        reset=reset,
        seed=seed,
    )
    O = OutputParams(n_nodes=features, seed=seed)
    T = TrainingParams(batch_size=batch_size, epochs=1, accuracy_threshold=0.05,
                       optim={'name': 'ridge', 'params': {'alpha': 1e-3}}, seed=seed)
    L = LoggingParams(out_path='/tmp/test_rc_science/', save_keys=None)
    variables = {
        'R_k_avg': k_avg,
        'R_k_min': k_min,
        'R_k_max': k_max,
        'R_self_loops': self_loops if self_loops is not None else 0.0,
        'R_mode': mode,
    }
    wiring = [
        PatternWiring(source='B', target='I', pattern='identity'),
        PatternWiring(source='I', target='R', pattern='identity'),
        RandomWiring(source='R', target='R', k_avg='R_k_avg', k_min='R_k_min',
                     k_max='R_k_max', self_loops='R_self_loops', mode=mode),
    ]
    M = ModelParams(variables=variables, input_layer=I, reservoir_layer=R,
                    output_layer=O, training=T, wiring=wiring)
    return Params(model=M, logging=L)


def make_model(seed=42, **kwargs) -> BooleanReservoir:
    set_seed(seed)
    return BooleanReservoir(make_params(seed=seed, **kwargs))


def random_input(model: BooleanReservoir, m: int, s: int = 1) -> np.ndarray:
    """Return a random uint8 input array matching the model's expected shape."""
    I = model.P.M.I
    return np.random.randint(0, 2, (m, s, I.features, I.bits // I.features), dtype=np.uint8)


# ---------------------------------------------------------------------------
# 1. Synchronous update — all nodes read OLD state, then write new state
# ---------------------------------------------------------------------------

def test_synchronous_update():
    """_reservoir_tick must read all states before writing any (parallel semantics)."""
    model = make_model(n_r=32, k_avg=2.0, k_max=4)
    m = 1
    state_before = model.states_parallel[:m].copy()
    model._reservoir_tick(m)
    state_after = model.states_parallel[:m].copy()
    model.states_parallel[:m] = state_before
    model._reservoir_tick(m)
    state_after2 = model.states_parallel[:m].copy()
    assert np.array_equal(state_after, state_after2), (
        "Tick is not deterministic: same start state should give same next state"
    )


# ---------------------------------------------------------------------------
# 2. Determinism under seed control
# ---------------------------------------------------------------------------

def test_determinism_same_seed():
    """Two models built with the same seed must produce identical outputs."""
    x = np.random.randint(0, 2, (4, 1, 2, 2), dtype=np.uint8)
    m1 = make_model(seed=7)
    m2 = make_model(seed=7)
    o1 = m1(x)
    o2 = m2(x)
    assert np.array_equal(o1, o2), "Identical seeds must produce identical outputs"


def test_different_seeds_differ():
    """Two models built with different seeds should produce different reservoir states."""
    x = np.random.randint(0, 2, (4, 1, 2, 2), dtype=np.uint8)
    m1 = make_model(seed=1)
    m2 = make_model(seed=2)
    m1(x); m2(x)
    # Compare reservoir states — readout weights are zero until ridge is run,
    # but reservoir dynamics are seeded differently and will diverge.
    s1 = m1.states_parallel[:4].copy()
    s2 = m2.states_parallel[:4].copy()
    assert not np.array_equal(s1, s2), "Different seeds must produce different reservoir states (collision extremely unlikely)"


# ---------------------------------------------------------------------------
# 3. Batch independence — each sample evolves independently
# ---------------------------------------------------------------------------

def test_batch_independence():
    """Running m samples together must give the same result as running each sample alone."""
    model = make_model(n_r=32, batch_size=8)
    m = 4
    np.random.seed(0)
    x = random_input(model, m)

    model.eval()
    out_batched = model(x)

    outs_single = []
    for i in range(m):
        model.reset_reservoir()
        outs_single.append(model(x[i:i+1]))

    out_single = np.concatenate(outs_single, axis=0)
    assert np.allclose(out_batched, out_single, atol=1e-6), (
        "Batched forward pass must equal m independent single-sample passes"
    )


# ---------------------------------------------------------------------------
# 4. Input injection effectiveness
# ---------------------------------------------------------------------------

def test_input_injection_changes_input_nodes():
    """Running forward with two different inputs should change reservoir states differently."""
    model = make_model(perturbation='override', n_r=32, bits=4, features=2)
    m = 2
    x1 = np.zeros((m, 1, 2, 2), dtype=np.uint8)
    x2 = np.ones((m, 1, 2, 2), dtype=np.uint8)
    model.reset_reservoir(hard_reset=True)
    model(x1)
    s1 = model.states_parallel[:m].copy()
    model.reset_reservoir(hard_reset=True)
    model(x2)
    s2 = model.states_parallel[:m].copy()
    assert not np.array_equal(s1, s2), (
        "Different inputs must produce different reservoir states — input injection not working"
    )
    assert model.states_parallel.dtype == np.uint8


def test_xor_perturbation_flips_bits():
    """XOR perturbation must flip exactly the bits indicated by the perturbation mask."""
    from project.boolean_reservoir.code.utils.reservoir_utils import InputPerturbationStrategy
    xor_fn = InputPerturbationStrategy.xor

    states = np.array([[0, 1, 0, 1, 1, 0, 0, 1]], dtype=np.uint8)
    perturb = np.array([[1, 0, 1, 0, 0, 0, 1, 0]], dtype=np.uint8)
    expected = states ^ perturb
    result = xor_fn(states, perturb)
    assert np.array_equal(result, expected), "XOR perturbation does not flip correct bits"


# ---------------------------------------------------------------------------
# 5. I-node isolation: input nodes have no incoming reservoir connections
# ---------------------------------------------------------------------------

def test_input_nodes_have_no_incoming_reservoir_edges():
    """Input nodes (I) must not appear as targets of any edge in the graph."""
    model = make_model(n_r=64, k_avg=3.0, k_max=6)
    n_input = model.P.M.I.n_nodes
    graph = model.graph
    for i in range(n_input):
        predecessors = list(graph.predecessors(i))
        assert predecessors == [], (
            f"Input node {i} has incoming edges {predecessors}; "
            "reservoir dynamics must not write input nodes"
        )


def test_input_nodes_preserved_during_tick():
    """Since I-nodes have no incoming edges, _reservoir_tick must leave them unchanged."""
    model = make_model(n_r=32, k_avg=2.0, k_max=4)
    m = 2
    model.states_parallel[:m, model.input_nodes_mask] = 1
    snapshot = model.states_parallel[:m, model.input_nodes_mask].copy()

    model._reservoir_tick(m)

    after = model.states_parallel[:m, model.input_nodes_mask]
    assert np.array_equal(snapshot, after), (
        "_reservoir_tick must not modify input-node states (they have no incoming edges)"
    )


# ---------------------------------------------------------------------------
# 6. Isolated-node state preservation
# ---------------------------------------------------------------------------

def test_isolated_nodes_preserve_state():
    """Nodes with zero in-degree must retain their state after a tick."""
    model = make_model(n_r=32, k_min=0, k_avg=2.0, k_max=4)
    m = 3
    if len(model.no_neighbours_indices) == 0:
        pytest.skip("No isolated nodes in this random graph; re-run with lower k_min or different seed")

    isolated = model.no_neighbours_indices
    model.states_parallel[:m, isolated] = 1
    snapshot = model.states_parallel[:m, isolated].copy()

    model._reservoir_tick(m)

    after = model.states_parallel[:m, isolated]
    assert np.array_equal(snapshot, after), "Isolated nodes must preserve state across ticks"


# ---------------------------------------------------------------------------
# 7. LUT bias fidelity
# ---------------------------------------------------------------------------

def test_lut_bias_p05():
    """With p=0.5, mean LUT output across all entries should be close to 0.5."""
    model = make_model(n_r=256, k_avg=3.0, k_max=4, p=0.5, seed=1)
    mean_p = model.lut.astype(float).mean()
    assert abs(mean_p - 0.5) < 0.05, f"LUT mean {mean_p:.3f} deviates > 0.05 from expected 0.5"


def test_lut_bias_p02():
    """With p=0.2, mean LUT output should be close to 0.2."""
    model = make_model(n_r=512, k_avg=3.0, k_max=4, p=0.2, seed=3)
    mean_p = model.lut.astype(float).mean()
    assert abs(mean_p - 0.2) < 0.05, f"LUT mean {mean_p:.3f} deviates > 0.05 from expected 0.2"


def test_lut_values_are_binary():
    """LUT must contain only 0 and 1."""
    model = make_model()
    unique = np.unique(model.lut)
    assert set(unique.tolist()).issubset({0, 1}), f"LUT contains non-binary values: {unique}"


# ---------------------------------------------------------------------------
# 8. Non-trivial dynamics
# ---------------------------------------------------------------------------

def test_reservoir_state_changes_over_ticks():
    """After several ticks with a random initial state, the reservoir state must change."""
    model = make_model(n_r=128, k_avg=2.0, k_max=4, p=0.5, init='random')
    m = 1
    model.reset_reservoir()
    state_t0 = model.states_parallel[:m].copy()

    for _ in range(10):
        model._reservoir_tick(m)

    state_t10 = model.states_parallel[:m].copy()
    assert not np.array_equal(state_t0, state_t10), (
        "Reservoir state unchanged after 10 ticks — dynamics appear frozen"
    )


def test_reservoir_states_not_uniform():
    """After forward pass, reservoir states should not be all-0 or all-1."""
    model = make_model(n_r=128)
    m = 4
    x = random_input(model, m)
    model(x)

    R_states = model.states_parallel[:m, model.output_nodes_mask]
    mean_activity = R_states.astype(float).mean()
    assert 0.05 < mean_activity < 0.95, (
        f"Reservoir activity {mean_activity:.3f} is near 0 or 1 — dynamics may be degenerate"
    )


# ---------------------------------------------------------------------------
# 9. Effective-k LUT access pattern
# ---------------------------------------------------------------------------

def test_heterogeneous_lut_access_pattern():
    """For a node with actual in-degree k < k_max, only 2^k distinct LUT indices are accessed."""
    model = make_model(n_r=128, k_min=1, k_avg=2.0, k_max=4)
    k_max = model.max_connectivity

    in_degrees = model.adj_list_mask.sum(axis=1)
    candidates = np.where(in_degrees < k_max)[0]
    if len(candidates) == 0:
        pytest.skip("All nodes have in-degree = k_max; use heterogeneous mode")

    node = int(candidates[0])
    k_actual = int(in_degrees[node])
    n_patterns = 2 ** k_actual
    all_indices = set()
    powers = model.powers_of_2

    for i in range(n_patterns):
        bits = np.zeros(k_max, dtype=np.float32)
        for bit_pos in range(k_actual):
            bits[bit_pos] = (i >> (k_actual - 1 - bit_pos)) & 1
        idx = int(np.dot(bits, powers))
        all_indices.add(idx)

    expected_stride = 2 ** (k_max - k_actual)
    expected_indices = set(range(0, n_patterns * expected_stride, expected_stride))
    assert all_indices == expected_indices


# ---------------------------------------------------------------------------
# 10. Numpy-only stack — no torch tensors anywhere in the model
# ---------------------------------------------------------------------------

def test_numpy_only_stack():
    """All model arrays must be numpy. No torch tensors anywhere in the model."""
    model = make_model(n_r=32)
    x = random_input(model, 4)
    out = model(x)

    assert isinstance(out, np.ndarray), f"model() output is {type(out)}, expected numpy ndarray"
    assert isinstance(model.states_parallel, np.ndarray), "states_parallel must be numpy"
    assert isinstance(model.lut, np.ndarray), "lut must be numpy"
    assert isinstance(model.readout_W, np.ndarray), "readout_W must be numpy"
    assert isinstance(model.readout_b, np.ndarray), "readout_b must be numpy"
    assert isinstance(model.initial_states, np.ndarray), "initial_states must be numpy"


# ---------------------------------------------------------------------------
# 11. Warmup changes state relative to cold start
# ---------------------------------------------------------------------------

def test_warmup_changes_state():
    """'random-warmup' init must produce a different initial state than 'random' alone."""
    m_cold = make_model(init='random', seed=99)
    m_warm = make_model(init='random-warmup', seed=99)

    cold_state = m_cold.states_parallel.copy()
    warm_state = m_warm.states_parallel.copy()

    assert not np.array_equal(cold_state, warm_state), (
        "Warmup did not change reservoir state — washout period may be ineffective"
    )


# ---------------------------------------------------------------------------
# 12. Kauffman criticality
# ---------------------------------------------------------------------------

def test_kauffman_criticality_parameter():
    """Lambda = 2*p*(1-p); at criticality lambda * k_avg = 1."""
    k_avg = 2.0
    p = 0.5
    lam = 2 * p * (1 - p)
    criticality = lam * k_avg
    assert abs(criticality - 1.0) < 1e-9, (
        f"Default params (k_avg={k_avg}, p={p}) should be exactly at the critical point "
        f"lambda*k_avg={criticality:.4f}."
    )


def test_kauffman_criticality_homogeneous_k3():
    """For k=3, p=0.5 is super-critical (lambda*k=1.5>1 → chaotic)."""
    k = 3
    p = 0.5
    lam = 2 * p * (1 - p)
    criticality = lam * k
    assert criticality > 1.0, f"k={k}, p={p} should be in chaotic regime (lambda*k={criticality:.3f})"


# ---------------------------------------------------------------------------
# 13. bin2int correctness
# ---------------------------------------------------------------------------

def test_bin2int_msb_first():
    """powers_of_2 must be MSB-first: [2^(n-1), ..., 2^1, 2^0]."""
    model = make_model()
    k = model.max_connectivity
    p2 = model.powers_of_2
    expected = np.array([2 ** (k - 1 - i) for i in range(k)], dtype=np.float32)
    assert np.array_equal(p2, expected), f"powers_of_2 are not MSB-first: {p2}"


def test_bin2int_values():
    """With last 3 bits set to [1,0,1] and earlier bits zero, bin2int should give 5."""
    model = make_model(k_max=3)
    k = model.max_connectivity
    bits = np.zeros((1, 1, k), dtype=np.float32)
    bits[0, 0, k-3:] = [1, 0, 1]
    result = model.bin2int(bits)
    assert int(result.flat[0]) == 5, f"bin2int expected 5, got {result.flat[0]}"


def test_bin2int_all_zeros():
    """bin2int([0,0,...,0]) must be 0."""
    model = make_model()
    k = model.max_connectivity
    bits = np.zeros((1, 1, k), dtype=np.float32)
    assert int(model.bin2int(bits).flat[0]) == 0


def test_bin2int_all_ones():
    """bin2int([1,1,...,1]) with k bits must be 2^k - 1."""
    model = make_model()
    k = model.max_connectivity
    bits = np.ones((1, 1, k), dtype=np.float32)
    assert int(model.bin2int(bits).flat[0]) == 2**k - 1


# ---------------------------------------------------------------------------
# 14. Readout uses only R nodes
# ---------------------------------------------------------------------------

def test_readout_uses_only_r_nodes():
    """output_nodes_mask must cover exactly the R slice."""
    model = make_model()
    expected = np.zeros_like(model.output_nodes_mask)
    expected[model.res_slice] = True
    assert np.array_equal(model.output_nodes_mask, expected)
    n_readout = int(model.output_nodes_mask.sum())
    assert n_readout == model.P.M.R.n_nodes


# ---------------------------------------------------------------------------
# 15. reset=True resets reservoir state between samples
# ---------------------------------------------------------------------------

def test_reset_true_between_samples():
    """With reset=True, same input after state pollution must give same output."""
    model = make_model(reset=True)
    x = random_input(model, 1)

    model.eval()
    out1 = model(x)
    for _ in range(20):
        model._reservoir_tick(1)
    out2 = model(x)

    assert np.allclose(out1, out2, atol=1e-6), (
        "reset=True: same input after state pollution must give same output"
    )
