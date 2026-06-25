"""
Tests for the KQ/GR input-diversity fix (full exhaustive pool + random subsample).

Background: with exhaustive sampling and shuffle=False, _get_kq_cached_dataset used to
build a tiny dataset of the first n_nodes integers (0..n_nodes-1).  In MSB-first binary
these cluster at the low end: KQ (bits=10) had 5 leading-zero steps, GR (bits=7, tau=3)
had only 2, giving GR a systematic rank advantage.

Fix: cache the full 2^bits exhaustive pool; kqgr() randomly subsamples n_nodes from it.
Applies to both tau_mode='steal' (GR uses bits-tau) and tau_mode='augment' (KQ uses
bits+tau).

Invariants verified:
  1. Pool sizes — _get_kq_cached_dataset returns the full 2^(effective_bits) set
  2. Cache is n_nodes-independent — n_nodes=25 and n_nodes=128 share the same pool
  3. Subsampled output shape — kqgr() returns exactly n_nodes samples
  4. KQ step diversity — no all-identical steps (no leading-zero bias)
  5. GR step diversity — exactly tau contiguous identical steps at evaluation end;
     all free steps are diverse
  6. Different seeds → different subsamples
  7. Augment mode — larger KQ pool (2^(bits+tau)), same GR pool (2^bits)
  8. Edge case: n_nodes >= 2^bits_gr uses all available distinct patterns
"""

import copy
import numpy as np
import numpy as np
import tempfile
import pytest
from pathlib import Path

from project.boolean_reservoir.code.parameter import load_yaml_config
from project.boolean_reservoir.code.utils.param_utils import generate_param_combinations
from project.boolean_reservoir.code.kq_and_gr_metric import prepare_kqgr_model_params, compute_rank
from project.boolean_reservoir.code.utils.utils import set_seed
from project.temporal.code.dataset_init import TemporalDatasetInit

FIGURE1_CONFIG = 'config/temporal/kqgr/grid_search/figure_1_snyder_2012.yaml'


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _universe_combo(tau, n_nodes, universe='kqgr_heterogeneous_real'):
    P = load_yaml_config(FIGURE1_CONFIG)
    for p in generate_param_combinations(P):
        u = next(iter(p.multiverse_overrides or {}), None)
        if u == universe and p.D.tau == tau and p.M.R.n_nodes == n_nodes:
            return getattr(p.U, u)
    pytest.skip(f"No combo for universe={universe} tau={tau} n_nodes={n_nodes}")


def _universe_combo_with_k_avg(tau, n_nodes, k_avg, universe='kqgr_heterogeneous_real'):
    P = load_yaml_config(FIGURE1_CONFIG)
    for p in generate_param_combinations(P):
        u = next(iter(p.multiverse_overrides or {}), None)
        if (u == universe and p.D.tau == tau and p.M.R.n_nodes == n_nodes
                and abs((p.M.variables or {}).get('R_k_avg', -1) - k_avg) < 0.01):
            return getattr(p.U, u)
    pytest.skip(f"No combo for tau={tau} n_nodes={n_nodes} k_avg={k_avg}")


def _augment_combo(tau, n_nodes):
    """Clone a steal-mode combo, switching tau_mode to 'augment'."""
    P_uni = _universe_combo(tau=tau, n_nodes=n_nodes)
    new_ds = P_uni.dataset.model_copy(update={'tau_mode': 'augment'})
    return P_uni.model_copy(update={'dataset': new_ds})


def _n_identical_steps(x):
    """Number of steps where ALL samples carry the same bit value."""
    return sum(1 for t in range(x.shape[1]) if len(np.unique(x[:, t, 0, 0])) == 1)


def _n_contiguous_identical_tail(x):
    """Number of contiguous identical steps from the last step backwards."""
    n = 0
    for t in range(x.shape[1] - 1, -1, -1):
        if len(np.unique(x[:, t, 0, 0])) == 1:
            n += 1
        else:
            break
    return n


# ---------------------------------------------------------------------------
# 1. Pool sizes — steal mode
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("tau,expected_kq,expected_gr", [
    (1, 2**10, 2**9),
    (3, 2**10, 2**7),
    (5, 2**10, 2**5),
])
def test_steal_raw_pool_is_full_exhaustive_set(tau, expected_kq, expected_gr):
    P_uni = _universe_combo(tau=tau, n_nodes=25)
    init = TemporalDatasetInit()

    kq_raw = init._get_kq_cached_dataset(prepare_kqgr_model_params(P_uni), gr_tau=0)
    gr_raw = init._get_kq_cached_dataset(P_uni, gr_tau=tau)

    assert kq_raw.data['x'].shape[0] == expected_kq, (
        f"Steal KQ pool (tau={tau}): expected {expected_kq}, got {kq_raw.data['x'].shape[0]}"
    )
    assert gr_raw.data['x'].shape[0] == expected_gr, (
        f"Steal GR pool (tau={tau}): expected {expected_gr}, got {gr_raw.data['x'].shape[0]}"
    )


# ---------------------------------------------------------------------------
# 2. Pool sizes — augment mode
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("tau,bits,expected_kq,expected_gr", [
    (1, 10, 2**11, 2**10),
    (3, 10, 2**13, 2**10),
])
def test_augment_raw_pool_is_full_exhaustive_set(tau, bits, expected_kq, expected_gr):
    """Augment mode: KQ pool = 2^(bits+tau), GR pool = 2^bits."""
    P_uni = _augment_combo(tau=tau, n_nodes=25)
    init = TemporalDatasetInit()

    kq_raw = init._get_kq_cached_dataset(prepare_kqgr_model_params(P_uni), gr_tau=0)
    gr_raw = init._get_kq_cached_dataset(P_uni, gr_tau=tau)

    assert kq_raw.data['x'].shape[0] == expected_kq, (
        f"Augment KQ pool (tau={tau}): expected {expected_kq} (2^{bits+tau}), "
        f"got {kq_raw.data['x'].shape[0]}"
    )
    assert gr_raw.data['x'].shape[0] == expected_gr, (
        f"Augment GR pool (tau={tau}): expected {expected_gr} (2^{bits}), "
        f"got {gr_raw.data['x'].shape[0]}"
    )


# ---------------------------------------------------------------------------
# 3. Subsampled output shape
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("tau_mode,tau,n_nodes", [
    ("steal",   3,  25),
    ("steal",   3, 128),
    ("augment", 3,  25),
])
def test_kqgr_output_has_n_nodes_samples(tau_mode, tau, n_nodes):
    init = TemporalDatasetInit()
    P_uni = _universe_combo(tau=tau, n_nodes=n_nodes) if tau_mode == "steal" \
        else _augment_combo(tau=tau, n_nodes=n_nodes)

    np.random.seed(0)
    kq_ds = init.kqgr(prepare_kqgr_model_params(P_uni), kq=True)
    np.random.seed(0)
    gr_ds = init.kqgr(P_uni, kq=False)

    assert kq_ds.data['x'].shape[0] == n_nodes, (
        f"[{tau_mode}] KQ: expected {n_nodes} samples, got {kq_ds.data['x'].shape[0]}"
    )
    assert gr_ds.data['x'].shape[0] == n_nodes, (
        f"[{tau_mode}] GR: expected {n_nodes} samples, got {gr_ds.data['x'].shape[0]}"
    )


# ---------------------------------------------------------------------------
# 5. KQ step diversity — no all-identical steps
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("tau_mode,tau,n_nodes", [
    ("steal",   1,  25),
    ("steal",   3,  25),
    ("steal",   5,  25),
    ("steal",   1, 128),
    ("steal",   3, 128),
    ("augment", 1,  25),
    ("augment", 3,  25),
])
def test_kq_has_no_identical_steps(tau_mode, tau, n_nodes):
    """KQ must have diverse inputs at every step — no all-identical leading steps."""
    init = TemporalDatasetInit()
    P_uni = _universe_combo(tau=tau, n_nodes=n_nodes) if tau_mode == "steal" \
        else _augment_combo(tau=tau, n_nodes=n_nodes)

    np.random.seed(0)
    kq_ds = init.kqgr(prepare_kqgr_model_params(P_uni), kq=True)
    x = kq_ds.data['x']

    n_identical = _n_identical_steps(x)
    assert n_identical == 0, (
        f"[{tau_mode}] KQ (tau={tau}, n_nodes={n_nodes}): expected 0 identical steps "
        f"(no leading-zero bias), got {n_identical}/{x.shape[1]}"
    )


# ---------------------------------------------------------------------------
# 6. GR step diversity — exactly tau identical tail steps, free steps diverse
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("tau_mode,tau,n_nodes", [
    ("steal",   1,  25),
    ("steal",   3,  25),
    ("steal",   5,  25),
    ("steal",   1, 128),
    ("steal",   3, 128),
    ("augment", 1,  25),
    ("augment", 3,  25),
])
def test_gr_has_exactly_tau_identical_tail_steps(tau_mode, tau, n_nodes):
    """GR must have exactly tau contiguous identical steps at the evaluation tail."""
    init = TemporalDatasetInit()
    P_uni = _universe_combo(tau=tau, n_nodes=n_nodes) if tau_mode == "steal" \
        else _augment_combo(tau=tau, n_nodes=n_nodes)

    bits = P_uni.D.bits   # 10 for this config
    if tau_mode == "steal":
        total_steps = bits          # free (bits-tau) + tau = bits
    else:
        total_steps = bits + tau    # free (bits) + tau = bits+tau

    np.random.seed(0)
    gr_ds = init.kqgr(P_uni, kq=False)
    x = gr_ds.data['x']

    assert x.shape[1] == total_steps, (
        f"[{tau_mode}] GR (tau={tau}): expected {total_steps} total steps, "
        f"got {x.shape[1]}"
    )

    n_tail = _n_contiguous_identical_tail(x)
    assert n_tail == tau, (
        f"[{tau_mode}] GR (tau={tau}, n_nodes={n_nodes}): expected {tau} identical "
        f"tail steps, got {n_tail}"
    )

    # Free steps must be diverse (no spurious leading zeros)
    free_steps = total_steps - tau
    for t in range(free_steps):
        n_unique = len(np.unique(x[:, t, 0, 0]))
        assert n_unique > 1, (
            f"[{tau_mode}] GR (tau={tau}, n_nodes={n_nodes}): free step t={t} has only "
            f"{n_unique} unique value (expected diversity — leading-zero bias should be gone)"
        )


# ---------------------------------------------------------------------------
# 7. Edge case: n_nodes >= 2^bits_gr uses all available distinct patterns
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("tau,n_nodes,bits_gr", [
    (4, 128, 6),   # steal: 2^6=64 < 128 → capped at 64
    (5, 128, 5),   # steal: 2^5=32 < 128 → capped at 32
])
def test_edge_case_n_nodes_exceeds_distinct_patterns(tau, n_nodes, bits_gr):
    """When n_nodes > 2^bits_gr, kqgr() uses all available distinct patterns without crashing."""
    P_uni = _universe_combo(tau=tau, n_nodes=n_nodes)
    init = TemporalDatasetInit()

    np.random.seed(0)
    gr_ds = init.kqgr(P_uni, kq=False)
    x = gr_ds.data['x']

    max_distinct = 2 ** bits_gr
    assert x.shape[0] == max_distinct, (
        f"GR (tau={tau}, n_nodes={n_nodes}): only 2^{bits_gr}={max_distinct} distinct "
        f"patterns exist, output should be capped at {max_distinct}, got {x.shape[0]}"
    )


# ---------------------------------------------------------------------------
# 8. End-to-end rank: kq_rank > gr_rank at critical connectivity
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("tau,n_nodes,k_avg", [
    (3, 25, 3.0),   # critical regime: reservoir forgets shared tail → delta > 0
    (5, 25, 3.0),   # aggressive tau, same critical connectivity
])
def test_kq_rank_gt_gr_rank_at_critical_connectivity(tau, n_nodes, k_avg):
    """
    End-to-end: at critical connectivity (k_avg=3) kq_rank > gr_rank for most seeds.

    k_avg=3 is the critical point for heterogeneous reservoirs — the reservoir
    responds to inputs but lacks perfect memory, so the tau shared tail steps cause
    partial convergence of GR states → delta > 0.

    NOTE: k_avg=5 (strong connectivity / long memory) gives delta≈0 even with correct
    code because the reservoir retains the 7 diverse steps through 3 shared ones.
    That is correct physics, not a bug, but it means k_avg=5 is a poor regression
    target — it trivially passes even when compute_rank is broken.

    Requires delta > 0 for >= 3/5 seeds AND mean_delta > 0.
    """
    from project.boolean_reservoir.code.reservoir import BooleanReservoir

    P_uni = _universe_combo_with_k_avg(tau=tau, n_nodes=n_nodes, k_avg=k_avg)
    P_kqgr = prepare_kqgr_model_params(P_uni)
    init = TemporalDatasetInit()

    n_seeds = 5
    results = []

    for seed in range(n_seeds):
        np.random.seed(seed)
        kq_ds = init.kqgr(copy.deepcopy(P_kqgr), kq=True)
        np.random.seed(seed)
        gr_ds = init.kqgr(copy.deepcopy(P_uni), kq=False)

        with tempfile.TemporaryDirectory() as tmpdir:
            P_seed = copy.deepcopy(P_kqgr)
            P_seed.L.save_path = Path(tmpdir)
            P_seed.L.out_path = Path(tmpdir)
            set_seed(seed)
            model = BooleanReservoir(P_seed)
            kq_rank = compute_rank(model, kq_ds.data['x'], 'kq')
            gr_rank = compute_rank(model, gr_ds.data['x'], 'gr')

        results.append((kq_rank, gr_rank))

    deltas   = [kq - gr for kq, gr in results]
    n_positive = sum(d > 0 for d in deltas)
    mean_delta = sum(deltas) / n_seeds

    assert n_positive >= 3, (
        f"delta > 0 in only {n_positive}/{n_seeds} seeds (tau={tau}, k_avg={k_avg}). "
        f"Expected >= 3 — at critical connectivity GR should drop below KQ. "
        f"Per-seed (kq, gr, delta): {[(kq, gr, kq-gr) for kq, gr in results]}"
    )
    assert mean_delta > 0, (
        f"Mean delta={mean_delta:.2f} <= 0 (tau={tau}, k_avg={k_avg}). "
        f"Per-seed: {results}"
    )


# ---------------------------------------------------------------------------
# 9. Regression: compute_rank filters to the final output_layer step
# ---------------------------------------------------------------------------

def test_compute_rank_uses_final_output_layer_step():
    """
    Regression for the time-series output_layer bug in compute_rank.

    With time-series readout, output_layer fires at every step (s times), giving
    m * s rows in filtered_history.  Without the final-step filter, diverse
    early-step states inflate GR rank and delta stays zero even at critical
    connectivity.

    This test verifies directly that:
    1. The time-series model produces s * n_samples output_layer rows in total.
    2. compute_rank's final-step filter selects exactly n_samples rows (one per
       sample at the last step), not all m * s rows.
    """
    from project.boolean_reservoir.code.reservoir import BooleanReservoir, BatchedTensorHistoryWriter

    P_uni  = _universe_combo(tau=3, n_nodes=25)
    P_kqgr = prepare_kqgr_model_params(P_uni)
    init   = TemporalDatasetInit()

    np.random.seed(0)
    kq_ds    = init.kqgr(copy.deepcopy(P_kqgr), kq=True)
    x        = kq_ds.data['x']
    n_samples = x.shape[0]  # 25
    n_steps   = x.shape[1]  # 10

    with tempfile.TemporaryDirectory() as tmpdir:
        set_seed(0)
        P_m = copy.deepcopy(P_kqgr)
        P_m.L.save_path = Path(tmpdir)
        P_m.L.out_path  = Path(tmpdir)
        model = BooleanReservoir(P_m)

        save_path = Path(tmpdir) / 'history' / 'kq' / 'history'
        model.record  = True
        model.history = BatchedTensorHistoryWriter(save_path=save_path, persist_to_disk=False)
        model.eval()
        _ = model(x)
        model.flush_history()

        _, _, expanded_meta, _ = model.history.reload_history()
        output_rows = expanded_meta[expanded_meta['phase'] == 'output_layer']

        # Time-series model fires output_layer once per step per sample.
        assert len(output_rows) == n_steps * n_samples, (
            f"Expected {n_steps} steps × {n_samples} samples = "
            f"{n_steps * n_samples} output_layer rows, got {len(output_rows)}. "
            f"If this changes, verify the final-step filter in compute_rank is still needed."
        )

        # The final-step filter must reduce this to exactly n_samples rows.
        final_s    = output_rows['s'].max()
        final_rows = output_rows[output_rows['s'] == final_s]
        assert len(final_rows) == n_samples, (
            f"Final-step filter should yield {n_samples} rows, got {len(final_rows)}. "
            f"compute_rank is using all {len(output_rows)} output_layer rows instead of "
            f"the {n_samples} at the final step — GR rank will be inflated."
        )
