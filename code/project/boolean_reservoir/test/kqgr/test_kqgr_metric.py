"""
Tests for the KQGR metric input construction and rank computation.

Verifies:
  1. Degenerate tau (too large) warns rather than crashes.
  2. State matrix rank vs distinct-input count for real reservoir runs.
  3. KQ and GR datasets have identical shapes (catches double-tau-counting bug).
  4. All KQ samples are distinct.
"""

import numpy as np
import pytest

from project.boolean_reservoir.code.parameter import load_yaml_config

FIGURE1_CONFIG = 'config/temporal/kqgr/grid_search/figure_1_snyder_2012.yaml'
N_NODES = 25  # reservoir size in figure_1 config


# ---------------------------------------------------------------------------
# 1. Degenerate tau warning
# ---------------------------------------------------------------------------

def _load_kqgr_dataset_tau(tau, eval_mode='last'):
    from project.temporal.code.dataset_init import TemporalDatasetInit
    P = load_yaml_config(FIGURE1_CONFIG)
    P_uni = P.U.kqgr_homogeneous
    P_uni.D.tau = tau
    P_uni.D.evaluation = eval_mode
    P_uni.M.R.n_nodes = N_NODES
    dataset = TemporalDatasetInit().kqgr(P_uni, kq=False)
    return dataset.x


def test_input_diversity_tau_degenerate_warns():
    """tau=7 with N=25, bits=10 → 2^3=8 < 25: should warn, not crash."""
    import warnings
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        x = _load_kqgr_dataset_tau(tau=7, eval_mode='last')
        assert any("repeat" in str(warning.message).lower() for warning in w),             "Expected a UserWarning about repeating inputs"
    assert x.shape[1] == 10


# ---------------------------------------------------------------------------
# 2. State matrix rank sanity
# ---------------------------------------------------------------------------

def _build_model_and_run_kqgr(tau, k_avg=5, eval_mode='last'):
    from project.boolean_reservoir.code.reservoir import BooleanReservoir, BatchedTensorHistoryWriter
    from project.temporal.code.dataset_init import TemporalDatasetInit
    from project.boolean_reservoir.code.parameter import (
        Params, ModelParams, InputParams, ReservoirParams, OutputParams, TrainingParams,
        PatternWiring,
    )
    from pathlib import Path
    import tempfile

    P = Params(
        model=ModelParams(
            input_layer=InputParams(
                n_nodes=1,
                perturbation='override', encoding='base2',
                features=1, resolution=10, chunk_size=1,
            ),
            reservoir_layer=ReservoirParams(
                n_nodes=N_NODES, k_avg=k_avg, k_min=k_avg, k_max=k_avg,
                mode='homogeneous', p=0.5, reset=True,
            ),
            output_layer=OutputParams(n_nodes=1),
            training=TrainingParams(),
            wiring=[
                PatternWiring(source='B', target='I', pattern='identity'),
                PatternWiring(source='I', target='R', pattern='identity'),
            ],
        ),
        multiverse_overrides={
            'kqgr': {'dataset': {'tau': tau, 'evaluation': eval_mode}}
        }
    )
    P_uni = P.U.kqgr

    dataset = TemporalDatasetInit().kqgr(P_uni, kq=False)
    x = dataset.x
    n_distinct_inputs = len(np.unique(x.reshape(x.shape[0], -1), axis=0))

    with tempfile.TemporaryDirectory() as tmpdir:
        P_uni.L.save_path = Path(tmpdir)
        P_uni.L.out_path = Path(tmpdir)
        model = BooleanReservoir(P_uni)
        model.eval()
        model.record = True
        model.history = BatchedTensorHistoryWriter(
            save_path=Path(tmpdir) / 'history',
            persist_to_disk=False
        )
        _ = model(x)
        model.flush_history()

        _, history, expanded_meta, _ = model.history.reload_history()
        df_filter = expanded_meta[expanded_meta['phase'] == 'output_layer']
        filtered = history[df_filter.index].astype(np.float32)
        state_matrix = filtered[:, ~model.input_nodes_mask]

    return state_matrix, n_distinct_inputs


@pytest.mark.parametrize("tau", [0, 1, 3])
def test_state_matrix_rank_sanity(tau):
    """Rank <= n_reservoir and rank <= distinct_inputs."""
    state_matrix, n_distinct_inputs = _build_model_and_run_kqgr(tau=tau, k_avg=5.0)
    rank = np.linalg.matrix_rank(state_matrix)
    n_reservoir = state_matrix.shape[1]
    assert rank <= n_reservoir
    assert rank <= n_distinct_inputs


# ---------------------------------------------------------------------------
# 3 & 4. KQ/GR shape invariant + KQ sample distinctness
# ---------------------------------------------------------------------------

PI_CONTINUOUS_CONFIG = 'config/path_integration/1D/grid_search/design_choices/continuous.yaml'


def _get_kqgr_datasets(config_path):
    from project.boolean_reservoir.code.utils.param_utils import generate_param_combinations
    from project.boolean_reservoir.code.kq_and_gr_metric import prepare_kqgr_model_params
    P = load_yaml_config(config_path)
    combos = generate_param_combinations(P)
    P0 = combos[0]
    universe_key = next(iter(P0.multiverse_overrides or {}), None)
    if universe_key is None:
        pytest.skip('No multiverse in config')
    P_universe = getattr(P0.U, universe_key)
    P_kqgr_model = prepare_kqgr_model_params(P_universe)
    init = P_universe.dataset_init_obj
    kq = init.kqgr(P_kqgr_model, kq=True)
    gr = init.kqgr(P_universe, kq=False)
    return kq.data['x'], gr.data['x'], P_universe


@pytest.mark.parametrize('config_path', [
    FIGURE1_CONFIG,
    PI_CONTINUOUS_CONFIG,
])
def test_kq_gr_same_shape(config_path):
    """KQ and GR datasets must have identical shapes (catches double-tau-counting bug)."""
    kq_x, gr_x, _ = _get_kqgr_datasets(config_path)
    assert kq_x.shape == gr_x.shape, (
        'KQ shape ' + str(kq_x.shape) + ' != GR shape ' + str(gr_x.shape) +
        ' for ' + config_path
    )


@pytest.mark.parametrize('config_path', [
    FIGURE1_CONFIG,
    PI_CONTINUOUS_CONFIG,
])
def test_kq_all_samples_distinct(config_path):
    """KQ measures input separability — every sample must be distinct."""
    kq_x, _, _ = _get_kqgr_datasets(config_path)
    flat = kq_x.reshape(kq_x.shape[0], -1)
    n_distinct = len(np.unique(flat, axis=0))
    assert n_distinct == len(flat), (
        'KQ dataset has only ' + str(n_distinct) + '/' + str(len(flat)) + ' distinct samples.'
    )
