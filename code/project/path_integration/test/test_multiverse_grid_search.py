"""Test multi-universe grid search correctness.

Verifies that:
- Additive multiverse expansion produces the right number of configs (Mother × additive variants)
- Each config has exactly one universe and one set of kqgr metrics (not a dict, not None)
- kqgr metrics are valid (non-negative ranks)
- Training results are populated for all configs
- Universe names are correctly recorded in P.L.universe
"""
from pathlib import Path
from shutil import rmtree
import logging

from project.boolean_reservoir.code.train_model_parallel import boolean_reservoir_grid_search
from project.boolean_reservoir.code.utils.load_save import load_params_df
from project.boolean_reservoir.code.utils.utils import configure_logging
from project.parallel_grid_search.code.parallel_utils import RUN

CONFIG = 'project/path_integration/test/config/2D/grid_search/multiverse_kqgr.yaml'
# Universe-aware expansion (squashing):
#   kqgr_PI: k_avg:[2,3] × samples:[2000,4000] × PI_variants=1 (tau:2, evaluation:first) = 4
#   kqgr_T:  k_avg:[2,3] × samples squashed (type-swap discards Mother dataset) × T_variants=1 = 2
#   Total = 4 + 2 = 6
EXPECTED_RECORDS = 4 + 2


def test_multiverse_grid_search():
    out_path = Path('/tmp/boolean_reservoir/test/path_integration/2D/grid_search/multiverse_kqgr')
    if out_path.exists():
        assert "/tmp/boolean_reservoir/test" in str(out_path)
        rmtree(out_path)

    P = boolean_reservoir_grid_search(
        CONFIG,
        cpu_memory_per_job_gb=0.5,
        cpu_cores_per_job=2,
    )

    parquet_path = P.L.out_path / RUN.data_dir / RUN.compacted_file
    assert parquet_path.exists(), f"{RUN.compacted_file} not found at {parquet_path}"

    df = load_params_df(parquet_path, fast=True)

    # --- Record count ---
    assert len(df) == EXPECTED_RECORDS, (
        f"Expected {EXPECTED_RECORDS} records (2 Mother × 2 additive universes), "
        f"got {len(df)}. Check _expand_multiverse_additive logic."
    )

    for i, row in df.iterrows():
        p = row['params']

        # --- Training results populated ---
        assert p.L.T is not None, f"Row {i}: P.L.train is None — training did not run"
        assert p.L.T.accuracy is not None, f"Row {i}: P.L.train.accuracy is None"
        assert p.L.T.loss is not None, f"Row {i}: P.L.train.loss is None"

        # --- kqgr metrics populated ---
        assert p.L.kqgr is not None, f"Row {i}: P.L.kqgr is None — kqgr did not run"
        assert p.L.kqgr.kq is not None and p.L.kqgr.kq >= 0, (
            f"Row {i}: P.L.kqgr.kq={p.L.kqgr.kq} is invalid"
        )
        assert p.L.kqgr.gr is not None and p.L.kqgr.gr >= 0, (
            f"Row {i}: P.L.kqgr.gr={p.L.kqgr.gr} is invalid"
        )

        # --- Universe name correctly recorded ---
        assert p.L.universe in ('kqgr_PI', 'kqgr_T'), (
            f"Row {i}: P.L.universe={p.L.universe!r} — expected 'kqgr_PI' or 'kqgr_T'"
        )

    # --- Both universes present, with correct per-universe counts ---
    universes = {row['params'].L.universe for _, row in df.iterrows()}
    assert universes == {'kqgr_PI', 'kqgr_T'}, (
        f"Expected both universes in results, got: {universes}"
    )
    pi_records = [row for _, row in df.iterrows() if row['params'].L.universe == 'kqgr_PI']
    t_records  = [row for _, row in df.iterrows() if row['params'].L.universe == 'kqgr_T']
    assert len(pi_records) == 4, f"Expected 4 kqgr_PI records (k_avg×samples), got {len(pi_records)}"
    assert len(t_records)  == 2, f"Expected 2 kqgr_T records (k_avg only; samples squashed), got {len(t_records)}"


def test_kqgr_model_shares_reservoir():
    """kqgr_model must reuse training model's graph/lut/init_state — not regenerate.

    The reservoir graph and lut are shared via load_dict to guarantee determinism on GPU.
    Re-generating from the same seed is not safe: async CUDA ops can corrupt random state
    between set_seed(I.seed) and set_seed(R.seed).
    """
    from project.boolean_reservoir.code.parameter import load_yaml_config
    from project.boolean_reservoir.code.reservoir import BooleanReservoir
    from project.boolean_reservoir.code.utils.param_utils import generate_param_combinations

    P = load_yaml_config(CONFIG)
    combos = generate_param_combinations(P)
    combo = combos[0]

    model = BooleanReservoir(combo)
    universe_key = next(iter(combo.multiverse_overrides))
    P_universe = getattr(combo.U, universe_key)
    kqgr_model = BooleanReservoir(P_universe, load_dict={
        'graph': model.graph,
        'lut': model.lut,
        'init_state': model.initial_states,
    })

    assert model.graph is kqgr_model.graph, (
        "kqgr_model.graph must be the same object as model.graph (shared, not re-generated)"
    )
    assert (model.lut == kqgr_model.lut).all(), "lut mismatch between training and kqgr model"
    assert (model.initial_states == kqgr_model.initial_states).all(), (
        "init_state mismatch between training and kqgr model"
    )


if __name__ == '__main__':
    configure_logging()
    test_multiverse_grid_search()
