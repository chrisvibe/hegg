from project.boolean_reservoir.code.reservoir import BooleanReservoir, BatchedTensorHistoryWriter
from project.boolean_reservoir.code.utils.utils import override_symlink
from pathlib import Path
import copy
import numpy as np  # torch removed
import numpy as np
from benchmark.utils.parameter import KQGRDatasetParams


def prepare_kqgr_model_params(P):
    """Deep-copy P and bump I.resolution by tau for augment mode with tau_axis='resolution'.
    For steal mode or tau_axis='steps', the copy is returned unchanged — steps augmentation
    adds extra time steps, not extra bits per step, so the model architecture is unchanged.
    Nulls derived I fields (bits, chunk_size, n_nodes) so the existing
    Pydantic validator chain (calculate_bits → handle_chunking → default_n_nodes)
    recomputes them from the new resolution — no manual arithmetic.
    """
    from project.boolean_reservoir.code.parameter import InputParams
    assert isinstance(P.D, KQGRDatasetParams)
    P = copy.deepcopy(P)
    if P.D.tau_mode != 'augment' or P.D.tau_axis != 'resolution':
        return P
    I = P.M.I
    updated = I.model_dump()
    updated.update({'resolution': I.resolution + P.D.tau,
                    'bits': None, 'chunk_size': None, 'n_nodes': None})
    P.M.input_layer = InputParams(**updated)
    return P

def compute_rank(model: BooleanReservoir, x, metric: str = '', reset_reservoir: bool = True) -> int:
    """Compute reservoir rank from the final state after a forward pass.

    Reads states_parallel directly — no history recording, no disk I/O.
    `metric` is accepted for call-site compatibility but not used.
    """
    m = x.shape[0]
    record = model.record
    model.record = False
    try:
        model.eval()
        _ = model(x)
    finally:
        model.record = record
    states = model.states_parallel[:m, model.res_slice].astype(np.float32)
    if reset_reservoir:
        model.reset_reservoir(hard_reset=True)
    return int(np.linalg.matrix_rank(states))


def compute_rank_flexible(model: BooleanReservoir, x, metric: str, reset_reservoir: bool = True) -> int:
    """Compute reservoir rank via full history recording (flexible but slower).

    Records all phases, filters to the final output_layer step, and returns the
    rank of the resulting (m × N_R) state matrix.  Use when you need to inspect
    intermediate states or save history artefacts alongside the rank.
    """
    nested_out = model.L.save_path / 'history' / metric
    new_save_path = nested_out / 'history'

    record = model.record
    try:
        model.record = True
        if model.history:
            model.history = BatchedTensorHistoryWriter(
                save_path=new_save_path,
                persist_to_disk=model.history.persist_to_disk,
                buffer_size=model.history.buffer_size
            )
        else:
            model.history = BatchedTensorHistoryWriter(save_path=new_save_path, persist_to_disk=False)
        model.eval()
        _ = model(x)
        model.flush_history()
    finally:
        model.record = record

    override_symlink(Path('../../checkpoint'), new_save_path / 'checkpoint')
    load_dict, history, expanded_meta, meta = model.history.reload_history()
    output_layer = expanded_meta[expanded_meta['phase'] == 'output_layer']
    final_s = output_layer['s'].max()
    df_filter = output_layer[output_layer['s'] == final_s]
    filtered_history = history[df_filter.index].astype(np.float32)
    reservoir_node_history = filtered_history[:, model.res_slice]
    if reset_reservoir:
        model.reset_reservoir(hard_reset=True)
    return int(np.linalg.matrix_rank(reservoir_node_history))