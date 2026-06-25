"""Shared utilities for ring-CA / BooleanReservoir test files."""

import numpy as np
from pathlib import Path

from project.boolean_reservoir.code.parameter import load_yaml_config
from project.boolean_reservoir.code.reservoir import BooleanReservoir
from project.boolean_reservoir.code.utils.utils import set_seed
from project.boolean_reservoir.code.encoding import BooleanTransformer


def make_ca_input(bits: np.ndarray, P) -> np.ndarray:
    """Shape a 1D binary stream into the correct input tensor for forward().

    When I.n_nodes == 0 (autonomous CA, no input layer): returns all-zero
    (1, T, 1, 1) — the zeros are ignored by Julia since N_I=0.

    Otherwise pipes bits through BooleanTransformer so redundancy, permute, and
    interleaving are applied consistently with the training pipeline.  With
    redundancy=2, each bit is automatically tiled to both chunk positions.

    bits: shape (T,) with values 0/1
    Returns: np.uint8 array of shape (1, T, 1, 1) or (1, T, I.chunks, I.chunk_size)
    """
    if P.M.I.n_nodes == 0:
        return np.zeros((1, len(bits), 1, 1), dtype=np.uint8)
    x = bits.reshape(1, -1, 1, 1).astype(np.uint8)  # (1, T, 1, 1)
    return BooleanTransformer(P)(x)


def make_ca_model(config: Path, install_fn) -> BooleanReservoir:
    """Load config, build a BooleanReservoir, install custom wiring, return model."""
    set_seed(0)
    model = BooleanReservoir(load_yaml_config(config))
    install_fn(model, model.R.n_nodes)
    return model


def override_to_viz_mode(P) -> None:
    """Set input layer to features=1 so each forward step = exactly 1 reservoir tick.
    Required when a training config uses features > 1 (multiple chunks per step)."""
    P.M.I.features   = 1
    P.M.I.bits       = 1
    P.M.I.n_nodes    = 1
    P.M.I.chunks     = 1
    P.M.I.chunk_size = 1


def run_and_plot(model: BooleanReservoir, x: np.ndarray, file_name: str,
                 highlight_input_nodes: bool = False) -> None:
    """Run forward pass, flush history, save model, generate activity trace SVG.

    This is the intended entry point for generating trace history for CA models.
    CA configs (config/test/boolean_reservoir/ca/*.yaml) cannot be run as standalone
    grid searches because the models require custom wiring installed in code (ring
    lattice, specific LUT overrides) that the YAML config system cannot express.
    TODO: consider a CA runner script or make_hegg_branch.sh step that calls each
    CA test's run_and_plot so trace history is generated automatically without
    having to run pytest manually.
    """
    model.eval()
    model(x)
    model.flush_history()
    model.save()
    try:
        from project.boolean_reservoir.code.visualization import plot_activity_trace
        plot_activity_trace(model.save_path, file_name=file_name)
    except ImportError:
        pass  # trace_plot_app not available; history and save are complete
    print(f'Saved to: {model.save_path}')
