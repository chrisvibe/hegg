"""
Gold-standard correctness test: Rule 90 Elementary Cellular Automaton.

Rule 90: state[i, t+1] = state[i-1, t] XOR state[i+1, t]

This is the XOR-neighbor LUT for K=2: [0, 1, 1, 0].
Seeding the center node of an N-node ring with 1 produces a Sierpinski triangle.

The test bypasses the input layer entirely and drives the reservoir via direct
_reservoir_tick() calls. Any bug in the CSR adjacency parsing, LUT indexing,
bit-shift ordering, or double-buffer logic will produce a wrong pattern.

LUT endianness note:
  In reservoir_tick! (Julia), the index is:
      idx = state[adj_list[n,0]] << (max_k-1) | state[adj_list[n,1]] << 0
  We place the left ring neighbor at column 0 (MSB) and the right at column 1 (LSB),
  so idx = left<<1 | right, which is the correct XOR lookup with LUT=[0,1,1,0].
"""

import numpy as np
from juliacall import Main as jl  
from pathlib import Path

from project.boolean_reservoir.code.reservoir import BooleanReservoir
from project.boolean_reservoir.code.utils.utils import set_seed
from project.boolean_reservoir.test.ca.ca_test_utils import make_ca_model, make_ca_input, run_and_plot

CONFIG = Path('config') / 'test' / 'boolean_reservoir' / 'ca' / 'rule90.yaml'

N = 64   # ring size (even → symmetric Sierpinski triangle)
T = 30   # time steps to verify


def install_rule90_structure(model: BooleanReservoir, n_r: int) -> None:
    """Replace the reservoir's graph and LUT with a ring lattice + XOR wiring."""
    # col 0 = left neighbor (MSB), col 1 = right neighbor (LSB): idx = left<<1 | right
    N_I = model.I.n_nodes
    adj_ring, mask_ring = BooleanReservoir.build_ring_lattice(n_r, offsets=(-1, +1))
    adj  = np.zeros((model.N_total, 2), dtype=np.int64)
    mask = np.zeros((model.N_total, 2), dtype=np.bool_)
    adj[N_I:N_I + n_r]  = adj_ring + N_I
    mask[N_I:N_I + n_r] = mask_ring
    xor_lut = np.tile(np.array([0, 1, 1, 0], dtype=np.uint8), (model.N_total, 1))
    model.install_wiring(adj, mask, lut=xor_lut)


def generate_rule90_reference(n: int, steps: int) -> np.ndarray:
    """Pure-Python Rule 90 on an n-node ring; returns (steps+1, n) history."""
    state = np.zeros(n, dtype=np.uint8)
    state[n // 2] = 1
    history = [state.copy()]
    for _ in range(steps):
        state = np.array(
            [state[(i - 1) % n] ^ state[(i + 1) % n] for i in range(n)],
            dtype=np.uint8,
        )
        history.append(state.copy())
    return np.stack(history)   # (steps+1, n)


def _seed_center(model: BooleanReservoir, n_r: int) -> None:
    """Set center R-node to 1 in initial_states and sync backups."""
    model.initial_states[0, model.I.n_nodes + n_r // 2] = 1
    model.reset_reservoir(hard_reset=True)


def test_rule90_eca_gold_standard():
    """
    The reservoir, wired as a 1D ring lattice with the XOR LUT, must reproduce
    the Sierpinski-triangle pattern of Rule 90 exactly at step T.

    n_nodes=0 in the config means no I-nodes exist; forward() runs T purely
    autonomous ticks with no perturbation.  reset=True resets to initial_states
    (center-seeded) at the start of each forward() call.

    Covers: CSR adjacency parsing, Julia 1-indexing conversion, LUT endianness,
    bit-shift ordering, and double-buffer synchronous update correctness.
    """
    model = make_ca_model(CONFIG, install_rule90_structure)
    _seed_center(model, model.R.n_nodes)

    model(np.zeros((1, T)))

    actual   = model.states_parallel[0, model.res_slice]
    expected = generate_rule90_reference(N, T)[-1]

    assert np.array_equal(actual, expected), (
        "Rule 90 mismatch at final step"
    )


# ── Quick-run demo ─────────────────────────────────────────────────────────────

if __name__ == '__main__':
    # Expected: Sierpinski triangle — a fractal of nested triangles radiating
    # downward from the center seed.  The pattern is LEFT-RIGHT SYMMETRIC:
    # because Rule 90 is XOR of two equidistant neighbours and the seed is at
    # the center cell, each column at offset ±k from center evolves identically.
    T_VIZ = 32

    set_seed(0)
    model = make_ca_model(CONFIG, install_rule90_structure)
    _seed_center(model, model.R.n_nodes)

    run_and_plot(model, make_ca_input(np.zeros(T_VIZ, dtype=np.uint8), model.P),
                 'rule90_sierpinski.svg')
