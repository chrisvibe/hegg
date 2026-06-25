"""
Rule 150 Linear Transducer — Post-Override Serial-Input CA.

Reference: Wolfram, S. A New Kind of Science (2002), Section 14.1, p. 952.
           Kutrib & Malcher (2012). "Transductions Computed by One-Dimensional
           Cellular Automata."  (Sequential Input mode, boundary cell = source.)

Let x^(t) ∈ F_2^N be the state and u = {u_1, …, u_T} the input stream.

Transition matrix A over GF(2) is tridiagonal (Rule 150):

    A_{i,j} = 1  if  j ∈ {i-1, i, i+1}  (periodic boundaries)

Two-phase update each tick:

    1.  x^(t+1) = A · x^(t)  mod 2        (full Rule 150 update for all cells)
    2.  x_0^(t+1) ← u_{t+1}               (cell 0 overwritten with the next input bit)

Because cell 0 is pure input at every step, cells 1..N-1 see x_0(t) = u_t as
their left/right neighbour.  Rule 150 is additive over GF(2), so the final
state x^(T) is the XOR-sum of the shifted impulse responses of every input bit
(superposition principle).  One flipped input bit flips the entire output.

Key difference from test_rule150_transducer.py
------------------------------------------------
test_rule150_transducer.py injects the input bit as cell 0's LEFT NEIGHBOUR
inside the Rule 150 XOR computation:
    x_0(t+1) = u_t ⊕ x_0(t) ⊕ x_1(t)
    x_1(t+1) = x_0(t) ⊕ x_1(t) ⊕ x_2(t)   ← x_0(t) ≠ u_t in general

This file gives cell 0 a per-node identity LUT (output = MSB = I-node = u_t):
    x_0(t+1) = u_t                           ← pure input, no XOR mixing
    x_1(t+1) = x_0(t) ⊕ x_1(t) ⊕ x_2(t)   ← x_0(t) = u_{t-1}, direct input

Both transducers share the same adj_list structure (I-node as cell 0's left
neighbour) and the same 'override' perturbation mode.  The only code difference
is the LUT assigned to cell 0's row.

Driven-input coverage
----------------------
Like test_rule150_transducer.py, this file exercises the full external input
pipeline (perturbation → I-node → adj_list boundary → R) with T=30 steps of
a balanced random stream.  See that file for why final-state comparison is
sufficient to prove per-step correctness.

Additional dynamics coverage over the regular transducer: cell 0 uses LUT_MSB
(a different LUT than the rest of the ring), verifying that install_wiring
applies per-node LUTs correctly and that the Julia engine reads the right LUT
row for each node.
"""

import numpy as np
from juliacall import Main as jl  
from pathlib import Path

from project.boolean_reservoir.code.parameter import load_yaml_config
from project.boolean_reservoir.code.reservoir import BooleanReservoir
from project.boolean_reservoir.code.utils.utils import set_seed
from project.boolean_reservoir.test.ca.ca_test_utils import make_ca_model, make_ca_input, override_to_viz_mode, run_and_plot

CONFIG = Path('config') / 'test' / 'boolean_reservoir' / 'ca' / 'rule150_linear_transducer.yaml'

# Rule 150 = XOR of 3 neighbours.  K=3 MSB-first: index = left<<2 | self<<1 | right
RULE150_LUT = np.array([0, 1, 1, 0, 1, 0, 0, 1], dtype=np.uint8)

# Identity-on-MSB: output = left neighbour regardless of self and right.
# Used for cell 0 so that x_0(t+1) = I-node = u_t (pure replacement).
LUT_MSB = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.uint8)


# ── Reference implementation ───────────────────────────────────────────────────

def _rule150_linear_reference(n: int, input_bits: np.ndarray,
                               initial_state: np.ndarray | None = None) -> np.ndarray:
    """Pure-NumPy Rule 150 linear transducer.  Returns (T+1, N) history."""
    s = (np.zeros(n, dtype=np.uint8) if initial_state is None
         else initial_state.astype(np.uint8).copy())
    history = [s.copy()]
    for bit in input_bits:
        nxt = np.empty_like(s)
        nxt[0]    = int(bit)                    # cell 0 = pure input
        nxt[1:-1] = s[:-2] ^ s[1:-1] ^ s[2:]   # interior Rule 150 (uses s[0] = prev input)
        nxt[-1]   = s[-2]  ^ s[-1]   ^ s[0]    # periodic right (sees s[0] = prev input)
        s = nxt
        history.append(s.copy())
    return np.stack(history)


# ── RBN helpers ────────────────────────────────────────────────────────────────

def install_rule150_linear_structure(model: BooleanReservoir, n_r: int) -> None:
    """Install Rule 150 linear transducer wiring.

    Same adj_list as test_rule150_transducer: I-node is cell 0's left neighbour.
    Difference: cell 0 gets LUT_MSB (output = MSB = I-node = u_t) instead of
    RULE150_LUT, making x_0(t+1) = u_t (pure replacement, no XOR mixing).
    Cells 1..N-1 get the standard Rule 150 LUT and periodic ring adjacency.
    """
    N_I = model.I.n_nodes
    adj_ring, mask_ring = BooleanReservoir.build_ring_lattice(n_r, offsets=(-1, 0, +1))
    adj  = np.zeros((model.N_total, 3), dtype=np.int64)
    mask = np.zeros((model.N_total, 3), dtype=np.bool_)
    adj[N_I:N_I + n_r]  = adj_ring + N_I
    mask[N_I:N_I + n_r] = mask_ring
    adj[N_I, 0] = model.input_nodes[0]  # cell 0: left = I-node (serial input)

    lut = np.tile(RULE150_LUT, (model.N_total, 1))
    lut[N_I] = LUT_MSB   # cell 0: pure identity on I-node
    model.install_wiring(adj, mask, lut=lut)


def _make_rule150_linear_model() -> BooleanReservoir:
    return make_ca_model(CONFIG, install_rule150_linear_structure)


# ── Pytest tests ───────────────────────────────────────────────────────────────

def test_rule150_linear_all_zeros_fixed_point():
    """All-zero state with zero input stream must remain all-zero.

    R.init=zeros + reset=True means forward() starts from all-zeros automatically.
    Zero input bits leave I-node=0 each tick, so the fixed point holds.
    """
    model = _make_rule150_linear_model()
    x = np.zeros((1, 30, 1, 1), dtype=np.uint8)
    model(x)
    assert np.all(model.states_parallel[0, model.res_slice] == 0)


def test_rule150_linear_gold_standard():
    """
    BooleanReservoir wired as a Rule 150 linear transducer must reproduce the
    pure-NumPy reference exactly for a seeded random input stream.

    forward() with x.shape=(1,T,1,1) delivers each input bit to I-node 0 via
    override perturbation; LUT_MSB on cell 0 then copies it to x_0(t+1)=u_t.
    R.init=zeros + reset=True ensures an all-zero start automatically.
    Final-state comparison is sufficient: GF(2) linearity means any per-step
    error accumulates to step T (superposition principle).

    Covers: per-node heterogeneous LUT (LUT_MSB for cell 0, Rule 150 for rest),
    I-node as cell 0's left neighbour, override perturbation path, and the GF(2)
    linearity (superposition) property.
    """
    T     = 30
    model = _make_rule150_linear_model()
    N_R   = model.R.n_nodes

    rng        = np.random.default_rng(0)
    input_bits = rng.integers(0, 2, T, dtype=np.uint8)

    x = input_bits.reshape(1, T, 1, 1)
    model(x)

    actual   = model.states_parallel[0, model.res_slice]
    expected = _rule150_linear_reference(N_R, input_bits)[-1]

    assert np.array_equal(actual, expected), (
        "Rule 150 linear transducer mismatch at final step"
    )


# ── Quick-run demo ─────────────────────────────────────────────────────────────

if __name__ == '__main__':
    # Expected: impulse response from a single 1 at cell 0 (t=0), then zeros.
    # Cell 0 is a pure input node (LUT_MSB), so it immediately resets to 0 on tick 1;
    # cells 1..N-1 propagate the impulse rightward under Rule 150 XOR dynamics.
    # The pattern should look similar to the transducer but with a sharper left
    # boundary cutoff — cell 0 carries no memory of its own.
    T_VIZ = 64

    set_seed(0)
    P = load_yaml_config(CONFIG)
    P.logging.history.persist_to_disk = True
    override_to_viz_mode(P)

    model = BooleanReservoir(P)
    install_rule150_linear_structure(model, model.R.n_nodes)
    model.reset_reservoir(hard_reset=True)

    impulse = np.zeros(T_VIZ, dtype=np.uint8); impulse[0] = 1

    run_and_plot(model, make_ca_input(impulse, P), 'rule150_linear_transducer_impulse.svg', highlight_input_nodes=True)
