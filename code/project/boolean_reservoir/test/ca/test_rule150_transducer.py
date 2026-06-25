"""
Rule 150 Transducer — Boundary-Driven Serial-Input Cellular Automaton.

This test bridges Rule 90 (autonomous ring dynamics, fixed initial state) with
serial-input reservoir computing.  At every tick a new bit is injected at the
left boundary: cell 0's left neighbour is the I-node rather than R-node N-1.
All other cells use the standard periodic ring.  This makes the reservoir a
temporal transducer — the input bit stream is XOR-mixed into the ring state
step by step, analogous to a shift register with XOR feedback.

Rule 150: S_i(t+1) = S_{i-1}(t) ⊕ S_i(t) ⊕ S_{i+1}(t)   (XOR of 3 neighbours)
Reference: Wolfram, S.  A New Kind of Science (2002), p. 886.

LUT encoding — K=3 columns, MSB-first:
  col  neighbour   bit-weight
   0   left        << 2   (MSB)
   1   self        << 1
   2   right       << 0   (LSB)

index = (left<<2)|(self<<1)|right

Heterogeneous topology:
  cell 0     : left = I-node 0  (serial input),  right = R-node 1   (ring)
  cell i > 0 : left = R-node i-1 (ring),          right = R-node (i+1)%N (periodic)

Why this is the primary driven-input correctness test
------------------------------------------------------
Rule 90 and GKL are autonomous: after install_*_structure the I-nodes have no
connections to R, so the external perturbation path is never exercised.  This
file is the only test that drives the full input pipeline end-to-end:

  external bit stream → override perturbation → I-node 0 → adj_list boundary → R

test_rule150_transducer_gold_standard uses T=30 steps of a balanced seeded
random stream (~50/50 bits).  Rule 150 is additive over GF(2), so any error in
the perturbation mode, the I-node read, or the adj_list boundary connection
accumulates across steps and produces a wrong final state.  Final-state equality
at step 30 therefore implies the driven input path is correct at every step.
"""

import numpy as np
from juliacall import Main as jl  
from pathlib import Path

from project.boolean_reservoir.code.parameter import load_yaml_config, InputParams
from project.boolean_reservoir.code.reservoir import BooleanReservoir
from project.boolean_reservoir.code.utils.utils import set_seed
from project.boolean_reservoir.test.ca.ca_test_utils import make_ca_model, make_ca_input, override_to_viz_mode, run_and_plot

CONFIG = Path('config') / 'test' / 'boolean_reservoir' / 'ca' / 'rule150_transducer.yaml'

# Rule 150 = XOR of 3 neighbours.  Binary: 10010110 = 150.
# K=3 MSB-first: index = left<<2 | self<<1 | right
RULE150_LUT = np.array([0, 1, 1, 0, 1, 0, 0, 1], dtype=np.uint8)


# ── Reference implementation ───────────────────────────────────────────────────

def _rule150_transducer_reference(n: int, input_bits: np.ndarray,
                                   initial_state: np.ndarray | None = None) -> np.ndarray:
    """Pure-NumPy Rule 150 transducer.  Returns (T+1, N) history (t=0 first row)."""
    s = (np.zeros(n, dtype=np.uint8) if initial_state is None
         else initial_state.astype(np.uint8).copy())
    history = [s.copy()]
    for bit in input_bits:
        nxt = np.empty_like(s)
        nxt[0]    = int(bit) ^ s[0] ^ s[1]      # left boundary: serial input replaces ring wrap
        nxt[1:-1] = s[:-2]  ^ s[1:-1] ^ s[2:]  # interior: vectorised XOR
        nxt[-1]   = s[-2]   ^ s[-1]   ^ s[0]   # right boundary: periodic
        s = nxt
        history.append(s.copy())
    return np.stack(history)


# ── RBN helpers ────────────────────────────────────────────────────────────────

def install_rule150_transducer_structure(model: BooleanReservoir, n_r: int) -> None:
    """Install Rule 150 wiring with serial input at evenly spaced ring positions.

    Each I-node is wired as the left neighbour of an evenly spaced R-node so
    that I.n_nodes controls the number of injection points automatically:
      I.n_nodes=1 (redundancy=1): I-node 0 → cell 0 only
      I.n_nodes=2 (redundancy=2): I-node 0 → cell 0, I-node 1 → cell N//2
    With override perturbation all I-nodes carry the same input bit u_t.
    """
    N_I = model.I.n_nodes
    adj_ring, mask_ring = BooleanReservoir.build_ring_lattice(n_r, offsets=(-1, 0, +1))
    adj  = np.zeros((model.N_total, 3), dtype=np.int64)
    mask = np.zeros((model.N_total, 3), dtype=np.bool_)
    adj[N_I:N_I + n_r]  = adj_ring + N_I
    mask[N_I:N_I + n_r] = mask_ring
    spacing = n_r // N_I
    for i in range(N_I):
        adj[N_I + i * spacing, 0] = model.input_nodes[i]
    model.install_wiring(adj, mask, lut=np.tile(RULE150_LUT, (model.N_total, 1)))


def _make_rule150_model() -> BooleanReservoir:
    return make_ca_model(CONFIG, install_rule150_transducer_structure)


# ── Pytest tests ───────────────────────────────────────────────────────────────

def test_rule150_transducer_all_zeros_fixed_point():
    """All-zero state with zero input must remain all-zero (0⊕0⊕0 = 0 everywhere).

    R.init=zeros + reset=True means forward() starts from all-zeros automatically.
    Zero input bits leave I-node=0 each tick, so the fixed point holds.
    """
    model = _make_rule150_model()
    x = np.zeros((1, 30, 1, 1), dtype=np.uint8)
    model(x)
    assert np.all(model.states_parallel[0, model.res_slice] == 0)


def test_rule150_transducer_gold_standard():
    """
    BooleanReservoir wired as a Rule 150 transducer must reproduce the reference
    exactly for a seeded random input stream.

    forward() with x.shape=(1,T,1,1) delivers each input bit to I-node 0 via
    override perturbation before each tick — exactly the serial-input transducer
    semantics.  R.init=zeros + reset=True ensures an all-zero start without
    manual state manipulation.  Final-state comparison is sufficient: GF(2)
    linearity means any per-step error accumulates to step T.

    Covers: K=3 LUT correctness, heterogeneous boundary (I-node replaces ring
    wrap for cell 0), build_ring_lattice + single-entry override, override
    perturbation path, and the Julia engine reading the I-node state during tick.
    """
    T  = 30
    model = _make_rule150_model()
    N_R   = model.R.n_nodes

    rng        = np.random.default_rng(0)
    input_bits = rng.integers(0, 2, T, dtype=np.uint8)

    x = input_bits.reshape(1, T, 1, 1)
    model(x)

    actual   = model.states_parallel[0, model.res_slice]
    expected = _rule150_transducer_reference(N_R, input_bits)[-1]

    assert np.array_equal(actual, expected), (
        "Rule 150 transducer mismatch at final step"
    )


# ── Quick-run demo ─────────────────────────────────────────────────────────────

if __name__ == '__main__':
    T_VIZ = 64   # each forward step = 1 tick

    set_seed(0)
    P = load_yaml_config(CONFIG)
    P.logging.history.persist_to_disk = True
    override_to_viz_mode(P)

    model = BooleanReservoir(P)
    install_rule150_transducer_structure(model, model.R.n_nodes)

    rng = np.random.default_rng(0)
    impulse = np.zeros(T_VIZ, dtype=np.uint8); impulse[0] = 1
    random_bits = rng.integers(0, 2, T_VIZ, dtype=np.uint8)

    # Plot 1 — impulse response (Green's function).
    # Expected: single activation at cell 0 spreads rightward as a diagonal stripe;
    # cell 0 resets each tick as it receives zero input.
    run_and_plot(model, make_ca_input(impulse, P), 'rule150_transducer_impulse.svg', highlight_input_nodes=True)

    # Plot 2 — balanced random input stream.
    # Expected: complex space-time texture; no single stripe, patterns shift and
    # interfere as each new bit is injected at cell 0.
    model.init_logging()
    run_and_plot(model, make_ca_input(random_bits, P), 'rule150_transducer_random.svg', highlight_input_nodes=True)

    # Plots 3 & 4 — redundancy=2: same bit u_t injected at cell 0 AND cell N//2.
    # install_rule150_transducer_structure spaces injection points evenly by I.n_nodes,
    # so redundancy=2 → I.n_nodes=2 → two equidistant injection points automatically.
    # make_ca_input handles the tiling via BooleanTransformer — no manual np.tile.
    P.M.input_layer = InputParams.model_validate({
        **P.M.I.model_dump(),
        'redundancy': 2,
        'bits': None,       # recomputed: features * resolution * redundancy = 1*1*2 = 2
        'n_nodes': None,    # recomputed: bits = 2
        'chunk_size': None, # recomputed: bits // chunks = 2//1 = 2
    })
    model_dual = BooleanReservoir(P)
    install_rule150_transducer_structure(model_dual, model_dual.R.n_nodes)

    # Plot 3 — dual impulse.
    # Expected: two symmetric wavefronts spreading inward from both injection sites.
    run_and_plot(model_dual, make_ca_input(impulse, P), 'rule150_transducer_dual_impulse.svg', highlight_input_nodes=True)

    # Plot 4 — dual random.
    # Expected: richer interference texture than plot 2; two driven boundary points
    # produce overlapping wavefronts that meet and cancel/reinforce mid-ring.
    # Pattern is LEFT-RIGHT SYMMETRIC around cell N//2: both injection sites
    # receive the identical bit stream u_t, so the ring is driven symmetrically.
    model_dual.init_logging()
    run_and_plot(model_dual, make_ca_input(random_bits, P), 'rule150_transducer_dual_random.svg', highlight_input_nodes=True)
