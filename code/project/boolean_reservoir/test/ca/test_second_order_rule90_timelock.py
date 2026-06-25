"""
Second-Order Rule 90 — Reversible Time-Lock Test.

Second-order (Toffoli-Margolus) Rule 90:
  x_i(t+1) = x_{i-1}(t) ⊕ x_{i+1}(t) ⊕ x_i(t-1)

This rule is always invertible for any N — the inverse is the same formula:
  x_i(t-1) = x_{i-1}(t) ⊕ x_{i+1}(t) ⊕ x_i(t+1)

Reference: Toffoli & Margolus (1987) "Cellular Automata Machines," MIT Press, Ch. 6.
           https://mitpress.mit.edu/9780262200608/cellular-automata-machines/

Doubled-state implementation (2N = 128 R-nodes for N = 64):
  Nodes   0..N-1   — current  layer x_i(t)
  Nodes   N..2N-1  — previous layer x_i(t-1)

Forward tick (lock direction):
  new_curr[i]  = left_curr ⊕ right_curr ⊕ self_prev   K=3, XOR3_LUT
  new_prev[i]  = curr[i]                               K=1, LUT_MSB (identity via MSB)

Backward tick (unlock direction):
  new_curr[i]  = prev[i]                               K=1, LUT_MSB
  new_prev[i]  = left_prev ⊕ right_prev ⊕ self_curr   K=3, XOR3_LUT

LUT encoding (MSB-first, max_k = 3):
  The Julia engine builds idx = state[col0]<<2 | state[col1]<<1 | state[col2]<<0.
  For K=1 nodes (only col 0 masked in), col0 gets shift max_k-1 = 2, so
  idx ∈ {0, 4} — making LUT_MSB = [0,0,0,0,1,1,1,1] an identity lookup.

Time-lock structure:
  Lock:   T forward steps via BooleanReservoir with key stream  u_0 … u_{T-1}
  Unlock: T inverse  steps via pure-Python reference with reversed key u_{T-1} … u_0
  The round-trip assertion follows from mathematical invertibility alone — no
  independent reference is needed.  Any forward error (wrong LUT, wiring, I-node
  override, or double-buffer) propagates over all T steps and prevents recovery.

This is a stronger test than rule90_eca or rule150_transducer because:
  - It exercises heterogeneous K (K=3 current layer, K=1 previous layer) in one model.
  - Invertibility catches errors that a single final-state comparison might miss.
  - The key stream exercises the override perturbation path on every step.
"""

import numpy as np
from juliacall import Main as jl  
from pathlib import Path

from project.boolean_reservoir.code.parameter import load_yaml_config
from project.boolean_reservoir.code.reservoir import BooleanReservoir
from project.boolean_reservoir.code.utils.utils import set_seed
from project.boolean_reservoir.test.ca.ca_test_utils import make_ca_model, make_ca_input, run_and_plot

CONFIG = Path('config') / 'test' / 'boolean_reservoir' / 'ca' / 'second_order_rule90.yaml'

N      = 32        # visible ring size; reservoir has 2*N = 64 R-nodes (same as other CA tests)
CENTER = N // 2    # = 16 — the cell driven by the I-node (key injection point)

# XOR of 3 inputs, MSB-first: idx = a<<2 | b<<1 | c<<0
XOR3_LUT = np.array([0, 1, 1, 0, 1, 0, 0, 1], dtype=np.uint8)

# Identity for a K=1 node in a max_k=3 framework.
# col 0 masked in → shift = max_k-1 = 2, so idx = state<<2 ∈ {0,4}.
LUT_MSB = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.uint8)


# ── Reference (pure-NumPy) ─────────────────────────────────────────────────────

def _second_order_rule90_unlock(locked: np.ndarray, key_reversed: np.ndarray) -> tuple:
    """Inverse of T forward steps: recovers (x(0), x(-1)) from locked state.

    locked:       1-D array of length 2N — [x(T) | x(T-1)]
    key_reversed: key stream in reverse order, shape (T,)
    Returns: (x_rec, xp_rec) — should equal (x0, x_minus1)
    """
    x, xp = locked[:N].copy(), locked[N:].copy()
    for u in key_reversed:
        left        = np.roll(xp, 1)   # left[i]  = xp[(i-1)%N]
        left[CENTER] = u
        new_xp      = left ^ np.roll(xp, -1) ^ x
        x, xp       = xp, new_xp
    return x, xp


# ── Custom wiring ──────────────────────────────────────────────────────────────

def _update_ir_graph_edge(model: BooleanReservoir, r_global: int) -> None:
    """Replace the patchwork I→R edges in model.graph with the actual injection edge.

    install_wiring replaces adj_list/lut but leaves model.graph unchanged.  The
    patchwork graph has 'identity' I→R (I-node 0 → R-node 0), which is wrong after
    custom wiring.  This fix makes the highlight_input_nodes subtitle show the
    correct injection point.  Must be called after install_wiring.
    """
    g = model.graph
    old_ir = [(u, v) for u, v, d in g.edges(data=True) if d.get('quadrant') == 'IR']
    g.remove_edges_from(old_ir)
    g.add_edge(0, r_global)
    model.add_graph_labels(g)


def _alloc_so_adj(model: BooleanReservoir):
    """Allocate zeroed (N_total, 3) adj_list and mask for the doubled-state ring."""
    nt = model.N_total
    return np.zeros((nt, 3), dtype=np.int64), np.zeros((nt, 3), dtype=np.bool_)


def install_second_order_rule90_forward(model: BooleanReservoir, n_r: int) -> None:
    """Install forward (lock) wiring: x(t+1) = left_curr ⊕ right_curr ⊕ self_prev.

    Current layer i (global N_I+i):     K=3 [left_curr, right_curr, self_prev] → XOR3
      Center cell: col 0 overridden to I-node 0 (key bit replaces left_curr).
    Previous layer i (global N_I+N+i):  K=1 [self_curr] → LUT_MSB (identity)
    """
    N_I = model.I.n_nodes
    adj, mask = _alloc_so_adj(model)
    lut = np.zeros((model.N_total, 8), dtype=np.uint8)

    for i in range(N):
        gc = N_I + i          # current layer global index
        gp = N_I + N + i      # previous layer global index

        adj[gc, 0]  = N_I + (i - 1) % N   # left_curr  (MSB, shift 2)
        adj[gc, 1]  = N_I + (i + 1) % N   # right_curr (shift 1)
        adj[gc, 2]  = N_I + N + i         # self_prev  (LSB, shift 0)
        mask[gc]    = True
        lut[gc]     = XOR3_LUT

        adj[gp, 0]  = N_I + i             # self_curr (shift 2 in max_k=3)
        mask[gp, 0] = True
        lut[gp]     = LUT_MSB

    adj[N_I + CENTER, 0] = model.input_nodes[0]   # I-node drives center's left_curr
    model.install_wiring(adj, mask, lut=lut)
    _update_ir_graph_edge(model, N_I + CENTER)   # graph IR edge → correct highlight title


def install_second_order_rule90_backward(model: BooleanReservoir, n_r: int) -> None:
    """Install backward (unlock) wiring: new_prev[i] = left_prev ⊕ right_prev ⊕ self_curr.

    Current layer i (global N_I+i):     K=1 [self_prev] → LUT_MSB (identity)
    Previous layer i (global N_I+N+i):  K=3 [left_prev, right_prev, self_curr] → XOR3
      Center cell: col 0 of previous layer overridden to I-node 0 (reversed key bit).
    """
    N_I = model.I.n_nodes
    adj, mask = _alloc_so_adj(model)
    lut = np.zeros((model.N_total, 8), dtype=np.uint8)

    for i in range(N):
        gc = N_I + i
        gp = N_I + N + i

        adj[gc, 0]  = N_I + N + i         # self_prev (shift 2 in max_k=3)
        mask[gc, 0] = True
        lut[gc]     = LUT_MSB

        adj[gp, 0]  = N_I + N + (i - 1) % N  # left_prev  (previous layer, MSB, shift 2)
        adj[gp, 1]  = N_I + N + (i + 1) % N  # right_prev (previous layer, shift 1)
        adj[gp, 2]  = N_I + i                 # self_curr  (current layer, LSB, shift 0)
        mask[gp]    = True
        lut[gp]     = XOR3_LUT

    adj[N_I + N + CENTER, 0] = model.input_nodes[0]   # I-node drives center's left_prev (backward)
    model.install_wiring(adj, mask, lut=lut)
    _update_ir_graph_edge(model, N_I + N + CENTER)  # graph IR edge → correct highlight title


def _make_second_order_model() -> BooleanReservoir:
    return make_ca_model(CONFIG, install_second_order_rule90_forward)


# ── Pytest tests ───────────────────────────────────────────────────────────────

def test_second_order_rule90_zeros_fixedpoint():
    """All-zero doubled state with zero key must remain all-zero.

    x(t+1)[i] = 0⊕0⊕0 = 0; previous layer copies 0.  R.init=zeros + reset=True
    starts from all-zeros automatically — no manual state setup required.
    """
    model = _make_second_order_model()
    x = make_ca_input(np.zeros(30, dtype=np.uint8), model.P)
    model(x)
    assert np.all(model.states_parallel[0, model.res_slice] == 0)


def test_second_order_rule90_timelock():
    """
    BooleanReservoir, forward-wired as second-order Rule 90, must recover the
    initial state after T forward steps (lock) + pure-Python inverse with reversed
    key (unlock).

    Time-lock structure:
      x0 = random initial state (current layer),  x_minus1 = zeros (previous layer)
      Lock:   T forward steps via BooleanReservoir with seeded key u_0…u_{T-1}
      Unlock: pure-Python inverse with reversed key u_{T-1}…u_0
      Assert: recovered current layer == x0  AND  recovered prev layer == zeros

    Invertibility of second-order Rule 90 is a mathematical guarantee for any N.
    Any error in the BooleanReservoir forward pass — wrong LUT, wiring, I-node
    override perturbation, or double-buffer synchrony — accumulates over T steps
    and prevents recovery.

    Covers: K=3 XOR3 LUT, K=1 LUT_MSB (identity), heterogeneous K in one model,
    I-node override at center cell for the current layer, ring-boundary wiring for
    the 2N doubled-state reservoir, and the Julia double-buffer synchronous update.
    """
    T   = 40
    rng = np.random.default_rng(42)
    x0       = rng.integers(0, 2, N, dtype=np.uint8)
    key_bits = rng.integers(0, 2, T, dtype=np.uint8)
    x_minus1 = np.zeros(N, dtype=np.uint8)

    model = _make_second_order_model()
    N_I   = model.I.n_nodes
    model.initial_states[0, N_I:N_I + N] = x0   # current layer ← x0
    # previous layer stays zero (R.init=zeros default)
    model.reset_reservoir(hard_reset=True)

    model(make_ca_input(key_bits, model.P))

    locked = model.states_parallel[0, model.res_slice].copy()   # shape (2N,)

    x_rec, xp_rec = _second_order_rule90_unlock(locked, key_bits[::-1])

    assert np.array_equal(x_rec, x0), (
        "Second-order Rule 90 time-lock: current layer not recovered after round-trip"
    )
    assert np.array_equal(xp_rec, x_minus1), (
        "Second-order Rule 90 time-lock: previous layer (zeros) not recovered"
    )


def test_second_order_rule90_backward_wiring():
    """
    BooleanReservoir with backward wiring must recover x0 exactly.

    This test exercises install_second_order_rule90_backward directly — the wiring
    used in the __main__ visualization to produce the unlock half of the SVG.
    It mirrors test_second_order_rule90_timelock but runs the backward pass through
    BooleanReservoir rather than the pure-Python reference, confirming that the
    start and end states in the SVG will match.

    Sequence:
      1. Forward  T steps with BooleanReservoir (forward wiring) → locked
      2. Backward T steps with BooleanReservoir (backward wiring) → recovered
      3. Assert: recovered current layer == x0, recovered previous == zeros
    """
    T   = 40
    rng = np.random.default_rng(42)
    x0       = rng.integers(0, 2, N, dtype=np.uint8)
    key_bits = rng.integers(0, 2, T, dtype=np.uint8)

    # --- Forward (lock) ---
    model = _make_second_order_model()
    N_I   = model.I.n_nodes
    model.initial_states[0, N_I:N_I + N] = x0
    model.reset_reservoir(hard_reset=True)

    model(make_ca_input(key_bits, model.P))
    locked = model.states_parallel[0, model.res_slice].copy()

    # --- Backward (unlock) via BooleanReservoir ---
    install_second_order_rule90_backward(model, model.R.n_nodes)
    model.initial_states[0, N_I:N_I + 2 * N] = locked
    model.reset_reservoir(hard_reset=True)

    model(make_ca_input(key_bits[::-1].copy(), model.P))

    recovered = model.states_parallel[0, model.res_slice]
    assert np.array_equal(recovered[:N], x0), (
        "Backward BooleanReservoir wiring: current layer not recovered"
    )
    assert np.array_equal(recovered[N:], np.zeros(N, dtype=np.uint8)), (
        "Backward BooleanReservoir wiring: previous layer (zeros) not recovered"
    )


# ── Quick-run demo ─────────────────────────────────────────────────────────────

if __name__ == '__main__':
    # Produces 3 canonical runs with different key streams (key_seed 0, 1, 2).
    # Each run: forward lock (T steps) then backward unlock (T steps).
    # The unlock always recovers the original state regardless of key.
    # All runs share out_path so the dashboard finds them via _find_run_dirs.
    # Delete out/test/ca/second_order_rule90/ and re-run to reproduce all three.
    import time
    T_VIZ = 48

    for key_seed in (0, 1, 2):
        set_seed(0)
        P = load_yaml_config(CONFIG)
        P.logging.history.persist_to_disk = True

        model = BooleanReservoir(P)
        install_second_order_rule90_forward(model, model.R.n_nodes)

        rng      = np.random.default_rng(key_seed)
        x0       = np.zeros(N, dtype=np.uint8)
        x0[N // 2] = 1
        key_bits = rng.integers(0, 2, T_VIZ, dtype=np.uint8)

        N_I = model.I.n_nodes
        model.initial_states[0, N_I:N_I + N] = x0
        model.reset_reservoir(hard_reset=True)

        # Lock: T_VIZ forward steps
        model.eval()
        model(make_ca_input(key_bits, model.P))
        model.flush_history()

        locked = model.states_parallel[0, N_I:N_I + 2 * N].copy()

        # Unlock: T_VIZ backward steps (appended to same history dir)
        install_second_order_rule90_backward(model, model.R.n_nodes)
        model.initial_states[0, N_I:N_I + 2 * N] = locked
        model.reset_reservoir(hard_reset=True)

        model(make_ca_input(key_bits[::-1].copy(), model.P))
        model.flush_history()

        model.save()
        print(f"key_seed={key_seed} written to {model.save_path}")
        time.sleep(0.01)  # ensure unique timestamp for next run
