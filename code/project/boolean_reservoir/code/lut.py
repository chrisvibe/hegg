import numpy as np

def calc_lut_p(lut: np.ndarray) -> float:
    """Realized fraction of 1-entries across all LUT rows."""
    return float(lut.mean())

def lut_random(n_nodes, max_incoming_edges, p=0.5):
    # make a independent lut for each node
    # lut[0] is the lut for node 0
    # let idx represent the state of the reservoir: lut[0][idx] is the next state of node 0 with probability p
    # Uses numpy RNG (not torch) so output is identical across CPU and GPU given the same seed.
    assert 0 <= p <= 1
    lut = np.random.rand(n_nodes, 2 ** max_incoming_edges) < p
    return lut.astype(np.uint8)  # np.ndarray; caller bridges to torch

def lut_index(pred_states: list, max_k: int) -> int:
    """MSB-first binary index from predecessor states.

    The first predecessor occupies the most-significant bit (2^(max_k-1)).
    This matches the Julia engine's bin2int with powers_of_2 = [2^(max_k-1), ..., 2^0].
    """
    k = len(pred_states)
    return sum(int(pred_states[i]) * (2 ** (max_k - 1 - i)) for i in range(k))


def lut_lookup(lut: np.ndarray, node: int, pred_states: list, max_k: int) -> int:
    """Return the LUT output for a node given its predecessor states."""
    return int(lut[node, lut_index(pred_states, max_k)])


# TODO add alternative with exactly pxN entries like the graphs atm which generate E edges then shuffle them, not like flipping a coin for each edge