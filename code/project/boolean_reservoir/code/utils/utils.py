import numpy as np
import random
from hashlib import sha256
from pathlib import Path
import networkx as nx
import time
import socket
import logging
import signal
import sys
from os import getpid, replace, environ

logger = logging.getLogger(__name__)

def configure_logging(level=logging.INFO):
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(process)d - %(filename)s - %(message)s',
        stream=sys.stdout,
        force=True,
    )
    logging.Formatter.converter = time.gmtime

def set_seed(seed=42):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

def generate_unique_seed(*args):
    combined_str = ','.join(map(str, args))
    hash_digest = sha256(combined_str.encode()).hexdigest()
    return int(hash_digest, 16) % (2**31 - 1)

def print_pretty_binary_matrix(data, input_nodes=None, reservoir_nodes=None, print_str=True, return_str=False):
    QUADRANT_COLORS = {
        'II': '\033[95m',
        'IR': '\033[93m',
        'RI': '\033[91m',
        'RR': '\033[92m',
        'default': '\033[94m',
    }

    def infer_quadrant(u, v):
        if u in input_nodes and v in input_nodes:
            return 'II'
        elif u in input_nodes and v in reservoir_nodes:
            return 'IR'
        elif u in reservoir_nodes and v in input_nodes:
            return 'RI'
        elif u in reservoir_nodes and v in reservoir_nodes:
            return 'RR'
        return 'default'

    if isinstance(data, np.ndarray):
        array = (data != 0).astype(int)
        color_matrix = np.full(array.shape, 'default', dtype=object)

    elif isinstance(data, (nx.Graph, nx.DiGraph)):
        nodes = list(data.nodes())
        node_index = {node: i for i, node in enumerate(nodes)}
        array = nx.to_numpy_array(data, nodelist=nodes, dtype=bool).astype(int)
        color_matrix = np.full(array.shape, 'default', dtype=object)

        for u, v, attrs in data.edges(data=True):
            i, j = node_index[u], node_index[v]
            if 'quadrant' in attrs:
                q = attrs['quadrant']
            elif input_nodes is not None and reservoir_nodes is not None:
                q = infer_quadrant(u, v)
            else:
                q = 'default'
            color_matrix[i, j] = q
            if not data.is_directed():
                color_matrix[j, i] = q
    else:
        raise TypeError("Input must be a np.ndarray or networkx Graph/DiGraph.")

    lines = []
    for i in range(array.shape[0]):
        row = ''
        for j in range(array.shape[1]):
            val = str(array[i, j])
            color = QUADRANT_COLORS.get(color_matrix[i, j], QUADRANT_COLORS['default'])
            row += f"{color}{val}\033[0m"
        lines.append(row)

    result = '\n'.join(lines)
    if print_str:
        print(result)
    if return_str:
        return result

def override_symlink(source: Path, link: Path = None):
    if link is None:
        link = Path(source.name)
    try:
        temp_link = Path(f"{link}.tmp.{getpid()}.{time.time_ns()}")
        temp_link.symlink_to(source)
        replace(str(temp_link), str(link))
    except Exception:
        if 'temp_link' in locals() and temp_link.exists():
            try:
                temp_link.unlink()
            except Exception:
                pass

def l2_distance(arr):
    return np.sqrt((arr ** 2).sum(axis=1))

def l2_distance_squared(arr):
    return (arr ** 2).sum(axis=1)

def manhattan_distance(arr):
    return np.abs(arr).sum(axis=1)

def balance_dataset(dataset, num_bins=100, distance_fn=l2_distance, labels_are_classes=False,
                    target_mode='samples_over_bins', verbose=False):
    x = dataset.data['x']
    y = dataset.data['y']

    distances = distance_fn(y)
    bins = np.linspace(distances.min(), distances.max(), num_bins + 1)
    if labels_are_classes:
        bin_indices = y.ravel().astype(int)
    else:
        bin_indices = np.digitize(distances, bins) - 1  # 0-indexed

    if target_mode == 'samples_over_bins':
        target_per_bin = max(len(y) // num_bins, 1)
    elif target_mode == 'minimum_bin':
        target_per_bin = int(np.bincount(y.ravel().astype(int)).min())

    balanced_indices = []
    for i in range(num_bins):
        bin_samples = np.where(bin_indices == i)[0]
        if len(bin_samples) > 0:
            if len(bin_samples) <= target_per_bin:
                balanced_indices.append(bin_samples)
            else:
                sampled = bin_samples[np.random.permutation(len(bin_samples))[:target_per_bin]]
                balanced_indices.append(sampled)

    balanced_indices = np.concatenate(balanced_indices) if balanced_indices else np.arange(len(y))

    if verbose:
        if labels_are_classes:
            print("Class distribution:")
            unique_labels = np.unique(y)
            for label in unique_labels:
                before_count = (y == label).sum()
                after_count = (y[balanced_indices] == label).sum()
                print(f" Class {int(label)}: {before_count} → {after_count} samples")
        else:
            print("Value distribution (quartiles):")
            for q in [0, 25, 50, 75, 100]:
                before = np.percentile(y, q)
                after  = np.percentile(y[balanced_indices], q)
                print(f" {q}%: {before:.4f} → {after:.4f}")

    balanced_indices = balanced_indices[np.random.permutation(len(balanced_indices))]
    dataset.set_data({'x': x[balanced_indices], 'y': y[balanced_indices]})

    if verbose:
        n_before, n_after = len(x), len(balanced_indices)
        print(f'Balanced dataset from {n_before} to {n_after} ({(n_before-n_after)/n_before*100:.2f}% reduction)')

    return dataset


def run_on_node(configs, node_job_assignments, run_fn, **kwargs):
    def handle_exit(signum, frame):
        logger.warning(f"Received signal {signum}. Shutting down node runner...")
        sys.exit(1)

    signal.signal(signal.SIGINT, handle_exit)
    signal.signal(signal.SIGTERM, handle_exit)

    node = environ.get("SLURMD_NODENAME") or socket.gethostname()
    if "hpc" in node:
        logger.info(f"HPC node: {node}")
        node_id = int(node[3:])
    else:
        logger.warning(f"Non-HPC node: {node}")
        node_id = 'unknown'

    indices = node_job_assignments.get(node_id, node_job_assignments.get('unknown', []))
    selected = [configs[i] for i in indices]

    for c in selected:
        logger.info(f"Starting config for index {indices[selected.index(c)]}")
        run_fn(c, **kwargs)

    print('done!')


def run_on_nodes(configs, run_fn, **kwargs):
    n_nodes = int(environ.get('TOTAL_NODES', '1'))
    node_id = environ.get('SLURMD_NODENAME', socket.gethostname()) if n_nodes > 1 else None

    array_task_id = environ.get('SLURM_ARRAY_TASK_ID')
    if array_task_id is not None:
        idx = int(array_task_id)
        if idx >= len(configs):
            raise ValueError(
                f"SLURM_ARRAY_TASK_ID={idx} is out of range — only {len(configs)} config(s) "
                f"(valid: 0..{len(configs)-1}). Check for a stale env var."
            )
        logger.info(f"Array mode: task {idx+1}/{len(configs)}, config: {configs[idx]}")
        configs = [configs[idx]]
    elif node_id:
        logger.info(f"Multi-node mode: node_id={node_id}, total_nodes={n_nodes}")
    else:
        logger.info("Single-node mode")

    for yaml_path in configs:
        logger.info(f"Starting config: {yaml_path}")
        run_fn(yaml_path, **kwargs)
