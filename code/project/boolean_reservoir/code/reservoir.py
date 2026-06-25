import numpy as np
from pathlib import Path
import networkx as nx
from project.boolean_reservoir.code.parameter import *
from project.boolean_reservoir.code.lut import lut_random
from project.boolean_reservoir.code.graph import generate_adjacency_matrix, graph2adjacency_list_incoming
from project.boolean_reservoir.code.utils.param_utils import ExpressionEvaluator
from project.boolean_reservoir.code.utils.utils import set_seed
from project.boolean_reservoir.code.utils.reservoir_utils import InputPerturbationStrategy, InitializationStrategy, OutputActivationStrategy, BatchedTensorHistoryWriter, SaveAndLoadModel, ChainedSelector, BipartiteMappingStrategy, homogenize_adj_list, build_ring_lattice, install_wiring as _install_wiring_util

_jl = None

def _get_jl():
    """Lazily import juliacall and load the reservoir engine on first use."""
    global _jl
    if _jl is None:
        from juliacall import Main as _jl_main
        _jl_main.include(str(Path(__file__).parent / 'res_engine.jl'))
        _jl = _jl_main
    return _jl


def _parse_warmup_ticks(init_str: str) -> int | None:
    parts = init_str.split('-')
    if 'warmup' in parts:
        idx = parts.index('warmup')
        if idx + 1 < len(parts):
            return int(parts[idx + 1])
    return None


class BooleanReservoir:
    def __init__(self, params: Params = None, load_path=None, load_dict=None):
        load_dict = load_dict or {}

        if load_path is not None:
            load_path = Path(load_path)
            if load_path.suffix in ('.yaml', '.yml'):
                params = load_yaml_config(load_path)
            elif load_path.is_dir():
                load_dict = SaveAndLoadModel.load(checkpoint_path=load_path, parameter_override=params)
                params = None
            else:
                raise AssertionError(f'load_path must be a yaml file or checkpoint directory, got: {load_path}')

        self.P: Params = params if params else load_dict['parameters']
        self.M = self.P.M
        self.L = self.P.L
        self.I = self.P.M.I
        self.R = self.P.M.R
        self.O = self.P.M.O
        self.T = self.P.M.T

        self.training = True

        # INPUT LAYER
        set_seed(self.I.seed)

        N_I = self.I.n_nodes
        N_R = self.R.n_nodes
        N_O = self.O.n_nodes
        N_total = self.M.n_nodes
        self.N_total     = N_total
        self.input_slice = slice(0, N_I)
        self.res_slice   = slice(N_I, N_I + N_R)
        self.out_slice   = slice(N_I + N_R, N_total)

        node_indices = np.arange(N_total, dtype=np.int64)
        cs = ChainedSelector(N_total, parameters={'I': self.M.I.n_nodes})
        input_nodes = cs.eval(self.M.I.selector)
        input_nodes_mask = np.zeros(N_total, dtype=np.bool_)
        input_nodes_mask[input_nodes] = True
        ticks = np.array([int(c) for c in self.I.ticks_expanded], dtype=np.uint8)

        # RESERVOIR LAYER
        set_seed(self.R.seed)
        w_bi_gen = graph_quadrants = None
        if 'graph' not in load_dict:
            w_bi_gen, graph_quadrants = self._build_patchwork_graph(self.P)
        w_bi = SaveAndLoadModel.load_or_generate('w_bi', load_dict, lambda: w_bi_gen)
        self.graph = SaveAndLoadModel.load_or_generate('graph', load_dict, lambda:
            self.build_graph_from_quadrants(graph_quadrants)
        )
        assert N_total == self.graph.number_of_nodes()

        adj_list_raw = graph2adjacency_list_incoming(self.graph)
        in_degrees = [d for _, d in self.graph.in_degree()]
        max_length = max(in_degrees) if in_degrees else 1
        adj_list_np, adj_list_mask_np, no_neighbours_mask_np = homogenize_adj_list(adj_list_raw, max_length=max_length)
        no_neighbours_indices = np.where(no_neighbours_mask_np)[0].astype(np.int64)

        if not np.array_equal(input_nodes, np.arange(N_I)):
            for ni in range(N_I):
                where = adj_list_mask_np & (adj_list_np == ni)
                adj_list_np[where] = input_nodes[ni]

        self.max_connectivity = adj_list_np.shape[-1]
        lut_raw = SaveAndLoadModel.load_or_generate('lut', load_dict, lambda:
            lut_random(N_total, self.max_connectivity, p=self.R.p)
        )
        lut = lut_raw if isinstance(lut_raw, np.ndarray) else np.array(lut_raw)
        lut_flat, lut_offsets = BooleanReservoir._make_lut_jagged(lut, adj_list_mask_np, self.max_connectivity)

        powers_of_2 = self.precompute_powers_of_2(self.max_connectivity)

        initial_states_raw = SaveAndLoadModel.load_or_generate('init_state', load_dict, lambda:
            self.initialization_strategy(self.P)
        )
        initial_states = initial_states_raw if isinstance(initial_states_raw, np.ndarray) else np.array(initial_states_raw)
        states_parallel = np.tile(initial_states, (self.T.batch_size, 1))

        # OUTPUT LAYER
        set_seed(self.O.seed)
        output_nodes_mask = np.zeros(N_total, dtype=np.bool_)
        output_nodes_mask[self.res_slice] = True
        self._bipolar_readout  = (self.O.encoding == 'bipolar')
        self._timeseries_readout = (self.O.mode == 'time-series')
        self.output_activation = self.output_activation_strategy(self.P)

        # Readout weights — zeros until ridge solve pushes trained values
        readout_W = np.zeros((self.O.n_nodes, self.R.n_nodes), dtype=np.float32)
        readout_b = np.zeros(self.O.n_nodes, dtype=np.float32)
        if 'weights' in load_dict:
            w = load_dict['weights']
            readout_W = w['W']
            readout_b = w['b']

        set_seed(self.R.seed)

        # LOGGING
        self.init_logging()

        # NUMPY ATTRIBUTE STORAGE
        self.node_indices          = node_indices
        self.input_nodes           = input_nodes
        self.input_nodes_mask      = input_nodes_mask
        self.ticks                 = ticks
        self.w_bi                  = w_bi
        self.adj_list              = adj_list_np
        self.adj_list_mask         = adj_list_mask_np
        self.no_neighbours_indices = no_neighbours_indices
        self.lut                   = lut
        self.lut_flat              = lut_flat
        self.lut_offsets           = lut_offsets
        self.powers_of_2           = powers_of_2
        self.initial_states        = initial_states
        self.states_parallel       = states_parallel
        self.output_nodes_mask     = output_nodes_mask
        self.readout_W             = readout_W
        self.readout_b             = readout_b
        self.states_train_backup   = states_parallel.copy()
        self.states_eval_backup    = states_parallel.copy()

        self._julia_engine         = None
        self._julia_forward_engine = None

        self.add_graph_labels(self.graph)
        self.reset_reservoir(hard_reset=True)

    def init_logging(self):
        self.out_path = Path(self.L.out_path)
        self.timestamp_utc = SaveAndLoadModel.get_timestamp_utc()
        self.save_path = self.out_path / 'runs' / self.timestamp_utc
        self.L.save_path = self.save_path
        self.L.timestamp_utc = self.timestamp_utc
        self.L.history.save_path = self.save_path / 'history'
        self.record = self.L.history.record
        self.history = BatchedTensorHistoryWriter(save_path=self.L.history.save_path, buffer_size=self.L.history.buffer_size, persist_to_disk=self.L.history.persist_to_disk) if self.record else None

    def add_graph_labels(self, graph):
        labels_mapping = {node: node for node in graph.nodes()}
        nx.set_node_attributes(graph, labels_mapping, 'id')
        labels_mapping = {k: bool(v) for k, v in zip(graph.nodes(), self.input_nodes_mask)}
        nx.set_node_attributes(graph, labels_mapping, 'I')
        labels_mapping = {k: bool(v) for k, v in zip(graph.nodes(), self.output_nodes_mask)}
        nx.set_node_attributes(graph, labels_mapping, 'bipartite')
        nx.set_node_attributes(graph, labels_mapping, 'R')
        nx.set_node_attributes(graph, labels_mapping, 'O')

        def _part(node):
            if self.input_nodes_mask[node]:
                return 'I'
            if self.out_slice.start <= node < self.out_slice.stop:
                return 'O'
            return 'R'

        edge_quadrants = {(u, v): _part(u) + _part(v) for u, v in graph.edges()}
        nx.set_edge_attributes(graph, edge_quadrants, 'quadrant')

    @staticmethod
    def build_graph_and_lut(params: Params):
        """Cheap init: graph + LUT only. Skips input/output layers and state init."""
        set_seed(params.M.R.seed)
        _, graph_quadrants = BooleanReservoir._build_patchwork_graph(params)
        graph = BooleanReservoir.build_graph_from_quadrants(graph_quadrants)
        adj_list_raw = graph2adjacency_list_incoming(graph)
        in_degrees = [d for _, d in graph.in_degree()]
        max_length = max(in_degrees) if in_degrees else 1
        adj_list_np, adj_list_mask_np, _ = homogenize_adj_list(adj_list_raw, max_length=max_length)
        max_connectivity = adj_list_np.shape[-1]
        lut_raw = lut_random(params.M.n_nodes, max_connectivity, p=params.M.R.p)
        lut = lut_raw if isinstance(lut_raw, np.ndarray) else np.array(lut_raw)
        lut_flat, lut_offsets = BooleanReservoir._make_lut_jagged(lut, adj_list_mask_np, max_connectivity)

        result = object.__new__(BooleanReservoir)
        result.graph = graph
        result.lut = lut
        result.lut_flat = lut_flat
        result.lut_offsets = lut_offsets
        return result

    @staticmethod
    def build_graph_from_quadrants(gq):
        row_I = np.concatenate([gq[('I','I')], gq[('I','R')], gq[('I','O')]], axis=1)
        row_R = np.concatenate([gq[('R','I')], gq[('R','R')], gq[('R','O')]], axis=1)
        row_O = np.concatenate([gq[('O','I')], gq[('O','R')], gq[('O','O')]], axis=1)
        w = np.concatenate([row_I, row_R, row_O], axis=0)
        return nx.from_numpy_array(w, create_using=nx.DiGraph)

    @staticmethod
    def _resolve_random_wiring_params(w, structural, variables):
        numeric_vars = {k: v for k, v in variables.items() if not isinstance(v, str)}
        k_avg = float(ExpressionEvaluator({**numeric_vars, **structural}).eval(w.k_avg))

        pre_ev = ExpressionEvaluator({**structural, **numeric_vars, 'k_avg': k_avg})
        all_vars: dict = {}
        for key, val in variables.items():
            if isinstance(val, str):
                try:
                    all_vars[key] = float(pre_ev.eval(val))
                except Exception:
                    all_vars[key] = val
            else:
                all_vars[key] = val

        ev = ExpressionEvaluator({**structural, **all_vars, 'k_avg': k_avg})
        k_min = int(round(ev.eval(w.k_min)))
        k_max = int(round(ev.eval(w.k_max)))
        self_loops = float(ev.eval(w.self_loops)) if w.self_loops is not None else None
        mode = str(all_vars.get(w.mode, w.mode))
        if mode == 'homogeneous':
            k_min = k_max = int(k_avg)
        return k_min, k_avg, k_max, self_loops, mode

    @staticmethod
    def _nx_graph_to_block(g, directed, same_layer, src_size, tgt_size):
        if directed and not g.is_directed():
            g = g.to_directed()
        if same_layer:
            return np.array(nx.to_numpy_array(g, nodelist=range(tgt_size)), dtype=np.uint8)
        biadj = nx.bipartite.biadjacency_matrix(g, row_nodes=range(src_size)).toarray()
        return np.array(biadj, dtype=np.uint8)

    @staticmethod
    def _build_patchwork_graph(params: Params):
        P = params
        GRAPH_LAYERS = 'IRO'
        layer_sizes = {
            'B': P.M.I.bits,
            'I': P.M.I.n_nodes,
            'R': P.M.R.n_nodes,
            'O': P.M.O.n_nodes,
        }
        all_layers = 'BIRO'
        quadrants = {(s, t): None for s in all_layers for t in all_layers}

        for w in P.M.wiring:
            src_list = w.source.split('+')
            tgt_list = w.target.split('+')

            src_same  = [s for s in src_list if s in tgt_list]
            src_other = [s for s in src_list if s not in tgt_list]
            ordered_src = src_same + src_other

            for s in ordered_src:
                for t in tgt_list:
                    assert quadrants[(s, t)] is None, f"Duplicate wiring for ({s}→{t})"

            src_size = sum(layer_sizes[s] for s in ordered_src)
            tgt_size = sum(layer_sizes[t] for t in tgt_list)
            structural = {
                'I_n_nodes': P.M.I.n_nodes, 'R_n_nodes': P.M.R.n_nodes,
                'src_size': src_size, 'tgt_size': tgt_size, 'n_nodes': tgt_size,
            }
            same_layer = (src_list == tgt_list and len(src_list) == 1)

            if isinstance(w, (GNMWiring, GNPWiring)):
                k_min, k_avg, k_max, self_loops, _ = BooleanReservoir._resolve_random_wiring_params(
                    w, structural, P.M.variables
                )
                if w.degree == 'out':
                    block = generate_adjacency_matrix(
                        n_nodes=src_size, k_min=k_min, k_avg=k_avg,
                        k_max=min(k_max, tgt_size), self_loops=self_loops, rows=tgt_size,
                        fixed_edges=isinstance(w, GNMWiring),
                    ).T
                else:
                    block = generate_adjacency_matrix(
                        n_nodes=tgt_size, k_min=k_min, k_avg=k_avg,
                        k_max=min(k_max, src_size), self_loops=self_loops, rows=src_size,
                        fixed_edges=isinstance(w, GNMWiring),
                    )

            elif isinstance(w, NetworkXWiring):
                ev = ExpressionEvaluator({**structural, **{k: v for k, v in P.M.variables.items()
                                                           if not isinstance(v, str)}})
                if same_layer:
                    fn = getattr(nx, w.graph.name, None)
                    if fn is None:
                        raise ValueError(f"nx has no function {w.graph.name!r}.")
                    g = w.graph.call(fn, evaluator=ev, n=tgt_size)
                else:
                    fn = getattr(nx.bipartite, w.graph.name, None)
                    if fn is None:
                        raise ValueError(f"nx.bipartite has no function {w.graph.name!r}.")
                    g = w.graph.call(fn, evaluator=ev, n=src_size, m=tgt_size)
                block = BooleanReservoir._nx_graph_to_block(g, w.directed, same_layer, src_size, tgt_size)
                if w.k_max is not None:
                    for col in np.where(block.sum(axis=0) > w.k_max)[0]:
                        excess = np.where(block[:, col])[0]
                        np.random.shuffle(excess)
                        block[excess[w.k_max:], col] = 0

            elif isinstance(w, ExplicitWiring):
                fn = getattr(nx, w.graph.name, None)
                if fn is None:
                    raise ValueError(f"nx has no function {w.graph.name!r}.")
                g = w.graph.call(fn)
                g = nx.convert_node_labels_to_integers(g)
                expected_nodes = tgt_size if same_layer else src_size + tgt_size
                if g.number_of_nodes() != expected_nodes:
                    raise ValueError(
                        f"ExplicitWiring: loaded graph has {g.number_of_nodes()} nodes, "
                        f"expected {expected_nodes} for {w.source!r} -> {w.target!r}."
                    )
                block = BooleanReservoir._nx_graph_to_block(g, w.directed, same_layer, src_size, tgt_size)

            elif isinstance(w, PatternWiring):
                strategy_fn = BipartiteMappingStrategy.get(
                    w.pattern, k_avg=P.M.variables.get('R_k_avg')
                )
                block = strategy_fn(P, src_size, tgt_size).astype(np.uint8)

            else:
                raise ValueError(f"Unknown wiring type: {type(w)}")

            if w.source_selector:
                rows = ChainedSelector(src_size).eval(w.source_selector)
                row_mask = np.zeros(src_size, dtype=bool)
                row_mask[rows] = True
                block[~row_mask, :] = 0
            if w.target_selector:
                cols = ChainedSelector(tgt_size).eval(w.target_selector)
                col_mask = np.zeros(tgt_size, dtype=bool)
                col_mask[cols] = True
                block[:, ~col_mask] = 0

            row_off = 0
            for s in ordered_src:
                col_off = 0
                for t in tgt_list:
                    rs, re_ = row_off, row_off + layer_sizes[s]
                    cs, ce  = col_off, col_off + layer_sizes[t]
                    quadrants[(s, t)] = np.array(block[rs:re_, cs:ce], dtype=np.uint8)
                    col_off += layer_sizes[t]
                row_off += layer_sizes[s]

        for (s, t) in list(quadrants.keys()):
            if quadrants[(s, t)] is None:
                quadrants[(s, t)] = np.zeros((layer_sizes[s], layer_sizes[t]), dtype=np.uint8)

        w_bi = quadrants[('B', 'I')]
        graph_quadrants = {(s, t): quadrants[(s, t)] for s in GRAPH_LAYERS for t in GRAPH_LAYERS}
        return w_bi, graph_quadrants

    @staticmethod
    def bipartite_mapping_strategy(p: Params, strategy: str, a: int, b: int):
        strategy_fn = BipartiteMappingStrategy.get(strategy, k_avg=p.M.variables.get('R_k_avg'))
        return strategy_fn(p, a, b)

    @staticmethod
    def input_perturbation_strategy(p: Params):
        return InputPerturbationStrategy.get(p.M.I.perturbation)

    @staticmethod
    def initialization_strategy(p: Params):
        return InitializationStrategy.get(p.M.R.init.split('-')[0])(p.M.n_nodes)

    @staticmethod
    def output_activation_strategy(p: Params):
        return OutputActivationStrategy.get(p.M.O.activation)

    @staticmethod
    def get_timestamp_utc():
        return SaveAndLoadModel.get_timestamp_utc()

    def save(self, save_path=None):
        if save_path is None:
            save_path = self.save_path

        paths, self.L.last_checkpoint = SaveAndLoadModel.save_model({
            'P':              self.P,
            'w_bi':           self.w_bi,
            'graph':          self.graph,
            'initial_states': self.initial_states,
            'lut':            self.lut,
            'weights':        {'W': self.readout_W, 'b': self.readout_b},
            'save_path':      save_path,
            'history':        self.L.history,
        })
        return paths

    def load(self, checkpoint_path: Path = None, paths: dict = None, parameter_override: Params = None):
        load_dict = SaveAndLoadModel.load(checkpoint_path=checkpoint_path, paths=paths, parameter_override=parameter_override)
        self.__init__(load_dict=load_dict)
        return load_dict

    def flush_history(self):
        if self.history:
            self.history.flush()

    @staticmethod
    def precompute_powers_of_2(bits):
        return (2.0 ** np.arange(bits, dtype=np.float32)[::-1]).copy()

    @staticmethod
    def _make_lut_jagged(lut: np.ndarray, adj_list_mask: np.ndarray, max_k: int):
        degrees   = adj_list_mask.sum(axis=1).astype(np.int64)
        lut_sizes = np.int64(1) << degrees
        lut_offsets = np.zeros(len(degrees) + 1, dtype=np.int64)
        lut_offsets[1:] = np.cumsum(lut_sizes)
        lut_flat = np.empty(int(lut_offsets[-1]), dtype=np.uint8)
        for i, (deg, size) in enumerate(zip(degrees.tolist(), lut_sizes.tolist())):
            stride = 1 << (max_k - deg)
            lut_flat[lut_offsets[i] : lut_offsets[i + 1]] = lut[i, ::stride][:size]
        return lut_flat, lut_offsets

    build_ring_lattice = staticmethod(build_ring_lattice)

    def install_wiring(self, adj_list: np.ndarray, adj_mask: np.ndarray, lut: np.ndarray) -> None:
        _install_wiring_util(self, adj_list, adj_mask, lut)

    def bin2int(self, x):
        return np.dot(x.astype(np.float32), self.powers_of_2).astype(np.int64)

    def train(self, mode: bool = True):
        if self.training == mode or self.R.reset:
            self.training = mode
            return self
        current_backup = self.states_train_backup if self.training else self.states_eval_backup
        np.copyto(current_backup, self.states_parallel)
        target_backup = self.states_train_backup if mode else self.states_eval_backup
        self.reset_reservoir(target_state=target_backup)
        self.training = mode
        return self

    def eval(self):
        return self.train(False)

    def reset_reservoir(self, samples=None, target_state=None, hard_reset=False):
        if samples is None:
            samples = self.states_parallel.shape[0]

        if target_state is None:
            target_state = np.broadcast_to(self.initial_states, (samples, self.initial_states.shape[1]))

        self.states_parallel[:samples] = target_state[:samples]
        if 'warmup' in self.R.init:
            self.warmup(ticks=_parse_warmup_ticks(self.R.init))

        if hard_reset and hasattr(self, 'states_train_backup'):
            self.states_train_backup[:samples] = self.states_parallel[:samples].copy()
            self.states_eval_backup[:samples]  = self.states_parallel[:samples].copy()

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        x_np = np.asarray(x)
        m = x_np.shape[0]

        if m > self.T.batch_size:
            outputs_list = []
            kb = 0
            for i in range(m // self.T.batch_size):
                ka = i * self.T.batch_size
                kb = ka + self.T.batch_size
                outputs_list.append(self.forward(x_np[ka:kb]))
            if kb < m:
                outputs_list.append(self.forward(x_np[kb:]))
            return np.concatenate(outputs_list, axis=0)

        if self.R.reset:
            self.reset_reservoir(samples=m)
        self.batch_record(m, phase='init', s=0, f=0)

        x_np = x_np.reshape(m, -1, self.I.chunks, self.I.chunk_size)
        s = x_np.shape[1]
        c = self.I.chunks
        k = self.I.bits // self.I.chunks
        return self._forward_julia(x_np, m, s, c, k)

    _PERT_CODE = {'xor': 0, 'override': 1, 'and': 2, 'or': 3}

    def _ensure_julia_engine(self):
        if self._julia_engine is not None:
            return
        jl = _get_jl()
        engine = jl.init_engine(
            self.adj_list,
            self.adj_list_mask,
            self.no_neighbours_indices,
            self.lut_flat,
            self.lut_offsets,
            self.w_bi,
            self.ticks,
            self.input_nodes,
            self.output_nodes_mask,
            self.N_total,
            self.max_connectivity,
            self.T.batch_size,
            self.I.n_nodes,
            self.I.chunks,
            self.I.bits // self.I.chunks,
            self._PERT_CODE[self.I.perturbation],
            self.O.n_nodes,
        )
        # Push initial readout weights into the Julia engine
        jl.set_readout_b(engine, self.readout_W, self.readout_b)
        self._julia_engine = (jl.warmup_b, engine)

    def _ensure_julia_forward_engine(self):
        if self._julia_forward_engine is not None:
            return
        self._ensure_julia_engine()
        jl = _get_jl()
        _, engine = self._julia_engine
        self._julia_forward_engine = (jl.forward_sequence_b, engine)

    def set_readout_weights(self, W: np.ndarray, b: np.ndarray):
        """Update readout weights after ridge solve. Pushes to Julia engine if initialised."""
        self.readout_W = W.astype(np.float32)
        self.readout_b = b.astype(np.float32)
        if self._julia_engine is not None:
            jl = _get_jl()
            _, engine = self._julia_engine
            jl.set_readout_b(engine, self.readout_W, self.readout_b)

    def get_states(self, x) -> np.ndarray:
        """Run reservoir forward pass and return pre-readout float32 states for ridge training.

        Classification mode: returns (m, n_r).
        Time-series mode: returns summed states over steps (m, n_r) with bias scaling applied.
        """
        x_np = np.asarray(x)
        m = x_np.shape[0]

        if self.R.reset:
            self.reset_reservoir(samples=m)

        x_np = x_np.reshape(m, -1, self.I.chunks, self.I.chunk_size)
        s = x_np.shape[1]

        self._ensure_julia_forward_engine()
        fwd_fn, engine = self._julia_forward_engine

        if self._timeseries_readout:
            step_buffer = np.empty((s, m, self.N_total), dtype=np.uint8, order='F')
        else:
            step_buffer = None

        x_forder = np.asfortranarray(x_np)
        fwd_fn(engine, self.states_parallel, x_forder, None, step_buffer)

        if self._timeseries_readout:
            # Convert F-order (Julia interop) to C-order before boolean-mask indexing
            o_sum = np.zeros((m, self.R.n_nodes), dtype=np.float32)
            step_c = np.ascontiguousarray(step_buffer)
            for si in range(s):
                o_step = step_c[si, :m, self.res_slice].astype(np.float32)
                if self._bipolar_readout:
                    o_step = o_step * 2 - 1
                o_sum += o_step
            return o_sum, s   # caller uses s as bias column value
        else:
            o = self.states_parallel[:m, self.res_slice].astype(np.float32, order='C')
            if self._bipolar_readout:
                o = o * 2 - 1
            return o, 1

    def _forward_julia(self, x_np: np.ndarray, m: int, s: int, c: int, k: int):
        self._ensure_julia_forward_engine()
        fwd_fn, engine = self._julia_forward_engine
        jl = _get_jl()

        if self.record:
            n_entries = s * (c + int(self.ticks.sum()))
            history_buffer = np.zeros((n_entries, m, self.N_total), dtype=np.uint8)
        else:
            history_buffer = None

        if self._timeseries_readout:
            step_buffer = np.empty((s, m, self.N_total), dtype=np.uint8, order='F')
        else:
            step_buffer = None

        x_forder = np.asfortranarray(x_np)
        fwd_fn(engine, self.states_parallel, x_forder, history_buffer, step_buffer)

        if self.record:
            for i, meta in enumerate(self._build_history_metadata(s, c)):
                self.history.append_batch(history_buffer[i], meta)

        if self._timeseries_readout:
            outputs = np.array(jl.apply_readout_timeseries_b(engine, step_buffer, m, s, self._bipolar_readout))
        else:
            outputs = np.array(jl.apply_readout_b(engine, self.states_parallel, m, self._bipolar_readout))

        if self.output_activation:
            outputs = self.output_activation(outputs)

        self.states_parallel[:m, self.out_slice] = (outputs > 0.5).astype(np.uint8)
        return outputs

    def _build_history_metadata(self, s: int, c: int) -> list:
        metadata = []
        for si in range(s):
            for ci in range(c):
                metadata.append({'phase': 'input_layer', 's': si + 1, 'f': ci + 1})
                t = int(self.ticks[ci])
                is_last_chunk = (ci == c - 1)
                for ti in range(t):
                    is_last_tick = (ti == t - 1)
                    if is_last_chunk and is_last_tick and (self._timeseries_readout or si == s - 1):
                        metadata.append({'phase': 'output_layer', 's': si + 1, 'f': ci + 1})
                    else:
                        metadata.append({'phase': 'reservoir_layer', 's': si + 1, 'f': ci + 1, 't': ti + 1})
        return metadata

    def _reservoir_tick(self, m):
        """Single synchronous tick of the reservoir. Useful for tests and manual stepping."""
        self._ensure_julia_engine()
        jl = _get_jl()
        _, engine = self._julia_engine
        jl.warmup_b(engine, self.states_parallel, int(m), 1)

    def warmup(self, ticks=None, m=None):
        if m is None:
            m = self.T.batch_size
        if ticks is None:
            ticks = self.ticks[0]
        self._ensure_julia_engine()
        jl = _get_jl()
        _, engine = self._julia_engine
        if self.record:
            for t in range(int(ticks)):
                jl.warmup_b(engine, self.states_parallel, int(m), 1)
                self.batch_record(m, phase='init', s=0, f=0, t=t + 1)
        else:
            jl.warmup_b(engine, self.states_parallel, int(m), int(ticks))

    def batch_record(self, m, **meta_data):
        if self.record:
            self.history.append_batch(self.states_parallel[:m], meta_data)


if __name__ == '__main__':
    I = InputParams(
        perturbation='override',
        encoding='base2',
        features=2,
        chunk_size=4,
        bits=4,
        n_nodes=8,
        ticks='2',
        seed=0,
    )
    R = ReservoirParams(n_nodes=10, k_min=0, k_avg=3, k_max=7, p=0.5, self_loops=0.1, seed=0)
    O = OutputParams(n_nodes=2, seed=0)
    T = TrainingParams(batch_size=3, epochs=10, accuracy_threshold=0.05, learning_rate=0.001, seed=0)
    L = LoggingParams(out_path='/out/delete/', history=HistoryParams(record=True, buffer_size=10))

    model_params = ModelParams(input_layer=I, reservoir_layer=R, output_layer=O, training=T, wiring=[
        PatternWiring(source='B', target='I', pattern='identity'),
        PatternWiring(source='I', target='R', pattern='identity'),
        RandomWiring(source='R', target='R', k_avg=3, k_min=0, k_max='7', self_loops='0.1'),
    ])
    params = Params(model=model_params, logging=L)
    model = BooleanReservoir(params)

    s = 1
    x = np.random.randint(0, 2, (T.batch_size, s, I.features, I.bits // I.features), dtype=np.uint8)
    print(model(x))
    model.flush_history()
    model.save()
