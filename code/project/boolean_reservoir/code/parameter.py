import re
from pydantic import BaseModel, Field, model_validator, field_validator, PrivateAttr, ConfigDict
import yaml
from typing import Annotated, Dict, Any, List, Union, Optional, Literal
from pathlib import Path
from benchmark.path_integration.parameter import PathIntegrationDatasetParams, KQGRPathIntegrationDatasetParams
from benchmark.temporal.parameter import TemporalDatasetParams, KQGRTemporalDatasetParams
from benchmark.neurogym.parameter import NeuroGymDatasetParams, KQGRNeuroGymDatasetParams
from project.boolean_reservoir.code.utils.param_utils import pydantic_init, calculate_w_broadcasting, DynamicParams, CallParams, ExpressionEvaluator, expand_ticks
import copy
from shutil import copy as shutil_copy
pydantic_init()


def deep_merge(base: dict, override: dict) -> dict:
    if not isinstance(base, dict) or not isinstance(override, dict):
        return copy.deepcopy(override)

    # Type-Swap Protector: if dataset type changes, discard base entirely
    if 'name' in base and 'name' in override and base['name'] != override['name']:
        return copy.deepcopy(override)

    merged = copy.deepcopy(base)
    for k, v in override.items():
        if k in merged and isinstance(merged[k], dict) and isinstance(v, dict):
            merged[k] = deep_merge(merged[k], v)
        else:
            merged[k] = copy.deepcopy(v)
    return merged


EncodingName = Literal['base2', 'primes', 'tally', 'binary_embedding', 'gray', 'rate', 'delta_sigma']


class InputParams(BaseModel):
    seed: Optional[int] = Field(
        None, description="Random seed, None disables seed")
    selector: Union[str, List[str]] = Field(
        'S :I', description="Selection chain. Supports F (filter), S (slice), R (random). Variables: i: index variable for F operation, I: I.n_nodes. Ie. assuming I=2 F i<4 -> S -3: -> R I ([0, 1, 2, 3]->[1, 2, 3]->[1, 3]). R samples: R without samples => scramble") # TODO make this generic for any wiring?
    perturbation: Union[Literal['xor', 'and', 'or', 'override'], List[Literal['xor', 'and', 'or', 'override']]] = Field(
        'xor', description=(
            "How the incoming input bit is combined with the current I-node state. "
            "'override': I[i] ← input[i] (pure replacement). "
            "'xor': I[i] ← I[i] ⊕ input[i]. "
            "'and': I[i] ← I[i] & input[i]. "
            "'or':  I[i] ← I[i] | input[i]. "
            "Only I-nodes are written; w_bi defines which input bits map to which I-nodes."
        ))
    encoding: Union[EncodingName, List[EncodingName]] = Field(
        'base2', description="Binary encoding type")
    interleaving: Union[int, List[int]] = Field(
        0, description="Multidimensionsional weaving of inputs, int dictates group size. n=1: abc, def -> ad, be, cf -> adb, ecf | n=2: abcd, efgh -> ab, ef, cd, gh -> abef, cdgh")
    n_nodes: Optional[Union[int, List[int]]] = Field(
        None, description="Number of input nodes; I")
    features: Union[int, List[int]] = Field(
        None, description=(
            "Number of input features before binary encoding. "
            "Defaults chunks=features so each feature occupies one sequential perturbation sub-step."
        ))
    bits: Optional[Union[int, List[int]]] = Field(
        None, description="Total bits after encoding (= features × resolution × redundancy = chunks × chunk_size).")
    resolution: Optional[Union[int, List[int]]] = Field(
        None, description="Bits per dimension before redundancy")
    redundancy: Union[int, List[int]] = Field(
        1, description="Redundancy factor of resolution")
    chunks: Optional[Union[int, List[int]]] = Field(
        None, description=(
            "Number of sequential perturbation sub-steps per forward step (c in x.shape=(m,s,c,b)). "
            "Defaults to features. Each chunk maps to a contiguous band of w_bi rows and is "
            "followed by ticks[ci] reservoir ticks."
        ))
    chunk_size: Optional[Union[int, List[int]]] = Field(
        None, description=(
            "Bits processed simultaneously per chunk (b in x.shape=(m,s,c,b); = bits // chunks). "
            "Defaults to resolution × redundancy per feature."
        ))
    permute: Optional[Union[str, List[str]]] = Field(
        None,
        description="Permutation of (features, redundancy, resolution) dims applied during "
                    "redundancy expansion. E.g. '021' interleaves redundancy copies so each "
                    "chunk contains K identical bits (parallel redundancy). None = block repeat.")
    ticks: Optional[Union[str, List[str]]] = Field(
        None, description=(
            "Reservoir ticks to run after each chunk's perturbation. "
            "Scalar '1' gives one tick per chunk; '1{n}' expands to n ones; "
            "'2{3}1{2}' gives three 2-tick chunks then two 1-tick chunks. "
            "Total ticks per forward call = s × sum(ticks_expanded). "
            "With features=1, bits=1, ticks='1': one tick per forward step."
        ))

    @model_validator(mode='after')
    def calculate_bits(self):
        """Calculate bits from other parameters if not set"""
        # Calculate from resolution and features
        if (self.bits is None and
            not isinstance(self.features, list) and
            not isinstance(self.resolution, list) and
                not isinstance(self.redundancy, list)):

            if self.resolution is not None and self.features is not None:
                self.bits = self.features * self.resolution * self.redundancy

        # Also handle the reverse: if bits is set, calculate resolution
        elif (self.bits is not None and
              self.resolution is None and
              not isinstance(self.bits, list) and
              not isinstance(self.features, list) and
              not isinstance(self.redundancy, list)):

            if self.features is not None:
                self.resolution = self.bits // (self.features *
                                                self.redundancy)

        return self

    # Both set - calculate bits if not set
    @model_validator(mode='after')
    def handle_chunking(self):
        """Handle chunks and chunk_size bidirectionally"""
        if (not isinstance(self.chunks, list) and
                not isinstance(self.chunk_size, list)):

            if self.chunks is not None and self.chunk_size is not None:
                if self.bits is None:
                    self.bits = self.chunks * self.chunk_size
            elif self.bits is not None and not isinstance(self.bits, list):
                if self.chunks is not None:
                    self.chunk_size = self.bits // self.chunks
                elif self.chunk_size is not None:
                    self.chunks = self.bits // self.chunk_size
                elif self.features is not None and not isinstance(self.features, list):
                    if (self.permute is not None and not isinstance(self.permute, list)
                            and not isinstance(self.redundancy, list)):
                        self.chunk_size = self.redundancy
                        self.chunks = self.bits // self.chunk_size
                    else:
                        self.chunks = self.features
                        if self.resolution is not None and not isinstance(self.resolution, list):
                            self.chunk_size = self.resolution * self.redundancy
        return self

    @model_validator(mode='after')
    def default_n_nodes(self):
        if self.n_nodes is None:
            self.n_nodes = calculate_w_broadcasting(
                lambda x, y: x, self.bits, None)
        return self

    @model_validator(mode='after')
    def handle_no_input(self):
        """When n_nodes=0 (autonomous CA, no input), fill in features/chunks/chunk_size/bits
        so the Julia engine initialises correctly (c=1, k_bits=1, N_I=0) and saved
        parameters round-trip through YAML without validation errors.
        The perturbation loop runs for n in 1:0, so no actual input is injected."""
        if self.n_nodes == 0 and self.chunks is None:
            self.features   = 1
            self.chunks     = 1
            self.chunk_size = 1
            self.bits       = 1
        return self

    @model_validator(mode='after')
    def set_default_ticks(self):
        if self.chunks is None or isinstance(self.chunks, list):
            return self
        if self.ticks is None:
            self.ticks = f'1{{{self.chunks}}}'
        return self

    @property
    def ticks_expanded(self):
        if self.ticks is None:
            return '1' * self.chunks if self.chunks else None

        expanded = expand_ticks(self.ticks)

        # Repeat/truncate to match chunks
        if self.chunks and len(expanded) != self.chunks:
            if len(expanded) < self.chunks:
                # Repeat pattern to fill
                expanded = (expanded * (self.chunks //
                            len(expanded) + 1))[:self.chunks]
            else:
                # Truncate
                expanded = expanded[:self.chunks]

        return expanded

    @property # TODO remove (was misspelled, should be perturbation)
    def pertubation(self):
        return self.perturbation


_INIT_PATTERN = re.compile(
    r'^(random|zeros|ones|every_other)(-warmup(-\d+)?)?$'
)


class ReservoirParams(BaseModel):
    seed: Optional[int] = Field(
        None, description="Random seed, None disables seed")
    n_nodes: Optional[Union[int, List[int]]] = Field(
        None, description="Number of reservoir nodes (R) excluding input nodes (I)")
    p: Union[float, List[float]] = Field(
        0.5, description="Probability for 1 in LUT (look up table)")
    reset: Optional[Union[bool, List[bool]]] = Field(
        True, description=(
            "If True, reset states_parallel to initial_states at the start of each forward() call. "
            "Set False for tasks where reservoir state must persist across calls "
            "(e.g. path integration with continuous trajectories)."
        ))
    init: Union[str, List[str]] = Field(
        'random',
        description=(
            "Init strategy: base or base-warmup or base-warmup-N. "
            "Base: random | zeros | ones | every_other. "
            "'-warmup' warms up for ticks[0] ticks; '-warmup-N' warms up for exactly N ticks."
        )
    )

    @field_validator('init', mode='before')
    @classmethod
    def _validate_init(cls, v):
        for item in (v if isinstance(v, list) else [v]):
            if not _INIT_PATTERN.match(str(item)):
                raise ValueError(
                    f"Invalid init strategy '{item}'. "
                    f"Expected base[-warmup[-N]] where base ∈ "
                    f"{{random, zeros, ones, every_other}}"
                )
        return v


# TODO add w_out and distribution like in input_layer. atm we assume full readout of R
class OutputParams(BaseModel):
    seed: Optional[int] = Field(
        None, description="Random seed, None disables seed")
    n_nodes: Union[int, List[int]] = Field(
        1, description="Dimension of output data")
    activation: Optional[Union[str, List[str]]] = Field(
        None, description=(
            "Activation applied to the linear readout output. "
            "Also controls what forward() returns and how O-node states are set: "
            "None → forward() returns raw linear readout, O-nodes set to (raw > 0.5); "
            "'sigmoid' → forward() returns sigmoid output, O-nodes set to (sigmoid > 0.5). "
            "Default None (linear) is correct for regression, MSE classification, and "
            "BCEWithLogitsLoss (sigmoid is applied inside the loss). "
            "Use 'sigmoid' only when the criterion is BCELoss, which expects inputs in (0,1). "
            "WARNING: combining 'sigmoid' with Ridge is incorrect — Ridge minimises MSE on the "
            "pre-sigmoid output so the training objective and evaluation are mismatched; "
            "use None (linear readout) whenever Ridge is in the optimizer sweep."
        ))
    encoding: Optional[Union[Literal['binary', 'bipolar'], List[Literal['binary', 'bipolar']]]] = Field(
        "binary", description="Encoding of reservoir states for readout: 'binary'={0,1}, 'bipolar'={-1,1}")
    mode: Optional[Union[Literal['classification', 'time-series'], List[Literal['classification', 'time-series']]]] = Field(
        "classification", description="'classification': single readout from final state. 'time-series': sum readouts from all steps, activation applied once on the sum.")


class TrainingParams(BaseModel):
    seed: Optional[int] = Field(
        None, description="Random seed, None disables seed")
    batch_size: Union[int, List[int]] = Field(
        128, description="Number of samples per forward pass")
    criterion: Optional[Union[str, List[str]]] = Field(
        'MSE', description="ML criterion, fex MSE, BCE")
    epochs: Union[int, List[int]] = Field(100, description="Number of epochs")
    accuracy_threshold: Union[float, List[float]] = Field(
        0.5, description="Threshold for generic accuracy metric")
    evaluation: Optional[str] = Field(
        'test', description="test, dev, train etc")
    shuffle: bool = Field(True, description="Shuffle dataset")
    drop_last: bool = Field(True, description="Drop last")
    optim: Union[DynamicParams, List[DynamicParams]] = Field(
        # standard for RC in litterature (ridge)
        default=DynamicParams(name='adam', params={
                              'lr': 1e-3, 'weight_decay': 1e-3}),
        description="Optimizer configuration"
    )
    accuracy: Optional[str] = Field(
        None,
        description="Accuracy function: 'Euclidean' or 'Boolean'. None infers from dataset type.",
        json_schema_extra={'expand': False}
    )

    @property
    def accuracy_obj(self):
        from project.boolean_reservoir.code.train_model import EuclideanDistanceAccuracy, BooleanAccuracy
        mapping = {'Euclidean': EuclideanDistanceAccuracy, 'Boolean': BooleanAccuracy}
        if self.accuracy not in mapping:
            raise ValueError(f"Unknown accuracy '{self.accuracy}'. Available: {list(mapping)}")
        return mapping[self.accuracy]().accuracy


class _RandomWiringBase(BaseModel):
    model_config = ConfigDict(extra='forbid')
    source: str
    target: str
    k_avg: Union[float, str]
    k_min: Union[int, str] = 0
    k_max: Union[int, str] = 'n_nodes'
    self_loops: Optional[Union[float, str]] = None
    mode: Union[str, List[str]] = 'heterogeneous'  # literal, variable reference (e.g. "R_mode"), or grid-search list
    degree: Literal['in', 'out'] = 'in'
    source_selector: Optional[str] = None
    target_selector: Optional[str] = None


class GNMWiring(_RandomWiringBase):
    """G(n,m) random wiring: exactly k_avg*n_nodes edges distributed via pigeonhole."""
    type: Literal['gnm'] = 'gnm'


class GNPWiring(_RandomWiringBase):
    """G(n,p) random wiring: each node's degree drawn from Binomial(k_max-k_min, p)."""
    type: Literal['gnp'] = 'gnp'


# Keep alias for any code that still imports RandomWiring directly
RandomWiring = GNMWiring


class NetworkXWiring(BaseModel):
    """Wraps any NetworkX graph generator as reservoir wiring.

    Same-layer (source == target): uses nx.<graph.name>(n=tgt_size, **graph.params).
    Cross-layer (source != target): uses nx.bipartite.<graph.name>(n=src_size, m=tgt_size, **graph.params)
    and extracts the biadjacency matrix.

    graph.params values can be numeric literals or structural-variable expressions
    (I_n_nodes, R_n_nodes, src_size, tgt_size, n_nodes, and non-string model.variables).
    """
    model_config = ConfigDict(extra='forbid')
    type: Literal['networkx'] = 'networkx'
    source: str
    target: str
    graph: DynamicParams
    directed: bool = True  # only applies to same-layer; bipartite extraction is always directed
    k_max: Optional[int] = None  # clip in-degree to this value; required for scale-free graphs where hub nodes would overflow the LUT
    source_selector: Optional[str] = None
    target_selector: Optional[str] = None


class ExplicitWiring(BaseModel):
    """Load a pre-existing graph via any NetworkX reader or converter.

    graph.name: a top-level networkx function, e.g. 'read_edgelist', 'read_adjlist',
                'read_graphml', 'read_gml'. graph.params: its kwargs (e.g. path).
    The loaded graph must have exactly tgt_size nodes (for same-layer wiring)
    or src_size + tgt_size nodes (cross-layer, bipartite extraction).
    graph.params values support structural-variable expressions.
    """
    model_config = ConfigDict(extra='forbid')
    type: Literal['explicit'] = 'explicit'
    source: str
    target: str
    graph: DynamicParams
    directed: bool = True
    source_selector: Optional[str] = None
    target_selector: Optional[str] = None


class PatternWiring(BaseModel):
    model_config = ConfigDict(extra='forbid')
    type: Literal['pattern'] = 'pattern'
    source: str
    target: str
    pattern: str  # 'identity' | 'zeroes'
    source_selector: Optional[str] = None
    target_selector: Optional[str] = None


WiringConfig = Annotated[
    Union[GNMWiring, GNPWiring, NetworkXWiring, ExplicitWiring, PatternWiring],
    Field(discriminator='type')
]


class ModelParams(BaseModel):
    variables: Dict[str, Any] = Field(
        default_factory=dict,
        description="Named scalars used in wiring expressions (e.g. R_k_avg, R_k_min). "
                    "List values trigger grid-search expansion."
    )
    input_layer: InputParams
    reservoir_layer: ReservoirParams
    output_layer: OutputParams
    training: TrainingParams
    wiring: List[WiringConfig] = Field(
        default_factory=list, json_schema_extra={'expand': False}
    )

    @property
    def I(self):
        return self.input_layer

    @property
    def R(self):
        return self.reservoir_layer

    @property
    def O(self):
        return self.output_layer

    @property
    def T(self):
        return self.training

    @property
    def n_nodes(self):
        return self.I.n_nodes + self.R.n_nodes + self.O.n_nodes


class GridSearchParams(BaseModel):
    seed: Optional[int] = Field(
        None, description="Random seed, None disables seed")
    n_samples: Optional[int] = Field(
        1, ge=1, description="Number of samples per configuration in grid search")
    run: List[str] = Field(
        default=['train'],
        description=(
            "Per-config run control: 'train' runs training, 'kqgr' runs KQGR metrics. "
            "Defaults to ['train']. Override per-universe via logging.grid_search.run."
        ),
        json_schema_extra={'expand': False}
    )


class HistoryParams(BaseModel):
    record: Optional[bool] = Field(
        False, description="Reservoir dynamics state recording")
    buffer_size: Optional[int] = Field(
        64, description="Number of batched snapshots per output file")
    save_path: Optional[Path] = Field(
        Path('out/history'), description="Where model is saved when recording history")
    persist_to_disk: Optional[bool] = Field(False, description="Save state history in files not RAM")

class ReservoirMetrics(BaseModel):
    spectral_radius: float | None = None
    lut_p: float | None = None

class TrainLog(BaseModel):
    config: int | None = None
    sample: int | None = None
    accuracy: float | None = None
    loss: float | None = None
    epoch: int | None = None

class KQGRMetrics(BaseModel):
    config: int | None = None
    sample: int | None = None
    kq: int | None = None
    gr: int | None = None
    delta: int | None = None

class LoggingParams(BaseModel):
    timestamp_utc: Optional[str] = Field(None, description="timestamp utc")
    out_path: Path = Field(
        Path('out'), description="Where to save all logs for this config")
    save_path: Optional[Path] = Field(
        Path('out'), description="Where last run was saved")
    last_checkpoint: Optional[Path] = Field(
        None, description="Where last checkpoint was saved")
    grid_search: Optional[GridSearchParams] = Field(None)
    history: HistoryParams = Field(
        default_factory=HistoryParams, description="Parameters pertaining to recoding of reservoir dynamics")
    universe: str | None = None
    train: TrainLog | None = None
    kqgr: KQGRMetrics | None = None
    reservoir_metrics: ReservoirMetrics | None = None
    save_keys: Optional[List[str]] = Field(
        default=['parameters', 'w_bi', 'graph',
                 'init_state', 'lut', 'weights'],
        description="Only save these model objects",
        json_schema_extra={'expand': False}  # Mark as non-expandable
    )

    @field_validator('save_keys', mode='before')
    @classmethod
    def _convert_string_to_list(cls, v):
        if isinstance(v, str):
            return [v]
        return v

    @property
    def M(self):
        return self.kqgr

    @property
    def T(self):
        return self.train


class UniverseWrapper:
    """Lazy wrapper that deep-merges universe overrides into child Params on access."""

    def __init__(self, mother: 'Params', overrides: dict):
        self._mother = mother
        self._overrides = overrides or {}
        self._cache: dict = {}

    def __getattr__(self, name: str) -> 'Params':
        if name.startswith('_'):
            raise AttributeError(name)
        if name not in self._cache:
            if name in self._overrides:
                merged = deep_merge(self._mother.model_dump(), self._overrides[name])
                self._cache[name] = Params(**merged)
            else:
                self._cache[name] = self._mother
        return self._cache[name]

# =============================================================================
# YAML CONFIGURATION GUIDE
# =============================================================================
# When crafting YAML configuration files, the `wiring` blocks can accept both 
# hardcoded numbers and mathematical string equations. 
# 
# Follow these three rules to avoid "Variable Soup" and crashes:
#
# 1. STRUCTURAL VARIABLES (Auto-injected - DO NOT declare in `variables`):
#    The system automatically calculates and provides the following variables 
#    based on your network size. You can use them freely in any string equation:
#      - I_n_nodes : Number of nodes in the Input layer
#      - R_n_nodes : Number of nodes in the Reservoir layer
#      - src_size  : Total number of nodes sending connections in a wiring block
#      - tgt_size  : Total number of nodes receiving connections in a wiring block
#      - n_nodes   : Alias for tgt_size
#    *Example usage:* k_max: "R_n_nodes"
#
# 2. CUSTOM HYPERPARAMETERS (MUST declare in `variables`):
#    If you invent a variable name because you want to grid-search it or easily
#    tweak it from the top of the file, you MUST define it in the `variables:` 
#    block under `model:`. If the math engine sees a string it doesn't recognize, 
#    it will crash.
#    *Example usage:* k_avg: "R_k_avg"  <-- R_k_avg must be in variables block
#
# 3. HARDCODED CONSTANTS (No quotes, no variables):
#    If a value is static and won't be modified by a grid search or override, 
#    just write the number. Do not create unnecessary variables.
#    *Example usage:* k_min: 0
#
# EXAMPLE WIRING BLOCK:
#    - type: random
#      source: R
#      target: R
#      k_avg: "R_k_avg"      # Custom hyperparameter (Rule 2)
#      k_min: 0              # Hardcoded constant (Rule 3)
#      k_max: "R_n_nodes"    # Auto-injected structural variable (Rule 1)
# =============================================================================

class Params(BaseModel):
    version: Literal['1.0'] = Field('1.0', description="Config schema version — bump when breaking changes require migration")
    model: ModelParams
    logging: LoggingParams = Field(LoggingParams())
    dataset: Optional[Union[KQGRTemporalDatasetParams, KQGRPathIntegrationDatasetParams, KQGRNeuroGymDatasetParams, PathIntegrationDatasetParams, TemporalDatasetParams, NeuroGymDatasetParams]] = None
    multiverse_overrides: Optional[dict] = Field(default=None, json_schema_extra={'expand': False})

    _universes: Optional[UniverseWrapper] = PrivateAttr(default=None)

    @property
    def U(self) -> UniverseWrapper:
        if self._universes is None:
            self._universes = UniverseWrapper(self, self.multiverse_overrides)
        return self._universes

    @property
    def M(self):
        return self.model

    @property
    def L(self):
        return self.logging

    @property
    def D(self):
        return self.dataset

    @property
    def dataset_init_obj(self):
        return self.D.init_obj

    @property
    def accuracy_obj(self):
        from project.boolean_reservoir.code.train_model import EuclideanDistanceAccuracy, BooleanAccuracy
        from benchmark.path_integration.parameter import PathIntegrationDatasetParams
        if self.M.T.accuracy is not None:
            return self.M.T.accuracy_obj
        # Infer from dataset type so existing YAML configs need no changes
        if isinstance(self.D, PathIntegrationDatasetParams):
            return EuclideanDistanceAccuracy().accuracy
        return BooleanAccuracy().accuracy


def load_yaml_config(filepath):
    with open(filepath, 'r') as file:
        config = yaml.safe_load(file)
    params = Params(**config)
    return params

def save_yaml_config(base_model: BaseModel, output_dir_path: Path, file_name='parameters', copy_from_original_file_path=None):
    if copy_from_original_file_path:
        shutil_copy(copy_from_original_file_path, output_dir_path / (file_name + '_original.yaml'))
    _save_yaml_config(base_model, output_dir_path / (file_name + '_short.yaml'), exclude_none=True, exclude_defaults=True)
    _save_yaml_config(base_model, output_dir_path / (file_name + '.yaml'))

def _save_yaml_config(base_model: BaseModel, filepath: Path, **kwargs):
    with open(filepath, 'w') as file:
        yaml.dump(base_model.model_dump(**kwargs), file)

if __name__ == '__main__':
    class Params(BaseModel):
        optim: Union[DynamicParams, List[DynamicParams]] = Field(
            # standard for RC in litterature (ridge)
            default=DynamicParams(name='adam', params={
                                  'lr': 1e-3, 'weight_decay': 1e-3}),
            description="Optimizer configuration"
        )
        save_keys: Optional[List[str]] = Field(
            default=['parameters'],
            description="Only save these model objects",
            json_schema_extra={'expand': False}  # Mark as non-expandable
        )
    p = Params(
        optim=[
            DynamicParams(
                name="adam",
                params=CallParams(
                    lr=[1e-1, 1e-2, 1e-3],
                    a=[1, 2],
                ),
            ),
            DynamicParams(
                name="adamw",
                params=CallParams(
                    lr=[1e-1, 1e-2, 1e-3],
                ),
            ),
        ]
    )
    from project.boolean_reservoir.code.utils.param_utils import generate_param_combinations
    P = generate_param_combinations(p)
    for p in P:
        print(p)
