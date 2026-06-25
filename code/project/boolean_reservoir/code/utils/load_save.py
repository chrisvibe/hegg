import hashlib
import json
import os
import warnings
import orjson
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
from pydantic import BaseModel
from enum import Enum
from project.boolean_reservoir.code.parameter import Params, load_yaml_config, deep_merge
from project.parallel_grid_search.code.run_layout import RUN
from enum import Enum
from inspect import getsource, getclosurevars
from typing import get_origin, get_args, Union, Type, Optional
import re

class _ParamsEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Path):
            return str(obj)
        if isinstance(obj, Enum):
            return obj.value
        return super().default(obj)


def save_grid_search_results(df: pd.DataFrame, path: Path):
    """Write grid search results to a Parquet file, atomically.

    Each call writes a fresh file — no append or dedup.  Callers are responsible
    for giving each batch a unique path (e.g. data/{node}_{seq}.parquet).
    df must have columns 'params' (Pydantic model or dict), 'i' (int), 'j' (int).
    """
    path = Path(path).with_suffix('.parquet')
    json_blobs = [
        json.dumps(p.model_dump() if hasattr(p, 'model_dump') else p, cls=_ParamsEncoder)
        for p in df['params']
    ]
    table = pa.table({
        'i':           pa.array(df['i'].tolist(), type=pa.int32()),
        'j':           pa.array(df['j'].tolist(), type=pa.int32()),
        'params_json': json_blobs,
    })
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.parquet.tmp')
    pq.write_table(table, tmp)
    os.replace(tmp, path)

class DotDict(dict):
    __slots__ = ('_tree', '_cls', '_merge_base')

    def __init__(self, data, cls=None, tree=None, _merge_base=None):
        super().__init__(data)
        object.__setattr__(self, '_cls', cls)
        object.__setattr__(self, '_merge_base', _merge_base)
        object.__setattr__(self, '_tree', tree or (DotDict._alias_tree(cls) if cls else {}))

    def __getattr__(self, key):
        merge_base = object.__getattribute__(self, '_merge_base')
        if merge_base is not None:
            # Universe view: key is the universe name; merge its overrides into base
            try:
                override = dict.__getitem__(self, key)
            except KeyError:
                return merge_base  # unknown universe → return base unchanged
            cls = object.__getattribute__(self, '_cls')
            return DotDict(deep_merge(dict(merge_base), override), cls=cls)

        if key == 'U':
            cls = object.__getattribute__(self, '_cls')
            overrides = dict.get(self, 'multiverse_overrides') or {}
            return DotDict(overrides, cls=cls, _merge_base=self)

        tree = object.__getattribute__(self, '_tree')
        resolved = tree.get('a', {}).get(key, key)
        try:
            val = self[resolved]
        except KeyError:
            raise AttributeError(key)
        if isinstance(val, dict):
            child_tree = tree.get('c', {}).get(resolved)
            return DotDict(val, tree=child_tree)
        return val

    _tree_cache: dict = {}

    @staticmethod
    def _alias_tree(cls: Type[BaseModel]) -> dict:
        """Build {'a': {alias: field}, 'c': {field: subtree}} for cls and children."""
        if cls in DotDict._tree_cache:
            return DotDict._tree_cache[cls]

        aliases = {}
        for name, obj in vars(cls).items():
            if isinstance(obj, property) and obj.fget:
                m = re.search(r'return self\.(\w+)\s*$', getsource(obj.fget), re.MULTILINE)
                if m:
                    aliases[name] = m.group(1)

        children = {}
        for fname, finfo in cls.model_fields.items():
            ann = finfo.annotation
            if get_origin(ann) is Union:
                ann = next((a for a in get_args(ann) if a is not type(None)), ann)
            if isinstance(ann, type) and issubclass(ann, BaseModel):
                children[fname] = DotDict._alias_tree(ann)

        result = {'a': aliases, 'c': children} if aliases or children else {}
        DotDict._tree_cache[cls] = result
        return result
    
    def to_pydantic(self):
        cls = object.__getattribute__(self, '_cls')
        if cls is None:
            raise ValueError("No Pydantic class associated")
        return cls.model_validate(dict(self))

def _read_data_dir(data_dir: Path, limit: Optional[int] = None) -> pd.DataFrame:
    """Concatenate all batch Parquet files in data_dir into a deduplicated DataFrame."""
    files = sorted(data_dir.glob('*.parquet'))
    if not files:
        return pd.DataFrame()
    tables = []
    for f in files:
        try:
            tables.append(pq.read_table(f))
        except Exception:
            pass
    if not tables:
        return pd.DataFrame()
    tbl = pa.concat_tables(tables)
    if limit is not None:
        tbl = tbl.slice(0, limit)
    df = tbl.to_pandas()
    if 'i' in df.columns and 'j' in df.columns:
        df = df.drop_duplicates(subset=['i', 'j'], keep='last').reset_index(drop=True)
    return df


def load_params_df(
    data_path: Path,
    model_class: Type[BaseModel] = Params,
    fast: bool = True,
    limit: Optional[int] = None,
    batch_size: Optional[int] = None,
    keep_params_json: bool = False,
) -> pd.DataFrame:
    """Load Parquet → DataFrame with hydrated 'params' column.

    If data.parquet does not exist, falls back to loading all batch Parquet files
    from the data/ subfolder (mid-run state).

    Args:
        data_path: Path to the parquet file (or its parent directory's data.parquet).
        model_class: Pydantic model class for validation.
        fast: If True, use DotDict (skip Pydantic validation) for faster loading.
              If False, use full Pydantic model_validate.
        limit: Maximum number of rows to return.
        batch_size: Number of rows to read per batch.
    """
    data_path = Path(data_path).with_suffix('.parquet')

    # Fall back to batch Parquet files in data/ subfolder (active or crashed run)
    if not data_path.exists():
        data_dir = data_path.parent / 'data'
        if data_dir.is_dir():
            df = _read_data_dir(data_dir, limit=limit)
            if not df.empty and 'params_json' in df.columns:
                if fast:
                    df['params'] = df['params_json'].apply(
                        lambda s: DotDict(orjson.loads(s), cls=Params)
                    )
                else:
                    df['params'] = df['params_json'].apply(
                        lambda s: model_class.model_validate(orjson.loads(s))
                    )
                if not keep_params_json:
                    df.drop(columns=['params_json'], inplace=True)
            return df

    # 1. Load data with optional row limit and batching
    if limit is not None or batch_size is not None:
        pf = pq.ParquetFile(data_path)
        batches = []
        rows_read = 0

        # Smart default: If no batch_size is provided, use the limit (if set)
        # to prevent fetching more rows than necessary. Otherwise fallback to PyArrow's default.
        _batch_size = batch_size if batch_size is not None else (limit if limit is not None else 65536)

        for batch in pf.iter_batches(batch_size=_batch_size):
            batches.append(batch)
            rows_read += batch.num_rows
            if limit is not None and rows_read >= limit:
                break

        if batches:
            table = pa.Table.from_batches(batches)
            if limit is not None:
                table = table.slice(length=limit)
            df = table.to_pandas()
        else:
            df = pd.DataFrame()

    else:
        # Fast path if we just want the whole file in memory
        df = pq.read_table(data_path).to_pandas()

    # 2. Process the 'params_json' column
    if not df.empty and 'params_json' in df.columns:
        if fast:
            df['params'] = df['params_json'].apply(
                lambda s: DotDict(orjson.loads(s), cls=Params)
            )
        else:
            df['params'] = df['params_json'].apply(
                lambda s: model_class.model_validate(orjson.loads(s))
            )
        if not keep_params_json:
            df.drop(columns=['params_json'], inplace=True)

    return df

def params_col_to_fields(df, extractions):
    """
    Projects structured parameter objects in `df['params']`
    into a new DataFrame of extracted fields.
    Args:
        df: DataFrame containing a `params` column.
        extractions: List of (prefix, getter, field_set) or
                     (prefix, getter, field_set, key_fn) tuples.
            - prefix: Column prefix or column name if capturing source.
            - getter: Function extracting a sub-model from params.
            - field_set: Set of field names to extract, empty set {} for all fields,
                        or None to capture source object.
            - key_fn: Optional. If provided, getter is called only once per unique
                      key — rows sharing the same key reuse the cached result.
    Returns:
        (new_df, factors):
            new_df: DataFrame with extracted fields (and captured sources).
            factors: List of extracted flattened column names.
    """
    params_list = list(df['params'])

    # Pre-compute sources for dedup extractions (4-tuple with key_fn)
    dedup_sources = {}  # extraction index → {key: source}
    for i, extraction in enumerate(extractions):
        if len(extraction) == 4:
            _, get_source, _, key_fn = extraction
            key_to_source = {}
            for params in params_list:
                key = key_fn(params)
                if key not in key_to_source:
                    key_to_source[key] = get_source(params)
            dedup_sources[i] = (key_fn, key_to_source)

    rows = []
    factors = []
    warned = set()
    for params in params_list:
        row = {}
        for i, extraction in enumerate(extractions):
            prefix, get_source, field_set = extraction[:3]
            if i in dedup_sources:
                key_fn, key_to_source = dedup_sources[i]
                source = key_to_source[key_fn(params)]
            else:
                source = get_source(params)
            if source is None:
                lambda_str = getsource(get_source).strip()
                if lambda_str not in warned:
                    print(f"Warning: Extraction source is None for extraction: {lambda_str}")
                    warned.add(lambda_str)
                continue
            if field_set is None:
                row[prefix] = source
                if not isinstance(source, (dict, BaseModel)) and prefix not in factors:
                    factors.append(prefix)
                continue

            dumped = source if isinstance(source, dict) else source.model_dump()
            fields_to_extract = dumped.keys() if not field_set else field_set

            for k in fields_to_extract:
                v = dumped.get(k)
                col = f"{prefix}_{k}"
                row[col] = str(v) if isinstance(v, Enum) else v
                if col not in factors:
                    factors.append(col)
        rows.append(row)
    return pd.DataFrame(rows), factors

def get_data_path(config_path, filename=RUN.compacted_file) -> Path:
    """Derive data path from config's out_path"""
    P = load_yaml_config(config_path)
    return P.L.out_path / 'data' / filename

def _extraction_cache_key(extraction, data_path: Path, limit, batch_size, df_filter_mask) -> str:
    """Cache key for one extraction applied to one data file."""
    prefix, getter, field_set = extraction[:3]
    key_fn = extraction[3] if len(extraction) == 4 else None
    def _fn_cache_vars(fn):
        if fn is None:
            return ''
        try:
            cv = getclosurevars(fn)
            stable = {}
            for k, v in {**cv.nonlocals, **cv.globals}.items():
                try:
                    stable[k] = getsource(v)
                except (TypeError, OSError):
                    stable[k] = repr(v)
            return repr(stable)
        except TypeError:
            return ''

    parts = (
        prefix,
        getsource(getter),
        _fn_cache_vars(getter),
        getsource(key_fn) if key_fn else '',
        str(sorted(field_set)) if field_set is not None else 'None',
        (str(data_path), data_path.stat().st_mtime, data_path.stat().st_size),
        limit, batch_size,
        getsource(df_filter_mask) if df_filter_mask else '',
    )
    return 'e_' + hashlib.md5(str(parts).encode()).hexdigest()


def _cache_files(extraction, path: Path, limit, batch_size, df_filter_mask,
                 cache_dir: Path) -> tuple[Path, Path]:
    """Return (parquet_path, factors_json_path) for one extraction × data file."""
    ekey = _extraction_cache_key(extraction, path, limit, batch_size, df_filter_mask)
    return (cache_dir / f'{ekey}.extraction.parquet',
            cache_dir / f'{ekey}.extraction.factors.json')


def _extraction_columns(extraction, df) -> list[str]:
    """Return the column names that an extraction produced in df."""
    prefix, _, field_set = extraction[:3]
    if field_set is None:
        return [prefix] if prefix in df.columns else []
    if not field_set:
        return [c for c in df.columns if c.startswith(f'{prefix}_')]
    return [f'{prefix}_{k}' for k in field_set if f'{prefix}_{k}' in df.columns]


_DEFAULT_CACHE_DIR = Path(os.environ.get('BOOLEAN_RESERVOIR_CACHE_DIR', '/tmp/boolean_reservoir/cache/custom_load_grid_search_data'))


def is_cache_warm(config_paths, extractions, cache_dir: Path | str = _DEFAULT_CACHE_DIR) -> bool:
    """Return True if every data-present (config_path × extraction) already has cache files.

    Checks file existence only — no data is loaded.
    Config paths with no data file are ignored (the loader skips them anyway).
    """
    cache_dir = Path(cache_dir)
    for cp in config_paths:
        path = get_data_path(cp)
        if not path.exists():
            continue
        for extraction in extractions:
            ep, fp = _cache_files(extraction, path, None, None, None, cache_dir)
            if not ep.exists() or not fp.exists():
                return False
    return True


def custom_load_grid_search_data(data_paths=None, config_paths=None, extractions=None, df_filter_mask=None, filename=RUN.compacted_file, limit=None, batch_size=None, keep_params_json: bool = False, cache_dir: Optional[Path | str] = _DEFAULT_CACHE_DIR, warmup: bool = False) -> tuple[pd.DataFrame, list[str]]:
    """Core loader with per-extraction caching.

    Each extraction is cached independently so that changing one extraction only
    recomputes that extraction. Params JSON is parsed at most once per call.

    Args:
        cache_dir: Directory for per-extraction cache files. Pass None to disable.
        warmup: If True, populate missing cache files then return immediately without
                loading data into memory. Cache-hit extractions are not read from disk.
    """
    if data_paths is None and config_paths is None:
        raise ValueError("Must provide data_paths or config_paths")

    if data_paths is None:
        if isinstance(config_paths, (str, Path)):
            config_paths = [config_paths]
        data_paths = [get_data_path(p, filename) for p in config_paths]

    if isinstance(data_paths, (str, Path)):
        data_paths = [data_paths]

    data_paths = [Path(p) for p in data_paths]

    use_cache = cache_dir is not None and extractions
    if use_cache:
        cache_dir = Path(cache_dir)

    # ── Load and extract (per data file) ─────────────────────────────────────
    dfs = []
    all_factors: list[str] = []

    for path in data_paths:
        # ── Per-path per-extraction cache check ──────────────────────────────
        cached_slabs: dict[int, pd.DataFrame] = {}
        cached_factors: dict[int, list] = {}
        uncached_indices: list[int] = []

        if use_cache:
            for i, extraction in enumerate(extractions):
                ep, fp = _cache_files(extraction, path, limit, batch_size, df_filter_mask, cache_dir)
                if ep.exists() and fp.exists():
                    if not warmup:
                        cached_slabs[i] = pd.read_parquet(ep)
                        cached_factors[i] = json.loads(fp.read_text())
                    print(f'  [cache hit] {path.name}  extraction {i} ({extraction[0]})')
                else:
                    uncached_indices.append(i)
        if warmup and not uncached_indices:
            continue
        elif not use_cache:
            uncached_indices = list(range(len(extractions))) if extractions else []
        elif not uncached_indices:
            print(f'  [all cached] {path.name}, skipping params load', file=__import__('sys').stderr, flush=True)

        # Parse params only if at least one extraction needs it for this path
        df_raw = None
        if not extractions or uncached_indices:
            df_raw = load_params_df(data_path=path, limit=limit, batch_size=batch_size,
                                    keep_params_json=keep_params_json)

        if not extractions:
            dfs.append(df_raw)
            continue

        # Compute uncached extractions in a single params pass
        fresh_slabs: dict[int, pd.DataFrame] = {}
        fresh_factors: dict[int, list] = {}
        if uncached_indices:
            batch = [extractions[i] for i in uncached_indices]
            combined, batch_factors = params_col_to_fields(df_raw, batch)
            for abs_i, extraction in zip(uncached_indices, batch):
                cols = _extraction_columns(extraction, combined)
                slab = combined[cols].reset_index(drop=True)
                e_factors = [c for c in cols if c in batch_factors]
                fresh_slabs[abs_i] = slab
                fresh_factors[abs_i] = e_factors

            if use_cache:
                cache_dir.mkdir(parents=True, exist_ok=True)
                for abs_i, extraction in zip(uncached_indices, batch):
                    ep, fp = _cache_files(extraction, path, limit, batch_size, df_filter_mask, cache_dir)
                    tmp_ep = ep.with_suffix('.parquet.tmp')
                    tmp_fp = fp.with_suffix('.json.tmp')
                    fresh_slabs[abs_i].to_parquet(tmp_ep, index=False)
                    tmp_ep.replace(ep)
                    tmp_fp.write_text(json.dumps(fresh_factors[abs_i]))
                    tmp_fp.replace(fp)
                    print(f'  [saved]     {path.name}  extraction {abs_i} ({extraction[0]})')
        if warmup:
            continue

        # Assemble columns in original extraction order
        n_rows = len(df_raw) if df_raw is not None else len(next(iter(cached_slabs.values())))
        parts = []
        for i in range(len(extractions)):
            slab = fresh_slabs[i] if i in fresh_slabs else cached_slabs.get(i)
            if slab is not None:
                parts.append(slab.reset_index(drop=True))
                for f in (fresh_factors.get(i) or cached_factors.get(i) or []):
                    if f not in all_factors:
                        all_factors.append(f)

        row_df = pd.concat(parts, axis=1) if parts else pd.DataFrame(index=range(n_rows))

        # Always carry i, j (config/sample identifiers)
        if df_raw is not None:
            for col in ('i', 'j'):
                if col in df_raw.columns:
                    row_df[col] = df_raw[col].reset_index(drop=True)
        else:
            avail = pq.read_schema(path).names
            ij_cols = [c for c in ('i', 'j') if c in avail]
            if ij_cols:
                tbl = pq.read_table(path, columns=ij_cols)
                if limit:
                    tbl = tbl.slice(0, limit)
                for col in ij_cols:
                    row_df[col] = tbl.to_pandas()[col].values

        if keep_params_json:
            if df_raw is not None:
                if 'params_json' in df_raw.columns:
                    row_df['params_json'] = df_raw['params_json'].reset_index(drop=True)
            else:
                avail = pq.read_schema(path).names
                if 'params_json' in avail:
                    tbl = pq.read_table(path, columns=['params_json'])
                    if limit:
                        tbl = tbl.slice(0, limit)
                    row_df['params_json'] = tbl.to_pandas()['params_json'].values

        if df_filter_mask:
            row_df = row_df[df_filter_mask(row_df)]

        dfs.append(row_df)

    if warmup:
        return pd.DataFrame(), []

    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='The behavior of DataFrame concatenation with empty or all-NA entries', category=FutureWarning)
        df = pd.concat(dfs, ignore_index=True) if len(dfs) > 1 else dfs[0]

    return df, all_factors


def make_combo_column(df, factors, keep=None, exclude=None, return_as_str=True):
    if keep is not None:
        factors_subset = [f for f in factors if f in keep]
    elif exclude is not None:
        factors_subset = [f for f in factors if f not in exclude]
    else:
        factors_subset = factors

    if return_as_str:
        first, *rest = factors_subset if factors_subset else [factors_subset]
        s = df[first].astype(str)
        combo = s.str.cat([df[c].astype(str) for c in rest], sep='_') if rest else s
    else:
        combo = df[factors_subset].apply(tuple, axis=1)

    return combo, factors_subset


def fix_factors_and_combo(df, factors=list(), keep=None, exclude=None):
    # need more than one to be considered a factor (and be part of a set)
    factors = [f for f in factors if f in df.columns]
    factors = list(df[factors].nunique()[df[factors].nunique() > 1].index)
    df['combo'], _ = make_combo_column(df, factors, return_as_str=False, keep=keep, exclude=exclude)
    df['combo_str'] = df['combo'].apply(lambda t: "_".join(map(str, t)))
    df['combo_id'] = df['combo_str'].astype('category').cat.codes
    return df, factors


def load_grid_search_data(data_paths=None, config_paths=None, extractions=None, df_filter_mask=None, filename=RUN.compacted_file, keep_params_json: bool = False) -> tuple[pd.DataFrame, list[str]]:
    """Convenience loader with default train_log extraction"""
    if extractions is None:
        extractions = [
            ('P', lambda p: p, None),
            ('T', lambda p: p.L.T, {'accuracy', 'loss'}),
        ]

    return custom_load_grid_search_data(data_paths=data_paths, config_paths=config_paths, extractions=extractions, df_filter_mask=df_filter_mask, filename=filename, keep_params_json=keep_params_json)