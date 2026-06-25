"""
Shared utilities for Dash dashboard apps (scatter, polar, trace).
"""
import json
from pathlib import Path

# ── Aggregation ───────────────────────────────────────────────────────────────

_AGG_OPTIONS = [
    {'label': 'Mean',   'value': 'mean'},
    {'label': 'Median', 'value': 'median'},
    {'label': 'Min',    'value': 'min'},
    {'label': 'Max',    'value': 'max'},
    {'label': 'Std',    'value': 'std'},
]
_AGG_OPTIONS_WITH_RAW = [{'label': 'None (raw)', 'value': 'raw'}] + _AGG_OPTIONS
_DEFAULT_AGG = 'mean'


def aggregate_df(plot_df, x_col, y_col, group_by_cols, y_agg, x_is_binned=False, extra_agg_cols=None):
    if y_agg == 'raw' or (not group_by_cols and not x_is_binned):
        return plot_df
    group_cols = list(dict.fromkeys(group_by_cols + [x_col]))
    group_cols = [c for c in group_cols if c in plot_df.columns]
    if y_col in group_cols:
        return plot_df
    cols_to_agg = [c for c in ([y_col] + (extra_agg_cols or []))
                   if c in plot_df.columns and c not in group_cols]
    named_agg = {col: (col, y_agg) for col in cols_to_agg}
    named_agg['_agg_count'] = (cols_to_agg[0], 'count')
    agged = plot_df.groupby(group_cols, dropna=False).agg(**named_agg).reset_index()
    if 'design' not in agged.columns:
        first, *rest = group_cols
        s = agged[first].astype(str)
        if rest:
            s = s.str.cat([agged[c].astype(str) for c in rest], sep=' | ')
        agged['design'] = s
    return agged

# ── View file I/O ─────────────────────────────────────────────────────────────

def _views_dir(path_str):
    return Path(path_str.strip()) if path_str and path_str.strip() else Path('/tmp/dashboard')

def _list_views(views_dir):
    if not views_dir.exists():
        return []
    return sorted(p.stem for p in views_dir.glob('*.json'))

def _view_dropdown_options(views_dir):
    return [{'label': name, 'value': name} for name in _list_views(views_dir)]

def view_file_path(name: str, vdir: Path) -> Path:
    return vdir / f'{name}.json'

def save_view_json(name: str, data: dict, vdir: Path) -> Path:
    vdir.mkdir(parents=True, exist_ok=True)
    p = view_file_path(name, vdir)
    p.write_text(json.dumps(data, indent=2, default=str))
    return p

def load_view_json(name: str, vdir: Path) -> 'dict | None':
    p = view_file_path(name, vdir)
    return json.loads(p.read_text()) if p.exists() else None

def delete_view_json(name: str, vdir: Path) -> bool:
    p = view_file_path(name, vdir)
    if p.exists():
        p.unlink()
        return True
    return False

def _is_safe_path(p: Path, safe_roots: list) -> bool:
    resolved = p.resolve()
    return any(resolved == r or r in resolved.parents for r in safe_roots)

def _fmt(n: int) -> str:
    if n >= 1_000_000: return f'{n/1_000_000:.1f}M'
    if n >= 10_000:    return f'{n/1_000:.0f}k'
    if n >= 1_000:     return f'{n/1_000:.1f}k'
    return str(n)

def _make_eval_ns(df):
    import numpy as np
    ns = {col: df[col] for col in df.columns}
    ns['np'] = np
    ns.update({'int': int, 'float': float, 'str': str, 'bool': bool,
               'True': True, 'False': False, 'None': None, 'abs': abs,
               'round': round, 'min': min, 'max': max, 'len': len, 'list': list})
    return ns

def parse_and_apply_fields(working_df, fields_text):
    """Parse 'name = expr' lines and eval onto df. Returns (df, info, error)."""
    if not fields_text or not fields_text.strip():
        return working_df, '', ''
    errors = []
    added = []
    eval_ns = None
    for i, line in enumerate(fields_text.strip().splitlines(), 1):
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        if '=' not in line:
            errors.append(f'L{i}: missing "=" in "{line}"')
            continue
        name, expr = line.split('=', 1)
        name = name.strip()
        expr = expr.strip()
        if not name or not expr:
            errors.append(f'L{i}: empty name or expression')
            continue
        try:
            working_df[name] = working_df.eval(expr)
            added.append(name)
        except Exception:
            try:
                if eval_ns is None:
                    eval_ns = _make_eval_ns(working_df)
                result = eval(expr, {"__builtins__": {}}, eval_ns)  # noqa: S307
                working_df[name] = result
                eval_ns[name] = working_df[name]
                added.append(name)
            except Exception as e:
                errors.append(f'L{i} ({name}): {e}')
    return working_df, (f'Added: {", ".join(added)}' if added else ''), ('\n'.join(errors) if errors else '')

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


def apply_trace_vis(fig, trace_visibility):
    """Apply a {name|mode: visible} dict to figure traces in-place.

    Accepts either a go.Figure or a plain dict (from fig.to_dict()).
    """
    if not trace_visibility:
        return
    if hasattr(fig, 'data'):
        for trace in fig.data:
            key = f"{trace.name or ''}|{getattr(trace, 'mode', None) or 'markers'}"
            if key in trace_visibility:
                trace.visible = trace_visibility[key]
    else:
        for t in fig.get('data', []):
            key = f"{t.get('name') or ''}|{t.get('mode') or 'markers'}"
            if key in trace_visibility:
                t['visible'] = trace_visibility[key]

_MIME = {'png': 'image/png', 'svg': 'image/svg+xml', 'pdf': 'application/pdf'}


def _strip_interactive_svg(svg: str) -> str:
    """Fix Plotly SVG for static rendering.

    Drag hitbox elements (angulardrag, radialdrag, subplot drag rects) use
    'fill: transparent' as a CSS colour value.  Static SVG renderers treat
    it as opaque black, producing thick artefacts.  'fill: none' is the
    correct SVG idiom for no fill and is safe to substitute globally because
    Plotly only uses 'fill: transparent' on these invisible hitboxes.
    """
    return svg.replace('fill: transparent', 'fill: none')


_MERGE_SVG_JS = """
    () => {
        const svgs = Array.from(document.querySelectorAll('.main-svg'));
        if (svgs.length === 1) return svgs[0].outerHTML;
        const root = svgs[0].cloneNode(true);
        for (const svg of svgs.slice(1))
            for (const child of svg.children)
                root.appendChild(child.cloneNode(true));
        return root.outerHTML;
    }
"""

def _render_html_to_image(html: str, width: int, height: int, fmt: str = 'png') -> bytes:
    from playwright.sync_api import sync_playwright
    with sync_playwright() as p:
        browser = p.chromium.launch(args=['--no-sandbox', '--disable-dev-shm-usage', '--disable-gpu'])
        page = browser.new_page(viewport={'width': width, 'height': height})
        page.set_content(html, wait_until='networkidle')
        if fmt in ('svg', 'pdf'):
            svg = _strip_interactive_svg(page.evaluate(_MERGE_SVG_JS))
            browser.close()
            if fmt == 'svg':
                return svg.encode()
            import cairosvg
            return cairosvg.svg2pdf(bytestring=svg.encode(), output_width=width, output_height=height)
        img_bytes = page.screenshot(full_page=False, type='png')
        browser.close()
        return img_bytes


def render_figure_image(fig, width: int, height: int, fmt: str = 'png') -> bytes:
    import plotly.io as pio
    return pio.to_image(fig, format=fmt, width=width, height=height)


def render_figure_png(fig, width: int, height: int) -> bytes:
    return render_figure_image(fig, width, height, fmt='png')


def register_export_route(app, reconstruct_fn, safe_roots: list,
                       default_search_dirs: list = None):
    """
    Register a ?<view_name>&format=<fmt> GET handler on app.server (Flask).

    Supported formats: png, svg

    reconstruct_fn may be:
      - a callable(view_data, width, height, theme_override) -> Figure
      - a dict[str, callable] — dispatched via ?target=<key>; defaults to the
        first key when ?target is omitted.

    Usage
    -----
        # single-target (callable)
        curl 'http://host:port/?my_view&format=png&width=1200&height=800' -o out.png

        # multi-target (dict)
        curl 'http://host:port/?my_view&format=png&target=polar&width=1200&height=800' -o polar.png
        curl 'http://host:port/?my_view&format=png&target=table&width=600&height=400'  -o table.png
    """
    import urllib.parse as _urlparse
    import orjson
    from flask import request as _req, Response as _Resp

    if default_search_dirs is None:
        default_search_dirs = [Path('/out/dashboard'), Path('/tmp/dashboard')]

    _safe     = [Path(r) for r in safe_roots]
    _defaults = [Path(d) for d in default_search_dirs]
    _multi    = isinstance(reconstruct_fn, dict)

    @app.server.before_request
    def _maybe_serve_image():
        args = _req.args
        fmt  = args.get('format', '').lower()
        if fmt not in _MIME:
            return None

        raw_qs    = _req.query_string.decode('utf-8')
        view_name = None
        for part in raw_qs.split('&'):
            decoded = _urlparse.unquote(part)
            if '=' not in decoded and decoded.strip():
                view_name = decoded.strip()
                break

        if not view_name:
            return _Resp(f'Missing view name — use ?<view_name>&format={fmt}',
                         status=400, content_type='text/plain; charset=utf-8')

        width  = int(args.get('width',  1200))
        height = int(args.get('height', 800))
        theme_param    = args.get('theme')
        theme_override = (['light'] if theme_param == 'light'
                          else (['dark'] if theme_param == 'dark' else None))

        if _multi:
            target = args.get('target', next(iter(reconstruct_fn)))
            fn = reconstruct_fn.get(target)
            if fn is None:
                return _Resp(
                    f'Unknown target "{target}". Valid targets: {list(reconstruct_fn)}',
                    status=400, content_type='text/plain; charset=utf-8')
        else:
            fn = reconstruct_fn

        if args.get('dir'):
            explicit = _views_dir(args.get('dir'))
            if not _is_safe_path(explicit, _safe):
                return _Resp(f'dir must be under {[str(r) for r in _safe]}',
                             status=400, content_type='text/plain; charset=utf-8')
            search_dirs = [explicit]
        else:
            search_dirs = _defaults

        found = None
        for d in search_dirs:
            c = d / f'{view_name}.json'
            if c.exists():
                found = c
                break

        if found is None:
            searched  = [str(d / f'{view_name}.json') for d in search_dirs]
            available = [p.stem for d in search_dirs if d.exists()
                         for p in sorted(d.glob('*.json'))]
            lines = [
                f'View "{view_name}" not found.',
                f'Searched: {searched}',
                f'Available: {available}' if available else 'No views saved yet.',
                'Save the view first via the Views panel in the dashboard.',
            ]
            return _Resp('\n'.join(lines), status=404,
                         content_type='text/plain; charset=utf-8')

        try:
            view_data = orjson.loads(found.read_bytes())
            fig = fn(view_data, width, height, theme_override)
        except Exception as e:
            import sys, traceback
            print(f'[export] reconstruct failed for "{view_name}": {e}\n'
                  + traceback.format_exc(), file=sys.stderr, flush=True)
            return _Resp(f'Failed to reconstruct view "{view_name}": {e}',
                         status=500, content_type='text/plain; charset=utf-8')

        try:
            img_bytes = render_figure_image(fig, width, height, fmt=fmt)
        except Exception as e:
            import sys, traceback
            print(f'[export] render failed for "{view_name}" ({fmt}): {e}\n'
                  + traceback.format_exc(), file=sys.stderr, flush=True)
            return _Resp(f'{fmt.upper()} rendering failed: {e}',
                         status=500, content_type='text/plain; charset=utf-8')

        return _Resp(img_bytes, mimetype=_MIME[fmt],
                     headers={'Content-Disposition':
                               f'inline; filename="{view_name}.{fmt}"'})
