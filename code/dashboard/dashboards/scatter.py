from pathlib import Path
import colorsys
import plotly.io as pio
from dash import Dash, dcc, html, callback, Input, Output, State, ALL, ctx, MATCH, no_update, Patch
from dash.exceptions import PreventUpdate
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from hashlib import md5
from dash.dependencies import Input, Output, State
import yaml
import orjson
import traceback as _traceback
import time as _time

from .scatter_layout import (
    build_layout, make_collapsible_panel,
    _DARK, _DARK3, _BORDER, _TEXT, _TEXT_DIM,
    btn_sm, SELECTION_VALUE, DARK_CSS,
)
from .utils import (
    register_export_route, make_combo_column,
    _views_dir, _list_views, _view_dropdown_options, _is_safe_path,
    parse_and_apply_fields, _fmt, apply_trace_vis,
    save_view_json, load_view_json, delete_view_json,
    aggregate_df, _AGG_OPTIONS_WITH_RAW,
)


def _integer_tick_kwargs(v_min: int, v_max: int, max_ticks: int = 20) -> dict:
    import math
    span = v_max - v_min
    if span <= max_ticks:
        return {'tickmode': 'array', 'tickvals': list(range(v_min, v_max + 1))}
    dtick = max(1, math.ceil(span / max_ticks))
    return {'tick0': v_min, 'dtick': dtick, 'tickformat': 'd'}

def _sorted_groups(series) -> list:
    groups = sorted(v for v in series.unique() if v != 'Other')
    if (series == 'Other').any():
        groups.append('Other')
    return groups

def _drop_constant_cols(df, cols):
    return [c for c in cols if df[c].nunique() > 1]



def _detect_gpu():
    """Detect if GPU-accelerated WebGL is likely available on the client."""
    import os, subprocess
    if not os.environ.get('DISPLAY') and not os.environ.get('WAYLAND_DISPLAY'):
        return True
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            return True
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    try:
        result = subprocess.run(['lspci'], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            for line in result.stdout.lower().split('\n'):
                if 'vga' in line or '3d controller' in line or 'display' in line:
                    return True
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return False

_COLOR_SPECIAL = frozenset({SELECTION_VALUE, '_header_discrete', '_header_continuous', '_header_derived'})

_HUE_STOPS = [0, 137, 275, 52, 190, 327, 105, 242, 20, 158]

def _to_light_color(color: str) -> str:
    color = color.strip()
    if color.startswith('rgb'):
        nums = color[color.index('(')+1:color.index(')')].split(',')
        r, g, b = [float(x.strip()) / 255 for x in nums]
    else:
        h_str = color.lstrip('#')
        r, g, b = [int(h_str[i:i+2], 16) / 255 for i in (0, 2, 4)]
    h, s, v = colorsys.rgb_to_hsv(r, g, b)
    r2, g2, b2 = colorsys.hsv_to_rgb(h, min(1.0, s * 1.1), v * 0.65)
    return '#{:02x}{:02x}{:02x}'.format(int(r2 * 255), int(g2 * 255), int(b2 * 255))

_PALETTE_DARK = (
    px.colors.qualitative.Dark24 +
    px.colors.qualitative.Vivid +
    px.colors.qualitative.Alphabet
)

_PALETTE_LIGHT = [_to_light_color(c) for c in _PALETTE_DARK]

def _stratified_sample(df, n, color_factors):
    """Sample n rows from df, stratified by color group so small groups aren't dropped."""
    strat_cols = [f for f in (color_factors or [])
                  if f and f not in _COLOR_SPECIAL and f in df.columns]
    if not strat_cols:
        return df.sample(n, random_state=None)
    key = (df[strat_cols[0]].astype(str) if len(strat_cols) == 1
           else df[strat_cols[0]].astype(str).str.cat(
               [df[c].astype(str) for c in strat_cols[1:]], sep='_'))
    per_group = max(1, n // key.nunique())
    sampled = (df.assign(_strat=key)
               .groupby('_strat', group_keys=False)
               .apply(lambda g: g.sample(min(len(g), per_group), random_state=None))
               .drop(columns='_strat', errors='ignore'))
    if len(sampled) < n:
        remainder = df.loc[~df.index.isin(sampled.index)]
        if len(remainder) > 0:
            sampled = pd.concat([sampled,
                                 remainder.sample(min(n - len(sampled), len(remainder)),
                                                  random_state=None)])
    return sampled

def _get_overlay_ks(view_data: dict) -> list:
    """Resolve per-line k/s from saved view data (handles both old and new format)."""
    ks = view_data.get('overlay-spline-ks')
    if ks:
        return ks
    k = view_data.get('overlay-spline-k')
    s = view_data.get('overlay-spline-s')
    return [[k, s]] if k is not None or s is not None else [[1, None]]

def _add_line_overlay(fig, df, x_col, y_col, group_col, color_map, overlay_ks=None,
                      use_webgl=False, anchor_left=False, anchor_right=False, ks_offset=0,
                      line_width=2, line_opacity=1.0):
    """Add one smoothed line trace per group onto fig, matching dot colors."""
    from scipy.interpolate import UnivariateSpline
    ScatterType = go.Scattergl if use_webgl else go.Scatter
    ks_list  = list(overlay_ks or [])
    default  = ks_list[0] if ks_list else [1, None]
    for i, (group_label, color) in enumerate((color_map or {}).items()):
        if i >= _MAX_OVERLAY_GROUPS:
            break
        ks = ks_list[i + ks_offset] if (i + ks_offset) < len(ks_list) else default
        k  = max(1, min(5, int(ks[0]))) if ks[0] is not None else 1
        s  = float(ks[1]) if ks[1] is not None else 0.0
        sub = (df[df[group_col] == group_label] if group_col is not None else df)
        sub = sub.dropna(subset=[x_col, y_col]).drop_duplicates(subset=[x_col]).sort_values(x_col)
        if len(sub) < k + 1:
            continue
        x_vals = sub[x_col].values.astype(float)
        y_vals = sub[y_col].values.astype(float)
        try:
            w = np.ones(len(x_vals))
            if anchor_left:
                w[0] = 1e6
            if anchor_right:
                w[-1] = 1e6
            spl = UnivariateSpline(x_vals, y_vals, k=k, s=s, w=w)
            x_line = np.linspace(x_vals.min(), x_vals.max(), 200)
            y_line = spl(x_line)
        except Exception:
            x_line, y_line = x_vals, y_vals  # fall back to connect-dots
        fig.add_trace(ScatterType(
            x=x_line, y=y_line,
            mode='lines',
            line=dict(color=color, width=line_width),
            opacity=line_opacity,
            name=group_label,
            showlegend=True,
        ))

_COSMETIC_TRIGGERS = frozenset({
    'color-offsets', 'legend-show-factor-name',
    'overlay-dots-toggle', 'overlay-line-toggle',
    'overlay-line-width', 'overlay-line-opacity',
    'overlay-spline-k', 'overlay-spline-s',
    'overlay-per-line-toggle', 'overlay-spline-ks',
    'overlay-anchor-left', 'overlay-anchor-right',
    'overlay-y-styles',
    'color-as-continuous',
    'color-by-values',
    'skip-sparse-factors',
})

_MAX_OVERLAY_GROUPS = 15

_Y_STYLES = [
    {'symbol': 'circle',        'dash': 'solid'},
    {'symbol': 'square',        'dash': 'dash'},
    {'symbol': 'diamond',       'dash': 'dot'},
    {'symbol': 'cross',         'dash': 'dashdot'},
    {'symbol': 'x',             'dash': 'longdash'},
    {'symbol': 'triangle-up',   'dash': 'solid'},
    {'symbol': 'triangle-down', 'dash': 'dash'},
    {'symbol': 'star',          'dash': 'dot'},
]

class ScatterDashboard:
    """
    Encapsulates a Plotly/Dash scatter dashboard with view save/load,
    server-side PNG export, and a two-stage data/figure callback pipeline.

    Usage:
        app = ScatterDashboard(df, factors).build()
        app.run(debug=False)
    """

    def __init__(
        self,
        df: pd.DataFrame,
        factors: list[str],
        discrete_threshold: int = 100,
        renderer: str = 'auto',
        initial_display_points: int = 10_000,
        safe_roots: list[str] = None,
        initial_views_dir: str = '/out/dashboard',
        url_prefix: str = '/',
    ):
        if safe_roots is None:
            safe_roots = ['/out', '/tmp']
        self._initial_views_dir = initial_views_dir
        self._url_prefix = url_prefix

        if renderer == 'auto':
            self.use_webgl = _detect_gpu()
        elif renderer == 'webgl':
            self.use_webgl = True
        else:
            self.use_webgl = False
        print(f"Scatter dashboard renderer: {'WebGL (Scattergl)' if self.use_webgl else 'SVG (Scatter)'}")

        self.df = df.copy()
        self.has_params_json = 'params_json' in df.columns
        self.initial_display_points = initial_display_points
        self.discrete_threshold = discrete_threshold
        self._safe_roots = [Path(r) for r in safe_roots]

        # Factor analysis
        self.df['design'], self.factors_subset = make_combo_column(self.df, factors, return_as_str=True)
        candidate_factors = list(self.factors_subset)
        self.discrete_factors   = [f for f in candidate_factors if self.df[f].nunique() < discrete_threshold]
        self.continuous_factors = [f for f in candidate_factors if self.df[f].nunique() >= discrete_threshold]
        self.numeric_discrete_factors = [
            f for f in self.discrete_factors if pd.api.types.is_numeric_dtype(self.df[f])
        ]
        self.factor_values = {
            f: [str(v) for v in sorted(v for v in self.df[f].unique() if pd.notna(v))]
            for f in self.discrete_factors
        }

        # Axis options
        numeric_cols = self.df.select_dtypes(include='number').columns.tolist()
        self.axis_options = list(dict.fromkeys(numeric_cols))
        self.default_x = 'R_k_avg'   if 'R_k_avg'   in self.axis_options else (self.axis_options[0] if self.axis_options else None)
        self.default_y = 'T_accuracy' if 'T_accuracy' in self.axis_options else (self.axis_options[1] if len(self.axis_options) > 1 else None)

        # Color — active palette is selected per-render in _build_figure via self._dark_mode
        self._dark_mode = True
        self.color_by_options = [
            {'label': '\u2b21 Filter Selection', 'value': SELECTION_VALUE},
            {'label': '\u2500\u2500 Discrete \u2500\u2500', 'value': '_header_discrete', 'disabled': True},
        ]
        for f in self.discrete_factors:
            self.color_by_options.append({'label': f'  {f}', 'value': f})
        if self.continuous_factors:
            self.color_by_options.append(
                {'label': '\u2500\u2500 Continuous \u2500\u2500', 'value': '_header_continuous', 'disabled': True}
            )
            for f in self.continuous_factors:
                self.color_by_options.append({'label': f'  {f}', 'value': f})

        self.ScatterType = go.Scattergl if self.use_webgl else go.Scatter
        self.app = None  # set in build()
        self._pipeline_cache_key: object = None
        self._pipeline_cache_val: tuple | None = None

    # ── Color helpers ─────────────────────────────────────────────────────────

    def _get_stable_color(self, combo_str, offset=0):
        if combo_str == 'Other':
            return '#CCCCCC' if self._dark_mode else '#888888'
        palette = _PALETTE_DARK if self._dark_mode else _PALETTE_LIGHT
        if isinstance(offset, dict):
            if 'hsv' in offset:
                h, s, v = offset['hsv']
                r, g, b = colorsys.hsv_to_rgb(h / 360.0, s / 100.0, v / 100.0)
                return '#{:02x}{:02x}{:02x}'.format(int(r * 255), int(g * 255), int(b * 255))
            if 'h' in offset:
                h_deg = _HUE_STOPS[offset['h'] % len(_HUE_STOPS)]
                s, v = (1.0, 1.0) if self._dark_mode else (0.85, 0.75)
                r, g, b = colorsys.hsv_to_rgb(h_deg / 360.0, s, v)
                return '#{:02x}{:02x}{:02x}'.format(int(r * 255), int(g * 255), int(b * 255))
            return palette[offset['d'] % len(palette)]
        hash_value = int(md5(f'{combo_str}_{offset}'.encode()).hexdigest(), 16)
        return palette[hash_value % len(palette)]

    def _make_color_map(self, unique_groups, color_offsets):
        return {
            g: self._get_stable_color(g, (color_offsets or {}).get(g, 0))
            for g in unique_groups
        }

    # ── Trace visibility helper ───────────────────────────────────────────────

    @staticmethod
    def _apply_trace_vis(fig, trace_visibility):
        apply_trace_vis(fig, trace_visibility)

    # ── Figure sub-builders ───────────────────────────────────────────────────

    def _make_discrete_figure(self, filtered_df, x_col, y_col, col_name, label, hover_base, color_offsets, cd_cols, show_label=True):
        unique_groups = _sorted_groups(filtered_df[col_name])
        color_map = self._make_color_map(unique_groups, color_offsets)
        fig = px.scatter(
            filtered_df, x=x_col, y=y_col,
            color=col_name, color_discrete_map=color_map,
            custom_data=cd_cols,
            category_orders={col_name: unique_groups},
            labels={col_name: label if show_label else ''},
            render_mode='webgl' if self.use_webgl else 'svg',
        )
        fig.update_traces(hovertemplate=hover_base)
        return fig

    def _make_uniform_figure(self, filtered_df, x_col, y_col, hover_base, cd_cols):
        return go.Figure(self.ScatterType(
            x=filtered_df[x_col], y=filtered_df[y_col],
            mode='markers', marker=dict(color='steelblue'), name='',
            customdata=filtered_df[cd_cols].values,
            hovertemplate=hover_base,
        ))

    # ── Main figure builder ───────────────────────────────────────────────────

    def _build_figure(self, filtered_df, x_col, y_col, color_cfg=None, color_offsets=None,
                      x_jitter=None, y_jitter=None, y_agg_label=None,
                      x_label=None, marker_size=5, marker_opacity=0.7,
                      overlay_line=False, overlay_ks=None, overlay_dots=True,
                      anchor_left=False, anchor_right=False,
                      template='plotly_dark', show_factor_name=True,
                      skip_sparse=True, line_width=2, line_opacity=1.0,
                      _overlay_info_out=None):
        if not x_col or not y_col:
            fig = go.Figure()
            fig.update_layout(title='Select X and Y axes to display data', xaxis_title='X', yaxis_title='Y')
            return fig
        if filtered_df is None or filtered_df.empty:
            fig = go.Figure()
            fig.update_layout(title='No data for selected filters', xaxis_title=x_col, yaxis_title=y_col)
            return fig

        self._dark_mode = (template == 'plotly_dark')
        filtered_df = filtered_df.copy()
        filtered_df['combo_str'] = filtered_df['design'].astype(str) if 'design' in filtered_df.columns else ''

        if '_row_idx' not in filtered_df.columns:
            filtered_df['_row_idx'] = -1 if y_agg_label else filtered_df.index

        cd_cols = ['_row_idx'] + (['_agg_count'] if '_agg_count' in filtered_df.columns else [])
        for f in self.factors_subset:
            if f not in filtered_df.columns:
                filtered_df[f] = None

        plot_x_col, plot_y_col = x_col, y_col
        if x_jitter:
            jit_col = x_col + '_jit'
            filtered_df[jit_col] = filtered_df[x_col] + np.random.uniform(x_jitter[0], x_jitter[1], size=len(filtered_df))
            plot_x_col = jit_col
        if y_jitter:
            jit_col = y_col + '_jit'
            filtered_df[jit_col] = filtered_df[y_col] + np.random.uniform(y_jitter[0], y_jitter[1], size=len(filtered_df))
            plot_y_col = jit_col

        hover_base = f'<b>{x_col}:</b> %{{x}}<br><b>{y_col}:</b> %{{y}}<extra>%{{fullData.name}}</extra>'

        def _selection_color_groups(fdf, filter_dict):
            active = {k: set(v) for k, v in (filter_dict or {}).items() if v and k in fdf.columns}
            if skip_sparse and active:
                varying = _drop_constant_cols(fdf, list(active))
                if varying:  # only narrow down when some factors actually vary
                    active = {k: active[k] for k in varying}
            if not active:
                return None
            is_other = np.zeros(len(fdf), dtype=bool)
            label_parts = []
            for factor, selected in active.items():
                col_str = fdf[factor].astype(str)
                is_other |= ~col_str.isin(selected)
                label_parts.append((factor + ':') + col_str if show_factor_name else col_str)
            group_labels = (label_parts[0] if len(label_parts) == 1
                            else label_parts[0].str.cat(label_parts[1:], sep=', '))
            fdf['color_group'] = group_labels
            fdf.loc[is_other, 'color_group'] = 'Other'
            return 'color_group'

        _overlay_group_col = None
        _overlay_color_map = None

        if color_cfg and color_cfg['type'] == 'continuous':
            col = color_cfg['column']
            fig = px.scatter(
                filtered_df, x=plot_x_col, y=plot_y_col, color=col,
                color_continuous_scale=color_cfg.get('scale', 'Viridis'),
                custom_data=cd_cols,
                render_mode='webgl' if self.use_webgl else 'svg',
            )
            fig.update_traces(hovertemplate=hover_base)
            # no discrete groups — overlay not supported for continuous coloring
        elif color_cfg and color_cfg['type'] == 'precomputed':
            col = color_cfg['column']
            if col in filtered_df.columns:
                unique_groups = _sorted_groups(filtered_df[col])
                fig = self._make_discrete_figure(filtered_df, plot_x_col, plot_y_col, col,
                                                 color_cfg.get('label', 'Group'), hover_base, color_offsets, cd_cols, show_label=show_factor_name)
                _overlay_group_col = col
                _overlay_color_map = self._make_color_map(unique_groups, color_offsets)
            else:
                fig = self._make_uniform_figure(filtered_df, plot_x_col, plot_y_col, hover_base, cd_cols)
        elif color_cfg and color_cfg['type'] == 'columns':
            cols = [c for c in color_cfg['columns'] if c in filtered_df.columns]
            if cols:
                varying_cols = (_drop_constant_cols(filtered_df, cols) if skip_sparse else cols) or cols
                _str_cols = [filtered_df[c].astype(str) for c in varying_cols]
                filtered_df['color_group'] = (
                    _str_cols[0].str.cat(_str_cols[1:], sep=' | ')
                    if len(_str_cols) > 1 else _str_cols[0]
                )
                unique_groups = _sorted_groups(filtered_df['color_group'])
                fig = self._make_discrete_figure(filtered_df, plot_x_col, plot_y_col,
                                                 'color_group', ' | '.join(cols), hover_base, color_offsets, cd_cols, show_label=show_factor_name)
                _overlay_group_col = 'color_group'
                _overlay_color_map = self._make_color_map(unique_groups, color_offsets)
            else:
                fig = self._make_uniform_figure(filtered_df, plot_x_col, plot_y_col, hover_base, cd_cols)
        elif color_cfg and color_cfg['type'] == 'selection':
            col_name = _selection_color_groups(filtered_df, color_cfg['filter_dict'])
            if col_name:
                unique_groups = _sorted_groups(filtered_df[col_name])
                fig = self._make_discrete_figure(filtered_df, plot_x_col, plot_y_col,
                                                 col_name, 'Selection', hover_base, color_offsets, cd_cols, show_label=show_factor_name)
                _overlay_group_col = col_name
                _overlay_color_map = self._make_color_map(unique_groups, color_offsets)
            else:
                fig = self._make_uniform_figure(filtered_df, plot_x_col, plot_y_col, hover_base, cd_cols)
        elif color_cfg and color_cfg['type'] == 'discrete':
            col = color_cfg['column']
            filtered_df['color_group'] = filtered_df[col].astype(str).where(
                filtered_df[col].astype(str).isin(color_cfg['values']), other='Other'
            )
            unique_groups = _sorted_groups(filtered_df['color_group'])
            fig = self._make_discrete_figure(filtered_df, plot_x_col, plot_y_col,
                                             'color_group', col, hover_base, color_offsets, cd_cols, show_label=show_factor_name)
            _overlay_group_col = 'color_group'
            _overlay_color_map = self._make_color_map(unique_groups, color_offsets)
        else:
            fig = self._make_uniform_figure(filtered_df, plot_x_col, plot_y_col, hover_base, cd_cols)
            _overlay_color_map = {'__all__': 'steelblue'}

        if _overlay_info_out is not None:
            _overlay_info_out.append((plot_x_col, plot_y_col, filtered_df,
                                      _overlay_group_col, _overlay_color_map))

        hide_dots = not overlay_dots

        if not hide_dots:
            fig.update_traces(marker=dict(size=marker_size, opacity=marker_opacity))

        if overlay_line and _overlay_color_map and _overlay_info_out is None:
            _add_line_overlay(fig, filtered_df, plot_x_col, plot_y_col,
                              _overlay_group_col, _overlay_color_map, overlay_ks,
                              use_webgl=self.use_webgl,
                              anchor_left=anchor_left, anchor_right=anchor_right,
                              ks_offset=0, line_width=line_width, line_opacity=line_opacity)

        if hide_dots:
            for trace in fig.data:
                if (getattr(trace, 'mode', None) or 'markers') != 'lines':
                    trace.visible = False
                    trace.showlegend = False

        y_label = f'{y_agg_label}({y_col})' if y_agg_label and y_agg_label != 'raw' else y_col
        _x_display = x_label or x_col
        fig.update_layout(
            template=template,
            title=f'{y_label} vs {_x_display}', title_x=0.5,
            xaxis_title=_x_display + (' (jittered)' if x_jitter else ''),
            yaxis_title=y_label  + (' (jittered)' if y_jitter else ''),
            uirevision='scatter-plot', clickmode='event+select',
        )

        if x_jitter and x_col in filtered_df.columns:
            try:
                col = filtered_df[x_col].dropna()
                if pd.api.types.is_integer_dtype(col):
                    fig.update_xaxes(**_integer_tick_kwargs(int(col.min()), int(col.max())))
            except (ValueError, TypeError):
                pass
        if y_jitter and y_col in filtered_df.columns:
            try:
                col = filtered_df[y_col].dropna()
                if pd.api.types.is_integer_dtype(col):
                    fig.update_yaxes(**_integer_tick_kwargs(int(col.min()), int(col.max())))
            except (ValueError, TypeError):
                pass
        if not y_jitter and y_col in filtered_df.columns and y_agg_label != 'std':
            y_min, y_max = filtered_df[y_col].min(), filtered_df[y_col].max()
            if y_min >= 0 and y_max <= 1:
                fig.update_yaxes(range=[0, 1])

        _has_line_traces = any((getattr(t, 'mode', None) or '') == 'lines' for t in fig.data)
        if not _has_line_traces and len({t.name for t in fig.data if t.name}) <= 1:
            fig.update_layout(showlegend=False)

        return fig

    # ── Multi-Y overlay figure builder ────────────────────────────────────────

    def _build_overlay_figure(self, filtered_df, x_col, y_cols, *,
                              filter_expr='', x_bin_width=1,
                              agg_group_by=None, agg_y_mode='raw',
                              color_factor=None, factor_filters=None,
                              y_styles=None,
                              **figure_kwargs):
        """Build one figure with traces from each Y column overlaid on a shared axis.

        Pass 1 (Y-major): scatter traces, each with a unique legendgroup so every
        (group, Y) entry is independently toggleable in the legend.
        Pass 2 (group-major): line overlay traces so their legend order and k/s panel
        indices mirror the scatter ordering — index 0 = first group × first Y, etc.
        """
        fig = go.Figure()
        single_fig = None
        no_color_factor = figure_kwargs.get('color_cfg') is None
        y_styles_list   = y_styles or []
        overlay_line    = figure_kwargs.pop('overlay_line', False)
        overlay_ks      = list(figure_kwargs.pop('overlay_ks', None) or [])
        line_width      = figure_kwargs.pop('line_width', 2)
        line_opacity    = figure_kwargs.pop('line_opacity', 1.0)
        anchor_left     = figure_kwargs.get('anchor_left', False)
        anchor_right    = figure_kwargs.get('anchor_right', False)
        per_y_infos     = []  # per-Y: None or (px_col, py_col, sub_df, group_col, color_map)
        _co             = figure_kwargs.get('color_offsets') or {}

        # Pass 1: collect scatter traces per-Y (deferred — added group-major below)
        scatter_traces = {}  # (orig_trace_name, i_y) → (trace, style, y_color, y)
        for i_y, y in enumerate(y_cols):
            sub_df, x_plot, x_bin_lbl, y_agg_lbl, _ = self._apply_data_pipeline(
                filtered_df, x_col, y,
                filter_expr='',
                x_bin_width=x_bin_width,
                agg_group_by=agg_group_by, agg_y_mode=agg_y_mode,
                color_factor=color_factor, factor_filters=factor_filters,
            )
            if sub_df.empty:
                per_y_infos.append(None)
                continue
            tmp = []
            single_fig = self._build_figure(
                sub_df, x_plot, y,
                y_agg_label=y_agg_lbl,
                x_label=x_bin_lbl,
                overlay_line=False,
                _overlay_info_out=tmp,
                **figure_kwargs,
            )
            per_y_infos.append(tmp[0] if tmp else None)
            # GUI-selected style takes priority; auto-cycle is the fallback
            saved  = (y_styles_list[i_y] if i_y < len(y_styles_list) else None) or {}
            style  = {'symbol': saved.get('symbol') or _Y_STYLES[i_y % len(_Y_STYLES)]['symbol'],
                      'dash':   saved.get('dash')   or _Y_STYLES[i_y % len(_Y_STYLES)]['dash']}
            # Per-Y color: use shift-click stored offset if present; else fall back to index
            y_offset = _co.get(y, i_y) if no_color_factor else None
            y_color  = self._get_stable_color(y, y_offset) if no_color_factor else None
            for trace in single_fig.data:
                scatter_traces[(trace.name, i_y)] = (trace, style, y_color, y)

        if single_fig is None:
            fig.update_layout(title='No data for selected filters')
            return fig

        # Shared group ordering — drives both scatter (group-major) and line (Pass 2) addition
        # Discrete coloring: groups are color_map keys; uniform: single trace with orig_name=''
        all_groups = list(dict.fromkeys(
            g for info in per_y_infos if info is not None
            for g in (info[4] or {})
            if g != '__all__'
        ))
        scatter_orig_names = all_groups if all_groups else ['']

        # Add scatter traces group-major so scatter and line legends share the same ordering
        for orig_name in scatter_orig_names:
            for i_y, y in enumerate(y_cols):
                key = (orig_name, i_y)
                if key not in scatter_traces:
                    continue
                trace, style, y_color, y_val = scatter_traces[key]
                marker_upd = dict(symbol=style['symbol'])
                line_upd   = dict(dash=style['dash'])
                if y_color is not None:
                    marker_upd['color'] = y_color
                    line_upd['color']   = y_color
                trace.update(marker=marker_upd, line=line_upd)
                trace.name = f'{orig_name} [{y_val}]' if orig_name else y_val
                # Unique legendgroup per (group, Y) so each entry is independently toggleable
                trace.legendgroup = trace.name
                # Per-(group,Y) shift-click override (only for discrete color case)
                if y_color is None:
                    per_trace_off = _co.get(trace.name)
                    if per_trace_off is not None:
                        oc = self._get_stable_color(trace.name, per_trace_off)
                        trace.update(marker={'color': oc}, line={'color': oc})
                fig.add_trace(trace)

        # Pass 2: line overlay in group-major order (all_groups shared from above)
        if overlay_line:
            ks_offset = 0
            from scipy.interpolate import UnivariateSpline
            ScatterType = go.Scattergl if self.use_webgl else go.Scatter
            for group in all_groups:
                for i_y, y in enumerate(y_cols):
                    info = per_y_infos[i_y] if i_y < len(per_y_infos) else None
                    if info is None or group not in (info[4] or {}):
                        continue
                    px_col, py_col, sub_df, gcol, cmap = info
                    trace_name = f'{group} [{y}]'
                    # ::line key for explicit line override; fall back to scatter offset or group color
                    per_trace_off = _co.get(f'{trace_name}::line') or _co.get(trace_name)
                    color     = (self._get_stable_color(trace_name, per_trace_off)
                                 if per_trace_off is not None else cmap[group])
                    saved     = (y_styles_list[i_y] if i_y < len(y_styles_list) else None) or {}
                    line_dash = saved.get('dash') or _Y_STYLES[i_y % len(_Y_STYLES)]['dash']
                    ks = overlay_ks[ks_offset] if ks_offset < len(overlay_ks) else (overlay_ks[0] if overlay_ks else [1, None])
                    k  = max(1, min(5, int(ks[0]))) if ks[0] is not None else 1
                    s  = float(ks[1]) if ks[1] is not None else 0.0
                    sub = (sub_df[sub_df[gcol] == group] if gcol else sub_df)
                    sub = sub.dropna(subset=[px_col, py_col]).drop_duplicates(subset=[px_col]).sort_values(px_col)
                    if len(sub) >= k + 1:
                        x_vals = sub[px_col].values.astype(float)
                        y_vals = sub[py_col].values.astype(float)
                        try:
                            w = np.ones(len(x_vals))
                            if anchor_left: w[0] = 1e6
                            if anchor_right: w[-1] = 1e6
                            spl = UnivariateSpline(x_vals, y_vals, k=k, s=s, w=w)
                            x_line = np.linspace(x_vals.min(), x_vals.max(), 200)
                            y_line = spl(x_line)
                        except Exception:
                            x_line, y_line = x_vals, y_vals
                        fig.add_trace(ScatterType(
                            x=x_line, y=y_line,
                            mode='lines',
                            line=dict(color=color, width=line_width, dash=line_dash),
                            opacity=line_opacity,
                            name=trace_name,
                            legendgroup=f'{trace_name}::line',
                            showlegend=True,
                        ))
                        ks_offset += 1

        fig.update_layout(single_fig.layout)
        y_title = ' | '.join(y_cols)
        fig.update_layout(
            title=f'{y_title} vs {x_col}', title_x=0.5,
            yaxis_title=y_title,
            uirevision='scatter-plot', clickmode='event+select',
            showlegend=True,
        )
        _has_line_traces = any((getattr(t, 'mode', None) or '') == 'lines' for t in fig.data)
        if not _has_line_traces and len({t.name for t in fig.data if t.name}) <= 1:
            fig.update_layout(showlegend=False)
        return fig

    # ── Shared helpers (used by live render and PNG export) ──────────────────

    def _build_color_cfg(self, color_factor, filter_dict,
                         color_values, color_scale, color_scale_reverse, as_continuous):
        """Resolve color_factor + settings into a color_cfg dict for _build_figure."""
        _non_special = [f for f in (color_factor or [])
                        if f and f not in (SELECTION_VALUE, None,
                                           '_header_discrete', '_header_continuous', '_header_derived')]
        if SELECTION_VALUE in (color_factor or []):
            return {'type': 'selection', 'filter_dict': filter_dict or {}}
        if len(_non_special) == 1:
            f = _non_special[0]
            continuous_override = as_continuous and 'on' in (as_continuous or [])
            is_cont = f in self.continuous_factors or (f in self.numeric_discrete_factors and continuous_override)
            if is_cont:
                scale = (color_scale or 'Viridis') + ('_r' if color_scale_reverse and 'on' in color_scale_reverse else '')
                return {'type': 'continuous', 'column': f, 'scale': scale}
            if color_values:
                return {'type': 'discrete', 'column': f, 'values': color_values}
        if _non_special:
            return {'type': 'columns', 'columns': _non_special}
        return None

    # ── Shared data pipeline (used by live render and PNG export) ────────────

    def _apply_data_pipeline(self, filtered_df, x_col, y_col, *,
                              filter_expr='', x_bin_width=1,
                              agg_group_by=None, agg_y_mode='raw', color_factor=None,
                              factor_filters=None):
        """Canonical query → x-binning → aggregation pipeline.

        Returns (filtered_df, x_plot_col, x_bin_label, y_agg_label, filter_error).
        filter_error is None on success or a string describing the query failure.
        """
        _factor_filters = factor_filters or {}
        _color_factor   = color_factor or []
        _agg_group_by   = agg_group_by or []

        filter_error = None
        if filter_expr and filter_expr.strip():
            try:
                filtered_df = filtered_df.query(filter_expr)
            except Exception as e:
                filter_error = str(e)

        x_plot_col, x_bin_label = x_col, None
        if agg_y_mode and agg_y_mode != 'raw' and x_bin_width and float(x_bin_width) > 0:
            bw = float(x_bin_width)
            _bin_col = x_col + '_binned'
            filtered_df = filtered_df.copy()
            filtered_df[_bin_col] = (filtered_df[x_col] / bw).round() * bw
            x_plot_col  = _bin_col
            x_bin_label = f'{x_col} (bin={bw})'

        x_is_binned = x_plot_col != x_col
        y_agg_label = agg_y_mode if (agg_y_mode and agg_y_mode != 'raw' and x_is_binned) else None
        if y_agg_label:
            effective_group = list(_agg_group_by)
            extra_agg_cols, str_collapsed = [], []
            if SELECTION_VALUE in _color_factor:
                for factor, values in _factor_filters.items():
                    if values and factor not in effective_group and factor in filtered_df.columns:
                        effective_group.append(factor)
            for cf in _color_factor:
                if not cf or cf in (SELECTION_VALUE, None) or cf == y_col:
                    continue
                if cf in effective_group or cf not in filtered_df.columns:
                    continue
                if pd.api.types.is_numeric_dtype(self.df[cf]):
                    extra_agg_cols.append(cf)
                else:
                    str_collapsed.append(cf)
            n_before = len(filtered_df)
            filtered_df = aggregate_df(filtered_df, x_plot_col, y_col, effective_group, agg_y_mode,
                                       x_is_binned=x_is_binned, extra_agg_cols=extra_agg_cols)
            for cf in str_collapsed:
                if cf in filtered_df.columns:
                    filtered_df[cf] = 'Other'
            # Groups so fine-grained that every row is its own group — no actual
            # aggregation happened. Fall back to raw so _row_idx stays valid.
            if len(filtered_df) == n_before:
                y_agg_label = None

        return filtered_df, x_plot_col, x_bin_label, y_agg_label, filter_error

    # ── Server-side figure reconstruction (PNG export) ────────────────────────

    def _reconstruct_figure(self, view_data, theme_override=None):
        """Rebuild a figure from a saved view dict without a browser."""
        theme_val = theme_override if theme_override is not None else (view_data.get('theme-toggle') or [])
        template  = 'plotly_white' if 'light' in theme_val else 'plotly_dark'
        x_col = view_data.get('x-axis')
        _y_raw = view_data.get('y-axis')
        y_cols = ([_y_raw] if isinstance(_y_raw, str) else list(_y_raw or []))
        _primary_y = y_cols[0] if y_cols else None
        if not x_col or not y_cols:
            fig = go.Figure()
            fig.update_layout(title='View has no axes configured', template=template)
            return fig

        _extras = view_data.get('_extras') or {}

        # 1. Static factor filters
        _factor_filters = view_data.get('_factor_filters') or _extras.get('factor_filters') or {}
        mask = np.ones(len(self.df), dtype=bool)
        for factor, values in _factor_filters.items():
            if values and factor in self.df.columns:
                mask &= self.df[factor].astype(str).isin(values)
        filtered_df = self.df[mask].copy()

        # 2. Computed fields
        fields_expr = view_data.get('fields-expr') or ''
        if fields_expr.strip():
            filtered_df, _, _ = parse_and_apply_fields(filtered_df, fields_expr)

        # 3. Derived factor filters (after fields so computed cols exist)
        _derived_filters = view_data.get('_derived_filter_filters') or _extras.get('derived_filter_filters') or {}
        for factor, values in _derived_filters.items():
            if values and factor in filtered_df.columns:
                filtered_df = filtered_df[filtered_df[factor].astype(str).isin(values)]

        # 4–6. Query → binning → aggregation (shared pipeline)
        color_factor = view_data.get('color-by-factor') or []
        _agg_y_mode  = view_data.get('agg-y-mode') or 'raw'
        _agg_group_by = view_data.get('agg-group-by')
        _pipeline_kwargs = dict(
            filter_expr=view_data.get('filter-expr') or '',
            x_bin_width=view_data.get('x-bin-width') or 1,
            agg_group_by=_agg_group_by,
            agg_y_mode=_agg_y_mode,
            color_factor=color_factor,
            factor_filters=_factor_filters,
        )
        if len(y_cols) == 1:
            filtered_df, x_plot_col, x_bin_label, y_agg_label, _ = self._apply_data_pipeline(
                filtered_df, x_col, _primary_y, **_pipeline_kwargs,
            )
        else:
            filtered_df, x_plot_col, x_bin_label, _, _ = self._apply_data_pipeline(
                filtered_df, x_col, _primary_y,
                **{**_pipeline_kwargs, 'agg_group_by': None, 'agg_y_mode': 'raw'},
            )
            y_agg_label = None

        # 7. Jitter
        x_jit_on = view_data.get('x-jitter-toggle') or []
        xmin, xmax = view_data.get('x-jitter-min'), view_data.get('x-jitter-max')
        x_jitter = (float(xmin), float(xmax)) if (x_jit_on and 'on' in x_jit_on and xmin is not None and xmax is not None) else None
        y_jit_on = view_data.get('y-jitter-toggle') or []
        ymin, ymax = view_data.get('y-jitter-min'), view_data.get('y-jitter-max')
        y_jitter = (float(ymin), float(ymax)) if (y_jit_on and 'on' in y_jit_on and ymin is not None and ymax is not None) else None

        # 8. Build figure via shared pipeline
        fig = self._build_scatter_from_pipeline(
            filtered_df, x_col, x_plot_col, y_cols,
            x_bin_label, y_agg_label, _agg_group_by, _agg_y_mode,
            color_factor, _factor_filters,
            color_offsets=_extras.get('color_offsets'),
            template=template,
            color_values=view_data.get('color-by-values') or [],
            color_scale=view_data.get('color-scale-picker') or 'Viridis',
            color_scale_reverse=view_data.get('color-scale-reverse') or [],
            as_continuous=view_data.get('color-as-continuous') or [],
            x_jitter=x_jitter, y_jitter=y_jitter,
            marker_size=view_data.get('setting-marker-size'),
            marker_opacity=view_data.get('setting-marker-opacity'),
            overlay_dots=view_data.get('overlay-dots-toggle', ['on']),
            overlay_line=view_data.get('overlay-line-toggle', []),
            overlay_ks=_get_overlay_ks(view_data),
            anchor_left=view_data.get('overlay-anchor-left') or [],
            anchor_right=view_data.get('overlay-anchor-right') or [],
            show_factor_name=view_data.get('legend-show-factor-name', ['on']),
            skip_sparse=view_data.get('skip-sparse-factors', ['on']),
            line_width=view_data.get('overlay-line-width'),
            line_opacity=view_data.get('overlay-line-opacity'),
            y_styles=view_data.get('overlay-y-styles') or [],
            x_bin_width=view_data.get('x-bin-width') or 1,
        )
        self._apply_trace_vis(fig, _extras.get('trace_visibility'))
        return fig

    def _build_scatter_from_pipeline(
        self, filtered_df, x_col, x_plot_col, y_cols,
        x_bin_label, y_agg_label, agg_group_by, agg_y_mode,
        color_factor, factor_filters, *,
        color_offsets, template,
        color_values, color_scale, color_scale_reverse, as_continuous,
        x_jitter, y_jitter,
        marker_size, marker_opacity,
        overlay_dots, overlay_line, overlay_ks,
        anchor_left, anchor_right,
        show_factor_name, skip_sparse,
        line_width, line_opacity,
        y_styles=None, x_bin_width=1,
    ):
        """Shared figure-building step used by both the render callback and export reconstruct."""
        color_cfg = self._build_color_cfg(
            color_factor, factor_filters,
            color_values=color_values,
            color_scale=color_scale,
            color_scale_reverse=color_scale_reverse,
            as_continuous=as_continuous,
        )
        _figure_kwargs = dict(
            color_cfg=color_cfg, color_offsets=color_offsets,
            x_jitter=x_jitter, y_jitter=y_jitter,
            marker_size=marker_size if marker_size is not None else 5,
            marker_opacity=marker_opacity if marker_opacity is not None else 0.7,
            overlay_dots=bool(overlay_dots and 'on' in overlay_dots),
            overlay_line=bool(overlay_line and 'on' in overlay_line),
            overlay_ks=overlay_ks,
            anchor_left=bool(anchor_left and 'on' in anchor_left),
            anchor_right=bool(anchor_right and 'on' in anchor_right),
            template=template,
            show_factor_name=bool(show_factor_name and 'on' in show_factor_name),
            skip_sparse=bool('on' in (skip_sparse or ['on'])),
            line_width=line_width if line_width is not None else 2,
            line_opacity=line_opacity if line_opacity is not None else 1.0,
        )
        _primary_y = y_cols[0] if y_cols else None
        if len(y_cols) == 1:
            return self._build_figure(
                filtered_df, x_plot_col, _primary_y,
                y_agg_label=y_agg_label, x_label=x_bin_label,
                **_figure_kwargs,
            )
        return self._build_overlay_figure(
            filtered_df, x_col, y_cols,
            filter_expr='',
            x_bin_width=x_bin_width,
            agg_group_by=agg_group_by, agg_y_mode=agg_y_mode,
            color_factor=color_factor, factor_filters=factor_filters,
            y_styles=y_styles or [],
            **_figure_kwargs,
        )

    # ── Callbacks ─────────────────────────────────────────────────────────────

    def _register_callbacks(self):
        # Capture shared state into locals so callbacks are plain closures
        df                    = self.df
        discrete_factors      = self.discrete_factors
        factors_subset        = self.factors_subset
        factor_values         = self.factor_values
        axis_options          = self.axis_options
        continuous_factors    = self.continuous_factors
        numeric_discrete_factors = self.numeric_discrete_factors
        has_params_json       = self.has_params_json
        initial_display_points = self.initial_display_points
        color_by_options      = self.color_by_options
        _safe_roots           = self._safe_roots
        app                   = self.app

        # ── Derived-factor helper closures ────────────────────────────────────
        def _get_factor_values_dict(derived):
            combined = dict(factor_values)
            for entry in (derived or []):
                combined[entry['col']] = entry['values']
            return combined

        def _get_all_discrete_factors(derived):
            return discrete_factors + [entry['col'] for entry in (derived or [])]

        def _build_color_by_options(derived):
            opts = list(color_by_options)
            derived_cols = [entry['col'] for entry in (derived or [])]
            if derived_cols:
                opts.append({'label': '\u2500\u2500 Derived \u2500\u2500', 'value': '_header_derived', 'disabled': True})
                for col in derived_cols:
                    opts.append({'label': f'  {col}', 'value': col})
            return opts

        def _build_agg_options(derived):
            return [{'label': f, 'value': f} for f in _get_all_discrete_factors(derived)]

        # ── Collapsible panel toggles ─────────────────────────────────────────
        _panel_ids = ['controls', 'custom', 'appearance', 'views', 'settings']
        _panel_labels = {
            'controls':   'Plot',
            'custom':     'Filters',
            'appearance': 'Appearance',
            'views':      'Views',
            'settings':   '\u2699 Settings',
        }

        def _register_panel_toggle(panel_id, label):
            @callback(
                Output(f'{panel_id}-body',   'style',    allow_duplicate=True),
                Output(f'{panel_id}-toggle', 'children', allow_duplicate=True),
                Input(f'{panel_id}-toggle', 'n_clicks'),
                State(f'{panel_id}-body', 'style'),
                prevent_initial_call=True,
            )
            def toggle(n_clicks, current_style):
                hidden = (current_style or {}).get('display') == 'none'
                if hidden:
                    return {'display': 'block'}, f'\u25bc {label}'
                return {'display': 'none'}, f'\u25b6 {label}'

        for pid, lbl in _panel_labels.items():
            _register_panel_toggle(pid, lbl)

        # Collapse all panels when ?expand=false is in the URL
        @callback(
            *[Output(f'{pid}-body',   'style',    allow_duplicate=True) for pid in _panel_ids],
            *[Output(f'{pid}-toggle', 'children', allow_duplicate=True) for pid in _panel_ids],
            Input('url', 'search'),
            prevent_initial_call='initial_duplicate',
        )
        def _init_panel_expand(search):
            raw = (search or '').lstrip('?').lower()
            params = {k: v for part in raw.split('&') if '=' in part
                      for k, v in [part.split('=', 1)]}
            if params.get('expand') == 'false':
                closed = {'display': 'none'}
                return (
                    *[closed for _ in _panel_ids],
                    *[f'\u25b6 {_panel_labels[pid]}' for pid in _panel_ids],
                )
            raise PreventUpdate

        # ── View save/load ────────────────────────────────────────────────────
        import json

        _view_fields = [
            ('x-axis',                  'value'),
            ('y-axis',                  'value'),
            ('x-jitter-toggle',         'value'),
            ('x-jitter-min',            'value'),
            ('x-jitter-max',            'value'),
            ('x-bin-width',             'value'),
            ('y-jitter-toggle',         'value'),
            ('y-jitter-min',            'value'),
            ('y-jitter-max',            'value'),
            ('color-by-factor',         'value'),
            ('color-by-values',         'value'),
            ('color-scale-picker',      'value'),
            ('color-scale-reverse',     'value'),
            ('agg-group-by',            'value'),
            ('agg-y-mode',              'value'),
            ('filter-expr',             'value'),
            ('fields-expr',             'value'),
            ('yaml-export-dir',         'value'),
            ('setting-display-points',  'value'),
            ('setting-discrete-threshold', 'value'),
            ('setting-marker-size',     'value'),
            ('setting-marker-opacity',  'value'),
            ('color-as-continuous',     'value'),
            ('legend-show-factor-name', 'value'),
            ('skip-sparse-factors',    'value'),
            ('overlay-dots-toggle',  'value'),
            ('overlay-line-toggle',  'value'),
            ('overlay-line-width',   'value'),
            ('overlay-line-opacity', 'value'),
            ('overlay-spline-k',          'value'),
            ('overlay-spline-s',          'value'),
            ('overlay-per-line-toggle',   'value'),
            ('overlay-spline-ks',         'data'),
            ('overlay-anchor-left',       'value'),
            ('overlay-anchor-right',      'value'),
            ('overlay-y-styles',          'data'),
            ('theme-toggle',         'value'),
        ]

        @callback(
            Output('view-status',       'children', allow_duplicate=True),
            Output('view-picker',       'options',  allow_duplicate=True),
            Output('active-view-name',  'data',     allow_duplicate=True),
            Output('views-dir',         'value',    allow_duplicate=True),
            Input('view-save-btn', 'n_clicks'),
            State('view-name',  'value'),
            State('views-dir',  'value'),
            *[State(fid, prop) for fid, prop in _view_fields],
            State({'type': 'dropdown', 'factor': ALL}, 'value'),
            State('color-offsets',          'data'),
            State('live-trace-visibility',  'data'),
            State('scatter-plot',           'figure'),
            prevent_initial_call=True,
        )
        def save_view(n_clicks, view_name, views_dir_str, *args):
            if not view_name or not view_name.strip():
                return '\u26a0 Enter a view name first.', no_update, no_update
            view_name = view_name.strip()
            vdir = _views_dir(views_dir_str)
            n_fields  = len(_view_fields)
            view_data = {fid: val for (fid, _), val in zip(_view_fields, args[:n_fields])}

            discrete_factors_set = set(discrete_factors)
            static_filters, derived_filters = {}, {}
            for item in ctx.states_list[2 + n_fields]:
                factor = item['id']['factor']
                val = item.get('value') or []
                (static_filters if factor in discrete_factors_set else derived_filters)[factor] = val
            color_offsets     = args[n_fields + 1] or {}
            # live-trace-visibility reflects actual browser state (legend clicks included),
            # unlike State('scatter-plot', 'figure') which only has the last server-sent figure.
            live_trace_vis    = args[n_fields + 2] or {}
            figure            = args[n_fields + 3] or {}
            traces            = figure.get('data') or []
            visible_names     = {t['name'] for t in traces if t.get('name')}
            # Use live visibility; fall back to figure data for any missing entries.
            trace_visibility  = {
                f"{t.get('name', '')}|{t.get('mode') or 'markers'}":
                    live_trace_vis.get(
                        f"{t.get('name', '')}|{t.get('mode') or 'markers'}",
                        t.get('visible', True)
                    )
                for t in traces if t.get('name')
            }
            layout = figure.get('layout') or {}
            xr = (layout.get('xaxis') or {}).get('range')
            yr = (layout.get('yaxis') or {}).get('range')
            view_data['_extras'] = {
                'factor_filters':         static_filters,
                'derived_filter_filters': derived_filters,
                'color_offsets':          {k: v for k, v in color_offsets.items()
                                           if k in visible_names
                                           or (k.endswith('::line') and k[:-6] in visible_names)},
                'viewport':               {'xaxis': xr, 'yaxis': yr} if (xr or yr) else None,
                'trace_visibility':       trace_visibility,
            }

            if not _is_safe_path(vdir, _safe_roots):
                return f'\u26a0 Directory must be under {[str(r) for r in _safe_roots]}', no_update, no_update, no_update
            save_view_json(view_name, view_data, vdir)
            return f'\u2713 Saved "{view_name}"', _view_dropdown_options(vdir), view_name, str(vdir)

        def _load_view_data(view_name, views_dir_str, trigger):
            """Shared helper: load view file and build callback outputs list."""
            view_data = load_view_json(view_name, _views_dir(views_dir_str))
            if view_data is None:
                fail = *([no_update] * (len(_view_fields) + len(discrete_factors))), \
                       f'\u26a0 View "{view_name}" not found.', no_update, no_update, no_update, no_update, no_update, no_update
                return fail
            extras = view_data.get('_extras', {})
            # backwards compat: old saves used top-level keys
            factor_filters = extras.get('factor_filters') or view_data.get('_factor_filters', {})
            outputs = [view_data.get(fid, no_update) for fid, _ in _view_fields]
            outputs += [factor_filters.get(f, no_update) for f in discrete_factors]
            outputs += [
                f'\u2713 Loaded "{view_name}"',
                view_name,
                (trigger or 0) + 1,
                extras.get('derived_filter_filters') or view_data.get('_derived_filter_filters', {}),
                extras.get('color_offsets') or view_data.get('_color_offsets', {}),
                extras.get('viewport') or view_data.get('_viewport', None),
                extras.get('trace_visibility') or None,
            ]
            return tuple(outputs)

        @callback(
            *[Output(fid, prop, allow_duplicate=True) for fid, prop in _view_fields],
            *[Output({'type': 'dropdown', 'factor': f}, 'value', allow_duplicate=True) for f in discrete_factors],
            Output('view-status',           'children',   allow_duplicate=True),
            Output('active-view-name',      'data',       allow_duplicate=True),
            Output('process-trigger',       'data',       allow_duplicate=True),
            Output('pending-derived-filters', 'data',     allow_duplicate=True),
            Output('color-offsets',           'data',       allow_duplicate=True),
            Output('saved-viewport',          'data',       allow_duplicate=True),
            Output('saved-trace-visibility',  'data',       allow_duplicate=True),
            Output('views-dir',               'value',      allow_duplicate=True),
            Input('view-picker', 'value'),
            State('views-dir',        'value'),
            State('process-trigger',  'data'),
            prevent_initial_call=True,
        )
        def load_view(view_name, views_dir_str, trigger):
            if not view_name:
                raise PreventUpdate
            vdir = _views_dir(views_dir_str) if views_dir_str and views_dir_str.strip() \
                else Path(self._initial_views_dir)
            actual_dir = str(vdir) if (vdir / f'{view_name}.json').exists() else no_update
            return (*_load_view_data(view_name, str(vdir), trigger), actual_dir)

        @callback(
            Output('view-status',      'children', allow_duplicate=True),
            Output('view-picker',      'options',  allow_duplicate=True),
            Output('view-picker',      'value',    allow_duplicate=True),
            Output('active-view-name', 'data',     allow_duplicate=True),
            Input('view-delete-btn', 'n_clicks'),
            State('view-picker', 'value'),
            State('views-dir',   'value'),
            prevent_initial_call=True,
        )
        def delete_view(n_clicks, view_name, views_dir_str):
            if not view_name:
                return '\u26a0 Select a view to delete.', no_update, no_update, no_update
            vdir = _views_dir(views_dir_str)
            delete_view_json(view_name, vdir)
            return f'\u2713 Deleted "{view_name}"', _view_dropdown_options(vdir), None, None

        @callback(
            *[Output(fid, prop, allow_duplicate=True) for fid, prop in _view_fields],
            *[Output({'type': 'dropdown', 'factor': f}, 'value', allow_duplicate=True) for f in discrete_factors],
            Output('view-status',           'children',   allow_duplicate=True),
            Output('active-view-name',      'data',       allow_duplicate=True),
            Output('process-trigger',       'data',       allow_duplicate=True),
            Output('pending-derived-filters', 'data',     allow_duplicate=True),
            Output('color-offsets',           'data',       allow_duplicate=True),
            Output('saved-viewport',          'data',       allow_duplicate=True),
            Output('saved-trace-visibility',  'data',       allow_duplicate=True),
            Output('views-dir',               'value',      allow_duplicate=True),
            Input('url', 'search'),
            State('views-dir',        'value'),
            State('process-trigger',  'data'),
            State('active-view-name', 'data'),
            prevent_initial_call='initial_duplicate',
        )
        def load_view_from_url(search, views_dir_str, trigger, active_name):
            raw = (search or '').lstrip('?')
            view_name = None
            for part in raw.split('&'):
                if '=' not in part and part.strip():
                    view_name = part.strip()
                    break
            if not view_name:
                raise PreventUpdate
            if view_name == active_name:
                raise PreventUpdate
            vdir = _views_dir(views_dir_str) if views_dir_str and views_dir_str.strip() \
                else Path(self._initial_views_dir)
            actual_dir = str(vdir) if (vdir / f'{view_name}.json').exists() else no_update
            return (*_load_view_data(view_name, str(vdir), trigger), actual_dir)

        # Update URL without reload when a view is activated
        app.clientside_callback(
            """
            function(view_name) {
                if (!view_name) return window.dash_clientside.no_update;
                window.history.replaceState(null, '', window.location.pathname + '?' + view_name);
                return window.dash_clientside.no_update;
            }
            """,
            Output('active-view-name', 'data', allow_duplicate=True),
            Input('active-view-name', 'data'),
            prevent_initial_call=True,
        )

        @callback(
            Output('view-name', 'value'),
            Input('active-view-name', 'data'),
        )
        def sync_view_name(name):
            return name or ''

        # ── Lazy mode toggle ──────────────────────────────────────────────────
        @callback(
            Output('lazy-mode-active',   'data'),
            Output('lazy-mode-toggle',   'children'),
            Output('lazy-mode-toggle',   'style'),
            Input('lazy-mode-toggle',    'n_clicks'),
            State('lazy-mode-active',    'data'),
            prevent_initial_call=True,
        )
        def toggle_lazy_mode(n_clicks, is_active):
            new_state = not is_active
            style = {
                'padding': '6px 12px', 'fontSize': '12px', 'cursor': 'pointer',
                'border': f'2px solid {_BORDER}', 'borderRadius': '4px', 'color': _TEXT,
                'background': '#1a3a5c' if new_state else _DARK3,
                'fontWeight': 'bold' if new_state else 'normal', 'marginRight': '8px',
            }
            return new_state, f'Lazy Mode ({"ON" if new_state else "OFF"})', style

        # ── Overlay spline controls visibility ───────────────────────────────
        @callback(
            Output('overlay-spline-controls',    'style'),
            Output('overlay-spline-per-line',    'style'),
            Output('overlay-per-line-container', 'style'),
            Output('overlay-anchor-controls',    'style'),
            Input('overlay-line-toggle',         'value'),
            Input('overlay-per-line-toggle',     'value'),
        )
        def toggle_spline_visibility(line_on, per_line_on):
            if not (line_on and 'on' in line_on):
                return ({'display': 'none'}, {'display': 'none'},
                        {'display': 'none'}, {'display': 'none'})
            per_line_active = per_line_on and 'on' in per_line_on
            controls_style  = ({'display': 'none'} if per_line_active
                               else {'display': 'flex', 'alignItems': 'center'})
            per_line_style  = ({'display': 'flex', 'flexWrap': 'wrap', 'gap': '4px'}
                               if per_line_active else {'display': 'none'})
            anchor_style    = {'display': 'flex', 'alignItems': 'center'}
            return controls_style, per_line_style, {'display': 'inline-block'}, anchor_style

        # ── Build per-line k/s inputs when per-line mode is active ────────────
        @callback(
            Output('overlay-spline-per-line', 'children'),
            Output('overlay-spline-ks',       'data', allow_duplicate=True),
            Input('overlay-line-toggle',      'value'),
            Input('overlay-per-line-toggle',  'value'),
            Input('scatter-plot',             'figure'),
            State('overlay-spline-ks',        'data'),
            prevent_initial_call=True,
        )
        def build_per_line_inputs(line_on, per_line_on, figure, ks_data):
            if not (line_on and 'on' in line_on and per_line_on and 'on' in per_line_on):
                return no_update, no_update
            if not figure or not figure.get('data'):
                return [], no_update
            line_traces = [t for t in figure['data'] if (t.get('mode') or '') == 'lines']
            n = max(1, len(line_traces))
            old = list(ks_data or [])
            new_ks = old[:n] + [[1, None]] * max(0, n - len(old))
            children = []
            for i, (k, s) in enumerate(new_ks):
                label = (line_traces[i].get('name', '') if i < len(line_traces) else '') or str(i)
                children.append(html.Div([
                    html.Span(label, style={'color': '#888', 'fontSize': '11px',
                                            'fontFamily': 'monospace', 'marginRight': '3px',
                                            'maxWidth': '120px', 'overflow': 'hidden',
                                            'textOverflow': 'ellipsis', 'whiteSpace': 'nowrap'}),
                    html.Span('k:', style={'fontSize': '11px', 'color': '#aaa',
                                          'marginRight': '2px'}),
                    dcc.Input(id={'type': 'spline-k', 'index': i}, type='number',
                              value=k, min=1, max=5, step=1,
                              style={'width': '34px', 'fontSize': '11px', 'padding': '2px 3px',
                                     'background': '#2a2a2a', 'color': '#ddd',
                                     'border': '1px solid #444', 'borderRadius': '3px'}),
                    html.Span('s:', style={'fontSize': '11px', 'color': '#aaa',
                                          'margin': '0 2px 0 5px'}),
                    dcc.Input(id={'type': 'spline-s', 'index': i}, type='number',
                              value=s, min=0, step='any', placeholder='auto',
                              style={'width': '46px', 'fontSize': '11px', 'padding': '2px 3px',
                                     'background': '#2a2a2a', 'color': '#ddd',
                                     'border': '1px solid #444', 'borderRadius': '3px'}),
                ], style={'display': 'flex', 'alignItems': 'center', 'marginRight': '6px'}))
            store_out = new_ks if new_ks != old else no_update
            return children, store_out

        # ── Collect per-line k/s inputs → store ──────────────────────────────
        @callback(
            Output('overlay-spline-ks', 'data', allow_duplicate=True),
            Input({'type': 'spline-k', 'index': ALL}, 'value'),
            Input({'type': 'spline-s', 'index': ALL}, 'value'),
            prevent_initial_call=True,
        )
        def collect_per_line_ks(k_vals, s_vals):
            if not k_vals:
                raise PreventUpdate
            return [[k if k is not None else 1, s] for k, s in zip(k_vals, s_vals)]

        # ── Per-Y style (symbol + dash) ───────────────────────────────────────
        _SYMBOL_OPTIONS = [{'label': s, 'value': s}
                           for s in ['circle', 'square', 'diamond', 'cross', 'x',
                                     'triangle-up', 'triangle-down', 'star']]
        _DASH_OPTIONS   = [{'label': d, 'value': d}
                           for d in ['solid', 'dash', 'dot', 'dashdot', 'longdash']]
        _dd_style = {'width': '100px', 'fontSize': '11px',
                     'background': '#2a2a2a', 'color': '#ddd'}

        @callback(
            Output('overlay-y-styles-container', 'style'),
            Input('y-axis', 'value'),
        )
        def toggle_y_styles_visibility(y_val):
            y_cols = [y_val] if isinstance(y_val, str) else list(y_val or [])
            if len(y_cols) > 1:
                return {'display': 'flex', 'flexDirection': 'column', 'gap': '6px',
                        'marginBottom': '12px'}
            return {'display': 'none'}

        @callback(
            Output('overlay-y-styles-container', 'children'),
            Output('overlay-y-styles', 'data', allow_duplicate=True),
            Input('y-axis', 'value'),
            State('overlay-y-styles', 'data'),
            prevent_initial_call=True,
        )
        def build_y_style_inputs(y_val, styles_data):
            y_cols = [y_val] if isinstance(y_val, str) else list(y_val or [])
            if len(y_cols) <= 1:
                return no_update, no_update
            old = list(styles_data or [])
            new_styles = (old + [None] * len(y_cols))[:len(y_cols)]
            children = []
            for i, y in enumerate(y_cols):
                saved   = new_styles[i] or {}
                sym_val  = saved.get('symbol', _Y_STYLES[i % len(_Y_STYLES)]['symbol'])
                dash_val = saved.get('dash',   _Y_STYLES[i % len(_Y_STYLES)]['dash'])
                children.append(html.Div([
                    html.Span(y, style={'color': '#888', 'fontSize': '11px',
                                        'fontFamily': 'monospace', 'marginRight': '4px',
                                        'maxWidth': '80px', 'overflow': 'hidden',
                                        'textOverflow': 'ellipsis', 'whiteSpace': 'nowrap'}),
                    dcc.Dropdown(id={'type': 'y-symbol', 'index': i},
                                 options=_SYMBOL_OPTIONS, value=sym_val,
                                 clearable=False, style=_dd_style),
                    dcc.Dropdown(id={'type': 'y-dash', 'index': i},
                                 options=_DASH_OPTIONS, value=dash_val,
                                 clearable=False, style={**_dd_style, 'marginLeft': '4px'}),
                ], style={'display': 'flex', 'alignItems': 'center'}))
            store_out = new_styles if new_styles != old else no_update
            return children, store_out

        @callback(
            Output('overlay-y-styles', 'data', allow_duplicate=True),
            Input({'type': 'y-symbol', 'index': ALL}, 'value'),
            Input({'type': 'y-dash',   'index': ALL}, 'value'),
            prevent_initial_call=True,
        )
        def collect_y_styles(symbols, dashes):
            if not symbols:
                raise PreventUpdate
            return [{'symbol': sym or 'circle', 'dash': dash or 'solid'}
                    for sym, dash in zip(symbols, dashes)]

        # ── Factor dropdown select-all / clear-all ────────────────────────────
        @callback(
            Output({'type': 'dropdown', 'factor': MATCH}, 'value'),
            Input({'type': 'select-all', 'factor': MATCH}, 'n_clicks'),
            Input({'type': 'clear-all',  'factor': MATCH}, 'n_clicks'),
            State({'type': 'dropdown',   'factor': MATCH}, 'id'),
            State('derived-factors', 'data'),
            prevent_initial_call=True,
        )
        def update_dropdown_selection(select_clicks, clear_clicks, dropdown_id, derived):
            triggered_id = ctx.triggered_id
            if triggered_id is None:
                raise PreventUpdate
            factor = triggered_id['factor']
            if triggered_id['type'] == 'select-all':
                return _get_factor_values_dict(derived).get(factor, [])
            return []

        @callback(
            Output('agg-group-by', 'value'),
            Input('agg-group-select-all', 'n_clicks'),
            Input('agg-group-clear-all',  'n_clicks'),
            State('derived-factors', 'data'),
            prevent_initial_call=True,
        )
        def update_agg_group_selection(select_clicks, clear_clicks, derived):
            if ctx.triggered_id == 'agg-group-select-all':
                return _get_all_discrete_factors(derived)
            return []

        # ── Bin width visibility — only shown when aggregation is active ─────
        @callback(
            Output('x-bin-width-container', 'style'),
            Input('agg-y-mode', 'value'),
        )
        def toggle_bin_width_visibility(agg_mode):
            if agg_mode and agg_mode != 'raw':
                return {'width': '10%', 'display': 'inline-block', 'verticalAlign': 'top'}
            return {'display': 'none'}

        # ── Color control visibility ──────────────────────────────────────────
        @callback(
            Output('color-by-values',              'options'),
            Output('color-by-values',              'value'),
            Output('color-values-container',       'style'),
            Output('color-scale-container',        'style'),
            Output('color-continuous-toggle-container', 'style'),
            Input('color-by-factor',  'value'),
            Input('color-as-continuous', 'value'),
            State('derived-factors',  'data'),
        )
        def update_color_controls(color_factor, as_continuous, derived):
            show_vals  = {'width': '18%', 'display': 'inline-block', 'marginRight': '2%'}
            hide_vals  = {'width': '18%', 'display': 'none',         'marginRight': '2%'}
            show_scale = {'width': '18%', 'display': 'inline-block', 'marginRight': '2%'}
            hide_scale = {'width': '18%', 'display': 'none',         'marginRight': '2%'}
            show_toggle = {'display': 'inline-block', 'width': '9%', 'marginRight': '1%', 'verticalAlign': 'bottom'}
            hide_toggle = {'display': 'none',         'width': '9%', 'marginRight': '1%', 'verticalAlign': 'bottom'}

            continuous_override = as_continuous and 'on' in as_continuous
            fv = _get_factor_values_dict(derived)
            derived_numeric = {entry['col'] for entry in (derived or []) if entry.get('is_numeric')}

            factors = [f for f in (color_factor or []) if f and f != SELECTION_VALUE]
            if len(factors) == 1:
                f = factors[0]
                if f in continuous_factors:
                    return [], [], hide_vals, show_scale, hide_toggle
                is_numeric = f in numeric_discrete_factors or f in derived_numeric
                vals = fv.get(f, [])
                opts = [{'label': v, 'value': v} for v in vals]
                if is_numeric and continuous_override:
                    return opts, vals, hide_vals, show_scale, show_toggle
                return opts, vals, show_vals, hide_scale, (show_toggle if is_numeric else hide_toggle)
            return [], [], hide_vals, hide_scale, hide_toggle

        # ── Legend shift-click color offset ──────────────────────────────────
        app.clientside_callback(
            """
            function(figure) {
                if (!window._shiftKeyState) {
                    window._shiftKeyState = {pressed: false, alt: false, pendingDigits: '', pendingActions: []};
                    document.addEventListener('keydown', function(e) {
                        if (e.key === 'Shift') window._shiftKeyState.pressed = true;
                        else if (e.key === 'Alt') window._shiftKeyState.alt = true;
                        else if (window._shiftKeyState.pressed && /^Digit[0-9]$/.test(e.code))
                            window._shiftKeyState.pendingDigits += e.code.slice(-1);
                        else if (window._shiftKeyState.pressed && e.code === 'Comma')
                            window._shiftKeyState.pendingDigits += ',';
                    });
                    document.addEventListener('keyup', function(e) {
                        if (e.key === 'Shift') {
                            window._shiftKeyState.pressed = false;
                            window._shiftKeyState.pendingDigits = '';
                            var actions = window._shiftKeyState.pendingActions;
                            window._shiftKeyState.pendingActions = [];
                            if (actions.length > 0)
                                dash_clientside.set_props('legend-shift-click', {data: JSON.stringify(actions)});
                        } else if (e.key === 'Alt') {
                            window._shiftKeyState.alt = false;
                        }
                    });
                }
                setTimeout(function() {
                    var gd = document.querySelector('#scatter-plot .js-plotly-plot');
                    if (!gd) return;
                    gd.removeAllListeners('plotly_legendclick');
                    gd.on('plotly_legendclick', function(eventData) {
                        if (!window._shiftKeyState.pressed) return;
                        var traceData = eventData.data[eventData.curveNumber];
                        var traceName = traceData.name;
                        var traceMode = traceData.mode || '';
                        var action;
                        if (window._shiftKeyState.pendingDigits) {
                            var pd = window._shiftKeyState.pendingDigits;
                            window._shiftKeyState.pendingDigits = '';
                            if (pd.indexOf(',') >= 0) {
                                var parts = pd.split(',').map(function(p) {
                                    return p === '' ? null : parseFloat(p);
                                });
                                action = {t: traceName, m: traceMode, action: 'hsv', value: parts};
                            } else {
                                action = {t: traceName, m: traceMode, action: 'hsv_h', value: parseInt(pd, 10)};
                            }
                        } else {
                            action = {t: traceName, m: traceMode, action: window._shiftKeyState.alt ? 'dec' : 'inc'};
                        }
                        window._shiftKeyState.pendingActions.push(action);
                        return false;
                    });
                }, 300);
                return window.dash_clientside.no_update;
            }
            """,
            Output('scatter-plot', 'id'),
            Input('scatter-plot', 'figure'),
        )

        @callback(
            Output('color-offsets', 'data'),
            Input('legend-shift-click', 'data'),
            State('color-offsets', 'data'),
            prevent_initial_call=True,
        )
        def bump_color_offset(shift_click_data, offsets):
            if not shift_click_data:
                raise PreventUpdate
            n = len(_PALETTE_DARK if self._dark_mode else _PALETTE_LIGHT)
            for item in orjson.loads(shift_click_data):
                # Line traces store under 'name::line' to allow independent colors from scatter.
                # Scatter stores under plain 'name'. Pass 2 checks ::line first, then falls back.
                trace_name = item['t']
                if item.get('m') == 'lines':
                    trace_name = trace_name + '::line'
                action  = item['action']
                current = offsets.get(trace_name, 0)
                if action == 'set':
                    offsets[trace_name] = {'d': int(item['value']) % n}
                elif action == 'hsv_h':
                    offsets[trace_name] = {'h': int(item['value']) % 10}
                elif action == 'hsv':
                    parts = item['value']
                    h = float(parts[0]) if (len(parts) > 0 and parts[0] is not None) else 0.0
                    s = float(parts[1]) if (len(parts) > 1 and parts[1] is not None) else 100.0
                    v = float(parts[2]) if (len(parts) > 2 and parts[2] is not None) else 100.0
                    offsets[trace_name] = {'hsv': [h, s, v]}
                elif isinstance(current, dict):
                    if 'h' in current:
                        offsets[trace_name] = {'h': (current['h'] + (1 if action == 'inc' else -1)) % 10}
                    elif 'hsv' in current:
                        h, s, v = current['hsv']
                        offsets[trace_name] = {'hsv': [(h + (36 if action == 'inc' else -36)) % 360, s, v]}
                    else:
                        offsets[trace_name] = {'d': (current['d'] + (1 if action == 'inc' else -1)) % n}
                elif action == 'inc':
                    offsets[trace_name] = (current + 1) % n
                else:
                    offsets[trace_name] = (current - 1) % n
            return offsets

        # ── Sticky colors toggle — clear palette when turned off ──────────────
        @callback(
            Output('color-offsets', 'data', allow_duplicate=True),
            Input('setting-sticky-colors', 'value'),
            prevent_initial_call=True,
        )
        def handle_sticky_colors_toggle(sticky):
            if not sticky or 'on' not in sticky:
                return {}
            raise PreventUpdate

        # ── Point click details ───────────────────────────────────────────────
        @callback(
            Output('point-details',    'children'),
            Output('selected-row-idx', 'data'),
            Output('export-yaml-btn',  'disabled'),
            Input('scatter-plot', 'clickData'),
            State('x-axis',      'value'),
            State('y-axis',      'value'),
            State('agg-group-by', 'value'),
            State('agg-y-mode',   'value'),
            State('trace-row-map', 'data'),
            prevent_initial_call=True,
        )
        def show_point_details(click_data, x_col, y_col, agg_group_by, agg_y_mode, trace_row_map):
            if not click_data or not click_data['points']:
                raise PreventUpdate
            pt = click_data['points'][0]
            curve_num  = pt.get('curveNumber', 0)
            point_num  = pt.get('pointNumber', 0)
            if trace_row_map and curve_num < len(trace_row_map):
                point_cd = trace_row_map[curve_num]
                customdata = point_cd[point_num] if point_num < len(point_cd) else []
            else:
                customdata = pt.get('customdata') or []
            row_idx_val = int(customdata[0]) if customdata else -1
            agg_count = int(customdata[1]) if len(customdata) > 1 else None
            y_cols = [y_col] if isinstance(y_col, str) else list(y_col or [])
            is_multi_y = len(y_cols) > 1
            lines = [f'{x_col}: {pt.get("x")}']
            shown_as_y: set = set()
            if is_multi_y and row_idx_val >= 0:
                try:
                    row = df.loc[row_idx_val]
                    for yc in y_cols:
                        if yc in df.columns:
                            v = row[yc]
                            if pd.notna(v):
                                lines.append(f'{yc}: {v}')
                                shown_as_y.add(yc)
                        else:
                            lines.append(f'{yc}: (computed)')
                except KeyError:
                    lines.append(f'y: {pt.get("y")}')
            elif is_multi_y:
                lines.append(f'y: {pt.get("y")}')
            else:
                y_display = y_cols[0] if y_cols else 'Y'
                y_label = f'{agg_y_mode}({y_display})' if agg_y_mode and agg_y_mode != 'raw' and agg_group_by else y_display
                lines.append(f'{y_label}: {pt.get("y")}')
            if row_idx_val >= 0:
                try:
                    row = df.loc[row_idx_val]
                    lines += [f'{k}: {v}' for k, v in row.items()
                              if k in factors_subset and k not in shown_as_y and pd.notna(v)]
                except KeyError:
                    pass
            else:
                count_str = f' × {agg_count}' if agg_count is not None else ''
                lines.append(f'(aggregated point{count_str})')
            selected = row_idx_val if row_idx_val >= 0 else None
            btn_disabled = not has_params_json
            return '\n'.join(lines), selected, btn_disabled

        # ── YAML export ───────────────────────────────────────────────────────
        @callback(
            Output('yaml-download',  'data'),
            Output('export-status',  'children'),
            Input('export-yaml-btn', 'n_clicks'),
            State('selected-row-idx',  'data'),
            State('yaml-export-dir',   'value'),
            prevent_initial_call=True,
        )
        def export_yaml(n_clicks, row_idx, export_dir_str):
            if not n_clicks:
                raise PreventUpdate
            if row_idx is None:
                return no_update, html.Span('⚠ Not available for aggregated points',
                                            style={'color': '#f66'})
            params_data = orjson.loads(df.at[row_idx, 'params_json'])
            if 'logging' in params_data and params_data['logging'] and 'out_path' in params_data['logging']:
                params_data['logging']['out_path'] = str(params_data['logging']['out_path']).replace('grid_search', 'single_run')
            yaml_str = yaml.dump(params_data, default_flow_style=False, sort_keys=False)
            if export_dir_str and export_dir_str.strip():
                p = Path(export_dir_str.strip())
                out_path = p if p.suffix else p / 'params.yaml'
                if not _is_safe_path(out_path.parent, _safe_roots):
                    return no_update, f'\u26a0 Directory must be under {[str(r) for r in _safe_roots]}'
                out_path.parent.mkdir(parents=True, exist_ok=True)
                out_path.write_text(yaml_str)
                return no_update, f'Saved to {out_path}'
            return dcc.send_string(yaml_str, filename='params.yaml'), ''

        # ── Derived filter dropdowns ──────────────────────────────────────────
        @callback(
            Output('derived-filter-dropdowns', 'children'),
            Output('pending-derived-filters',  'data'),
            Input('derived-factors', 'data'),
            State('pending-derived-filters', 'data'),
            prevent_initial_call=True,
        )
        def render_derived_filters(derived_factors_data, pending):
            children = []
            for entry in (derived_factors_data or []):
                col, values = entry['col'], entry['values']
                selected = (pending or {}).get(col, [])
                children.append(html.Div([
                    html.Label(col),
                    html.Div([
                        html.Button('Select All', id={'type': 'select-all', 'factor': col},
                                    n_clicks=0, style={**btn_sm, 'marginRight': '4px'}),
                        html.Button('Clear All', id={'type': 'clear-all', 'factor': col},
                                    n_clicks=0, style=btn_sm),
                    ], style={'marginBottom': '4px'}),
                    dcc.Dropdown(id={'type': 'dropdown', 'factor': col},
                                 options=[{'label': v, 'value': v} for v in values],
                                 value=selected, multi=True),
                ], style={'width': '18%', 'display': 'inline-block', 'marginRight': '2%', 'verticalAlign': 'top'}))
            return children, {}

        # ── Render: data pipeline + figure (fires on any data-changing input) ──
        @callback(
            Output('scatter-plot',           'figure'),
            Output('saved-viewport',         'data', allow_duplicate=True),
            Output('saved-trace-visibility', 'data', allow_duplicate=True),
            Output('filter-error',    'children'),
            Output('filter-count',    'children'),
            Output('fields-error',    'children'),
            Output('fields-info',     'children'),
            Output('x-axis',          'options'),
            Output('y-axis',          'options'),
            Output('derived-factors', 'data'),
            Output('color-by-factor', 'options', allow_duplicate=True),
            Output('agg-group-by',    'options', allow_duplicate=True),
            Output('trace-row-map',   'data'),
            Input('refresh-btn',                'n_clicks'),
            Input({'type': 'dropdown', 'factor': ALL}, 'value'),
            Input('agg-group-by',               'value'),
            Input('agg-y-mode',                 'value'),
            Input('x-bin-width',                'value'),
            Input('setting-display-points',      'value'),
            Input('setting-stratified-sample',   'value'),
            Input('setting-discrete-threshold',  'value'),
            Input('x-axis',                     'value'),
            Input('y-axis',                     'value'),
            Input('color-by-factor',            'value'),
            Input('process-trigger',            'data'),
            Input('color-by-values',     'value'),
            Input('color-offsets',       'data'),
            Input('color-as-continuous', 'value'),
            Input('x-jitter-toggle',     'value'),
            Input('x-jitter-min',        'value'),
            Input('x-jitter-max',        'value'),
            Input('y-jitter-toggle',     'value'),
            Input('y-jitter-min',        'value'),
            Input('y-jitter-max',        'value'),
            Input('legend-show-factor-name', 'value'),
            Input('skip-sparse-factors',     'value'),
            Input('overlay-dots-toggle', 'value'),
            Input('overlay-line-toggle', 'value'),
            Input('overlay-spline-k',          'value'),
            Input('overlay-spline-s',          'value'),
            Input('overlay-per-line-toggle',   'value'),
            Input('overlay-spline-ks',         'data'),
            Input('overlay-anchor-left',       'value'),
            Input('overlay-anchor-right',      'value'),
            Input('overlay-y-styles',          'data'),
            Input('overlay-line-width',        'value'),
            Input('overlay-line-opacity',      'value'),
            State('lazy-mode-active',         'data'),
            State('filter-expr',              'value'),
            State('fields-expr',              'value'),
            State('scatter-plot',             'relayoutData'),
            State('derived-factors',          'data'),
            State('setting-marker-size',      'value'),
            State('setting-marker-opacity',   'value'),
            State('saved-viewport',           'data'),
            State('saved-trace-visibility',   'data'),
            State('theme-toggle',             'value'),
            State('color-scale-picker',       'value'),
            State('color-scale-reverse',      'value'),
            prevent_initial_call='initial_duplicate',
        )
        def render_figure(refresh_clicks, filter_values, agg_group_by, agg_y_mode,
                          x_bin_width, display_pts, stratified_sample, dynamic_discrete_threshold,
                          x_col, y_col, color_factor, process_trigger,
                          color_values, color_offsets, as_continuous,
                          x_jit_on, x_jit_min, x_jit_max,
                          y_jit_on, y_jit_min, y_jit_max,
                          legend_show_factor, skip_sparse,
                          overlay_dots_toggle, overlay_toggle, overlay_k, overlay_s,
                          overlay_per_line, overlay_ks,
                          anchor_left, anchor_right, overlay_y_styles,
                          line_width, line_opacity,
                          lazy_mode, filter_expr, fields_expr, relayout_data, current_derived,
                          marker_size, marker_opacity, saved_viewport, saved_trace_vis,
                          theme, color_scale, color_scale_reverse):
            _lazy_inputs = {
                'x-axis', 'y-axis',
                'agg-group-by', 'agg-y-mode',
                'x-bin-width',
                'setting-display-points',
            }
            if lazy_mode and ctx.triggered_id is not None:
                def _lazy(pid):
                    cid = pid.rpartition('.')[0]
                    return cid in _lazy_inputs or (cid.startswith('{') and '"dropdown"' in cid)
                if all(_lazy(p) for p in ctx.triggered_prop_ids):
                    raise PreventUpdate

            template = 'plotly_white' if (theme and 'light' in theme) else 'plotly_dark'

            # Normalise y_col: multi-select dropdown may return a string or list
            y_cols = ([y_col] if isinstance(y_col, str) else list(y_col or []))
            _primary_y = y_cols[0] if y_cols else None

            filter_error = ''
            fields_error = ''
            fields_info  = ''
            current_axis_opts = [{'label': c, 'value': c} for c in axis_options]
            _pipeline = []

            def _empty(title, x=None, y=None):
                fig = go.Figure()
                fig.update_layout(
                    title=title, xaxis_title=x or 'X', yaxis_title=y or 'Y',
                    template=template,
                )
                return (fig, no_update, no_update,
                        filter_error, '', fields_error, fields_info,
                        current_axis_opts, current_axis_opts,
                        no_update, no_update, no_update, no_update)

            def _apply_viewport(fig):
                """Apply saved viewport once, then clear the store."""
                if not saved_viewport:
                    return fig, no_update
                xr = saved_viewport.get('xaxis')
                yr = saved_viewport.get('yaxis')
                # New uirevision forces Plotly to reset zoom and apply explicit ranges.
                fig.update_layout(uirevision=_time.time())
                if xr:
                    fig.update_xaxes(range=xr)
                if yr:
                    fig.update_yaxes(range=yr)
                return fig, None  # None clears saved-viewport so it applies only once

            def _apply_trace_visibility(fig):
                """Apply saved per-trace visibility once, then clear the store."""
                if not saved_trace_vis:
                    return fig, no_update
                ScatterDashboard._apply_trace_vis(fig, saved_trace_vis)
                return fig, None  # None clears the store

            if not x_col or not y_cols:
                return _empty('Select X and Y axes to display data')

            # ── Pipeline cache: skip filter/agg/sample on pure cosmetic triggers ──
            _pipe_key = (
                x_col or '', tuple(y_cols),
                tuple(sorted(str(g) for g in (agg_group_by or []))),
                agg_y_mode or 'raw', str(x_bin_width or 1), str(display_pts),
                tuple(str(v) if v else '' for v in (filter_values or [])),
                filter_expr or '', fields_expr or '',
                tuple(str(f) for f in (color_factor or [])),
                str(process_trigger), str(dynamic_discrete_threshold),
                str(stratified_sample),
            )
            _has_vp = bool(relayout_data and any(
                relayout_data.get(k) is not None
                for k in ('xaxis.range[0]', 'yaxis.range[0]')
            ))
            _use_cache = (
                isinstance(ctx.triggered_id, str)
                and ctx.triggered_id in _COSMETIC_TRIGGERS
                and not _has_vp
                and self._pipeline_cache_key == _pipe_key
            )

            if _use_cache:
                (filtered_df, x_plot_col, x_bin_label, y_agg_label, _pipeline,
                 derived_discrete, current_axis_opts, filter_error, fields_error,
                 fields_info, _ff_dict) = self._pipeline_cache_val
                filtered_df = filtered_df.copy()
            else:
                # 1. Factor filters
                mask = np.ones(len(df), dtype=bool)
                for item in ctx.inputs_list[1]:
                    factor = item['id']['factor']
                    values = item.get('value') or []
                    if values and factor in df.columns:
                        mask &= df[factor].astype(str).isin(values)
                filtered_df = df[mask].copy()
                _pipeline.append((len(df), ''))
                if len(filtered_df) != len(df):
                    _pipeline.append((len(filtered_df), 'filter'))

                # 2. Computed fields
                derived_discrete = []
                if fields_expr and fields_expr.strip():
                    filtered_df, fields_info, fields_error = parse_and_apply_fields(filtered_df, fields_expr)
                    new_num_cols = [c for c in filtered_df.select_dtypes(include='number').columns
                                    if c not in axis_options]
                    if new_num_cols:
                        current_axis_opts = (
                            [{'label': c, 'value': c} for c in axis_options]
                            + [{'label': '── computed ──', 'value': '_h_computed', 'disabled': True}]
                            + [{'label': f'  {c}', 'value': c} for c in new_num_cols]
                        )

                # 2b. Dynamic discrete detection
                eff_threshold = dynamic_discrete_threshold if dynamic_discrete_threshold is not None else self.discrete_threshold
                static_discrete_set = set(discrete_factors)
                orig_cols = set(df.columns)
                candidates = {col: filtered_df[col] for col in filtered_df.columns if col not in orig_cols}
                candidates.update({col: df[col] for col in df.columns
                                   if col not in static_discrete_set and col not in candidates})
                for col, series in candidates.items():
                    n_uniq = series.nunique()
                    if not (1 < n_uniq < eff_threshold):
                        continue
                    is_numeric = pd.api.types.is_numeric_dtype(series)
                    vals = sorted(str(v) for v in series.dropna().unique())
                    derived_discrete.append({'col': col, 'values': vals, 'is_numeric': is_numeric})

                # 3. Validate axes
                missing = [c for c in ([x_col] + y_cols) if c not in filtered_df.columns]
                if missing:
                    if not fields_error:
                        fields_error = f'⚠ Column(s) not in data: {", ".join(missing)}. Define in Computed Fields.'
                    return _empty(f"Column(s) not found: {', '.join(missing)}", x_col, _primary_y)

                # 4–5. Query → binning → aggregation (shared pipeline)
                _ff_dict = {f: v for f, v in zip(discrete_factors, filter_values) if v}
                if len(y_cols) == 1:
                    # Single-Y: full pipeline including aggregation
                    filtered_df, x_plot_col, x_bin_label, y_agg_label, _query_err = self._apply_data_pipeline(
                        filtered_df, x_col, _primary_y,
                        filter_expr=filter_expr,
                        x_bin_width=x_bin_width,
                        agg_group_by=agg_group_by, agg_y_mode=agg_y_mode,
                        color_factor=color_factor, factor_filters=_ff_dict,
                    )
                else:
                    # Multi-Y: apply query filter inline; aggregation + binning run per-Y
                    # inside _build_overlay_figure, so avoid a redundant outer pipeline call.
                    _query_err = None
                    if filter_expr and filter_expr.strip():
                        try:
                            filtered_df = filtered_df.query(filter_expr)
                        except Exception as _e:
                            _query_err = str(_e)
                    x_plot_col  = x_col
                    x_bin_label = None
                    if agg_y_mode and agg_y_mode != 'raw' and x_bin_width and float(x_bin_width) > 0:
                        x_plot_col  = x_col + '_binned'
                        x_bin_label = f'{x_col} (bin={float(x_bin_width)})'
                    y_agg_label = agg_y_mode if x_bin_label else None
                if _query_err:
                    filter_error = f'⚠ {_query_err}'
                elif filter_expr and filter_expr.strip():
                    _pipeline.append((len(filtered_df), 'query'))
                if y_agg_label:
                    _pipeline.append((len(filtered_df), 'agg'))

                # 5.5. Viewport-aware sampling
                total_points = len(filtered_df)
                if relayout_data and x_plot_col and _primary_y:
                    x0 = relayout_data.get('xaxis.range[0]')
                    x1 = relayout_data.get('xaxis.range[1]')
                    y0 = relayout_data.get('yaxis.range[0]')
                    y1 = relayout_data.get('yaxis.range[1]')
                    if x0 is not None or y0 is not None:
                        vp_mask = np.ones(len(filtered_df), dtype=bool)
                        _vp_x = x_plot_col if x_plot_col in filtered_df.columns else x_col
                        if x0 is not None and x1 is not None and _vp_x in filtered_df.columns:
                            vp_mask &= (filtered_df[_vp_x] >= x0) & (filtered_df[_vp_x] <= x1)
                        # Y-range viewport only valid in single-Y mode (multi-Y overlays share one axis)
                        if len(y_cols) == 1 and y0 is not None and y1 is not None and _primary_y in filtered_df.columns:
                            vp_mask &= (filtered_df[_primary_y] >= y0) & (filtered_df[_primary_y] <= y1)
                        viewport_df = filtered_df[vp_mask]
                        if len(viewport_df) > 0:
                            filtered_df = viewport_df
                            _pipeline.append((len(filtered_df), 'viewport'))
                _display_limit = display_pts if display_pts is not None else initial_display_points
                # Skip sampling in multi-Y + agg mode: aggregation collapses rows inside
                # _build_overlay_figure, so sampling here would give statistically wrong means.
                _multi_y_agg = len(y_cols) > 1 and agg_y_mode and agg_y_mode != 'raw'
                if (not _multi_y_agg
                        and len(filtered_df) == total_points
                        and _display_limit and total_points > _display_limit):
                    _use_strat = bool(stratified_sample and 'on' in stratified_sample)
                    filtered_df = (_stratified_sample(filtered_df, _display_limit, color_factor)
                                   if _use_strat else filtered_df.sample(_display_limit, random_state=None))
                    _pipeline.append((len(filtered_df), 'sample'))

                # 6. Row index — always set so customdata stays slim (single int per point)
                filtered_df = filtered_df.copy()
                _is_agg = bool(y_agg_label) or (len(y_cols) > 1 and agg_y_mode != 'raw')
                filtered_df['_row_idx'] = -1 if _is_agg else filtered_df.index

                # Store pipeline result so cosmetic-only re-renders can skip it
                self._pipeline_cache_key = _pipe_key
                self._pipeline_cache_val = (
                    filtered_df.copy(), x_plot_col, x_bin_label, y_agg_label,
                    list(_pipeline), list(derived_discrete), list(current_axis_opts),
                    filter_error, fields_error, fields_info, dict(_ff_dict),
                )

            pipeline_parts = [_fmt(_pipeline[0][0])] if _pipeline else []
            for n, label in (_pipeline[1:] if _pipeline else []):
                pipeline_parts.append(f'{_fmt(n)} ({label})')

            current_cols = [e['col'] for e in (current_derived or [])]
            new_cols     = [e['col'] for e in derived_discrete]
            derived_store_out = (no_update if _use_cache
                                 else (derived_discrete if new_cols != current_cols else no_update))

            try:
                if filtered_df.empty:
                    return _empty('No data for selected filters', x_col, _primary_y)

                filter_dict = {item['id']['factor']: item.get('value') for item in ctx.inputs_list[1]}
                fig = self._build_scatter_from_pipeline(
                    filtered_df, x_col, x_plot_col, y_cols,
                    x_bin_label, y_agg_label, agg_group_by, agg_y_mode,
                    color_factor, filter_dict,
                    color_offsets=color_offsets,
                    template=template,
                    color_values=color_values,
                    color_scale=color_scale,
                    color_scale_reverse=color_scale_reverse,
                    as_continuous=as_continuous,
                    x_jitter=(float(x_jit_min), float(x_jit_max)) if (x_jit_on and 'on' in x_jit_on and x_jit_min is not None and x_jit_max is not None) else None,
                    y_jitter=(float(y_jit_min), float(y_jit_max)) if (y_jit_on and 'on' in y_jit_on and y_jit_min is not None and y_jit_max is not None) else None,
                    marker_size=marker_size,
                    marker_opacity=marker_opacity,
                    overlay_dots=overlay_dots_toggle,
                    overlay_line=overlay_toggle,
                    overlay_ks=(overlay_ks if (overlay_per_line and 'on' in overlay_per_line) else [[overlay_k, overlay_s]]),
                    anchor_left=anchor_left,
                    anchor_right=anchor_right,
                    show_factor_name=legend_show_factor,
                    skip_sparse=skip_sparse,
                    line_width=line_width,
                    line_opacity=line_opacity,
                    y_styles=overlay_y_styles or [],
                    x_bin_width=x_bin_width,
                )
                fig, vp_out = _apply_viewport(fig)
                fig, tv_out = _apply_trace_visibility(fig)
                trace_row_map = [
                    (t.customdata.tolist() if t.customdata is not None and len(t.customdata) > 0 else [])
                    for t in fig.data
                ]
                if _use_cache:
                    return (fig, vp_out, tv_out,
                            no_update, ' → '.join(pipeline_parts), no_update, no_update,
                            no_update, no_update,
                            no_update, no_update, no_update, trace_row_map)
                return (fig, vp_out, tv_out,
                        filter_error, ' → '.join(pipeline_parts), fields_error, fields_info,
                        current_axis_opts, current_axis_opts,
                        derived_store_out,
                        _build_color_by_options(derived_discrete),
                        _build_agg_options(derived_discrete),
                        trace_row_map)
            except Exception:
                err = _traceback.format_exc()
                print(f'[render_figure ERROR]\n{err}', flush=True, file=__import__('sys').stderr)
                return _empty('render_figure error — check server log')

        # ── Theme patch (tiny diff, no data transfer) ───────────────────────────────────
        @callback(
            Output('scatter-plot', 'figure', allow_duplicate=True),
            Input('theme-toggle', 'value'),
            State('scatter-plot', 'figure'),
            prevent_initial_call=True,
        )
        def update_theme(theme, current_fig):
            if not current_fig or not current_fig.get('data'):
                return no_update
            patch = Patch()
            patch['layout']['template'] = (
                'plotly_white' if (theme and 'light' in theme) else 'plotly_dark'
            )
            return patch

        # ── Stage 3: Clientside colorscale update (continuous color mode only) ─
        # color-scale-picker / color-scale-reverse are no longer Inputs to
        # render_figure; this callback handles them without a server round-trip.
        # It only acts when layout.coloraxis exists (continuous coloring mode).
        app.clientside_callback(
            """
            function(scale, reverse, figure) {
                if (!figure || !figure.layout || !figure.layout.coloraxis)
                    return window.dash_clientside.no_update;
                var rev = reverse && reverse.includes('on');
                var fullscale = (scale || 'Viridis') + (rev ? '_r' : '');
                var layout = Object.assign({}, figure.layout, {
                    coloraxis: Object.assign({}, figure.layout.coloraxis, {colorscale: fullscale})
                });
                return Object.assign({}, figure, {layout: layout});
            }
            """,
            Output('scatter-plot', 'figure', allow_duplicate=True),
            Input('color-scale-picker',  'value'),
            Input('color-scale-reverse', 'value'),
            State('scatter-plot', 'figure'),
            prevent_initial_call=True,
        )

        # ── Stage 4: Clientside marker update (zero server round-trip) ────────
        app.clientside_callback(
            """
            function(size, opacity, figure) {
                if (!figure || !figure.data || !figure.data.length) return window.dash_clientside.no_update;
                var traces = figure.data.map(function(t) {
                    return Object.assign({}, t, {marker: Object.assign({}, t.marker || {}, {size: size, opacity: opacity})});
                });
                return Object.assign({}, figure, {data: traces});
            }
            """,
            Output('scatter-plot', 'figure', allow_duplicate=True),
            Input('setting-marker-size',    'value'),
            Input('setting-marker-opacity', 'value'),
            State('scatter-plot', 'figure'),
            prevent_initial_call=True,
        )

        # ── Stage 5: Track client-side trace visibility (legend clicks) ─────────
        # State('scatter-plot', 'figure') only returns the last server-sent figure;
        # legend clicks toggle visibility in the browser but never propagate back.
        # This callback (a) snapshots visibility when the server pushes a new figure
        # and (b) re-registers a plotly_restyle listener so any legend click updates
        # the store. save_view reads from this store instead of from the figure State.
        app.clientside_callback(
            """
            function(figure, prevVis) {
                function captureVis(data) {
                    var vis = {};
                    (data || []).forEach(function(t) {
                        if (t.name) {
                            var v = t.visible;
                            vis[t.name + '|' + (t.mode || 'markers')] =
                                (v === undefined || v === true) ? true : v;
                        }
                    });
                    return vis;
                }
                setTimeout(function() {
                    var gd = document.querySelector('#scatter-plot .js-plotly-plot');
                    if (!gd) return;
                    gd.removeAllListeners('plotly_restyle');
                    gd.on('plotly_restyle', function() {
                        dash_clientside.set_props('live-trace-visibility',
                            {data: captureVis(gd.data)});
                    });
                }, 150);
                var newVis = captureVis((figure || {}).data);
                var prev = prevVis || {};
                var merged = {};
                for (var k in newVis) {
                    // Server doesn't know about legend clicks — preserve legendonly
                    // from previous state when the server only says true.
                    merged[k] = (newVis[k] === true && prev[k] === 'legendonly')
                        ? 'legendonly' : newVis[k];
                }
                return merged;
            }
            """,
            Output('live-trace-visibility', 'data'),
            Input('scatter-plot', 'figure'),
            State('live-trace-visibility', 'data'),
            prevent_initial_call=True,
        )

    # ── PNG route ─────────────────────────────────────────────────────────────

    def _register_export_route(self):
        """
        Serve a PNG snapshot of any saved view via GET request.

        Usage:
            curl 'http://127.0.0.1:8051/?my_view&format=png&width=1200&height=800' -o my_view.png
        """
        from .utils import register_export_route

        def _reconstruct(view_data, width, height, theme_override):
            return self._reconstruct_figure(view_data, theme_override=theme_override)

        register_export_route(
            self.app,
            reconstruct_fn=_reconstruct,
            safe_roots=self._safe_roots,
            default_search_dirs=[Path(self._initial_views_dir), Path('/out/dashboard'), Path('/tmp/dashboard')],
        )

    # ── Assembly ──────────────────────────────────────────────────────────────

    def build(self) -> Dash:
        sample_cols = self.axis_options[:6]
        cols_hint = ', '.join(sample_cols) + (', ...' if len(self.axis_options) > 6 else '')

        app = Dash(__name__, suppress_callback_exceptions=True,
                   url_base_pathname=self._url_prefix)
        app.index_string = (
            app.index_string
            .replace('<body>', '<body style="background:#1e1e1e;color:#ddd;margin:0;">')
            .replace('</head>', DARK_CSS)
        )
        app.layout = build_layout(
            discrete_factors=self.discrete_factors,
            factor_values=self.factor_values,
            axis_options=self.axis_options,
            default_x=self.default_x,
            default_y=self.default_y,
            color_by_options=self.color_by_options,
            initial_display_points=self.initial_display_points,
            discrete_threshold=self.discrete_threshold,
            has_params_json=self.has_params_json,
            cols_hint=cols_hint,
            initial_views_dir=self._initial_views_dir,
        )
        self.app = app
        self._register_callbacks()
        self._register_export_route()
        from wsgi.base import register_view_refresh
        register_view_refresh(
            app,
            open_trigger_id='views-toggle',
            list_options_fn=lambda p: _view_dropdown_options(_views_dir(p)),
            body_state_id='views-body',
        )
        return app

def create_scatter_dashboard(
    df: pd.DataFrame,
    factors: list[str],
    discrete_threshold: int = 100,
    renderer: str = 'auto',
    initial_display_points: int = 10_000,
    safe_roots: list[str] = None,
    initial_views_dir: str = '/out/dashboard',
    url_prefix: str = '/',
) -> Dash:
    return ScatterDashboard(
        df, factors,
        discrete_threshold=discrete_threshold,
        renderer=renderer,
        initial_display_points=initial_display_points,
        safe_roots=safe_roots or ['/out', '/tmp'],
        initial_views_dir=initial_views_dir,
        url_prefix=url_prefix,
    ).build()

