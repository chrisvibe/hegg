import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib
import numpy as np
from pathlib import Path
matplotlib.use('Agg')

_KINEMATIC_ORDER = {
    'positions': 0,
    'velocities': 1,
    'a_ext': 2, 'a_net': 2, 'a_bnd': 2,
}

_SIGNAL_LABELS = {
    'positions':  'displacement',
    'velocities': 'velocity',
    'a_ext':      'ext. acceleration',
    'a_net':      'net acceleration',
    'a_bnd':      'boundary acceleration',
}

def _auto_label(signal, suffix, extra=''):
    sig = _SIGNAL_LABELS.get(signal, signal.replace('_', ' '))
    return f'{suffix.title()} {sig}{extra}'

def _to_path_dict(x):
    if isinstance(x, np.ndarray):
        return {'positions': x}
    return x

def _naive_path(start, signal, n_integrate):
    """Cumulative-integrate signal n_integrate times and prepend start as origin."""
    path = signal.copy().astype(float)
    for _ in range(n_integrate):
        path = np.cumsum(path, axis=0)
    return np.vstack([start[np.newaxis], path])

def _resolve_comparison(path, path_b, suffix_a, suffix_b):
    """Normalise inputs to a comparison dict and a has_dual flag."""
    path = _to_path_dict(path)
    if path_b is not None:
        path_b = _to_path_dict(path_b)
        d = {f'{k}_{suffix_a}': v for k, v in path.items()}
        d.update({f'{k}_{suffix_b}': v for k, v in path_b.items()})
        return d, True
    elif f'positions_{suffix_a}' in path:
        return path, True   # already a comparison dict (e.g. from generate_dual_trajectory)
    else:
        d = {f'{k}_{suffix_a}': v for k, v in path.items()}
        return d, False     # single path — no dual line

def _plot_traj(ax, data, dimensions, time, **kwargs):
    if dimensions == 1:
        ax.plot(data[:, 0], time, **kwargs)
    elif dimensions == 2:
        ax.plot(data[:, 0], data[:, 1], **kwargs)
    else:
        ax.plot(data[:, 0], data[:, 1], data[:, 2], **kwargs)

def plot_random_walk(dir_path, path, strategy, boundary, path_b=None,
                     suffix_a='real', suffix_b='ideal',
                     label=None, dual_label=None,
                     primary='positions', overlays=None,
                     file_prepend='', sub_dir='visualizations/random_walk'):
    """
    path: single random_walk dict, comparison dict from compare_paths /
          generate_dual_trajectory, or a raw numpy positions array.
    path_b: optional second random_walk dict or positions array. When provided,
            compare_paths is called internally to merge path and path_b.
    primary: key stem for the reference axes ('positions', 'velocities', …).
    overlays: key stems to integrate onto the primary axes (abs kinematic order difference
              determines how many cumsums are applied).

    Examples (d = generate_dual_trajectory(...)):
        plot_random_walk(dir, d, ...)                                        # real vs ideal displacement
        plot_random_walk(dir, extract_path(d, 'real'), ..., overlays=['velocities'])          # + naive velocity
        plot_random_walk(dir, extract_path(d, 'real'), ..., overlays=['velocities', 'a_ext']) # + naive accel
        plot_random_walk(dir, extract_path(d, 'ideal'), ..., overlays=['a_ext'])              # frictionless world
        plot_random_walk(dir, d, ..., primary='velocities')                  # real vs ideal velocity
        plot_random_walk(dir, extract_path(d, 'real'), ..., primary='velocities', overlays=['a_ext'])  # accel vs vel
    """
    comparison, has_dual = _resolve_comparison(path, path_b, suffix_a, suffix_b)

    primary_data = comparison[f'{primary}_{suffix_a}']
    if primary_data.ndim == 1:
        primary_data = primary_data[:, np.newaxis]

    secondary_data = comparison.get(f'{primary}_{suffix_b}') if has_dual else None
    if secondary_data is not None and secondary_data.ndim == 1:
        secondary_data = secondary_data[:, np.newaxis]

    n_steps = primary_data.shape[0] - 1
    dimensions = primary_data.shape[1]
    time = np.arange(primary_data.shape[0])

    fig = plt.figure(figsize=(10, 10))
    if dimensions == 3:
        ax = fig.add_subplot(111, projection='3d')
        ax.set_aspect('equal', adjustable='box')
    else:
        ax = fig.add_subplot(111)
        if dimensions == 2:
            ax.set_aspect('equal', adjustable='box')

    _plot_traj(ax, primary_data, dimensions, time, color='C0',
               label=label or _auto_label(primary, suffix_a))
    if secondary_data is not None:
        _plot_traj(ax, secondary_data, dimensions, time, color='C0', linestyle='--',
                   label=dual_label or _auto_label(primary, suffix_b))

    if overlays:
        order_primary = _KINEMATIC_ORDER.get(primary, 0)
        for i, stem in enumerate(overlays):
            signal = comparison.get(f'{stem}_{suffix_a}')
            if signal is None:
                signal = comparison.get(stem)  # bare key (e.g. a_ext from generate_dual_trajectory)
            if signal is None:
                continue
            if signal.ndim == 1:
                signal = signal[:, np.newaxis]
            n_integrate = abs(_KINEMATIC_ORDER.get(stem, 0) - order_primary)
            naive = _naive_path(primary_data[0], signal, n_integrate)
            if len(naive) > len(primary_data):
                naive = naive[1:]
            _plot_traj(ax, naive, dimensions, time[:len(naive)],
                       color=f'C{i + 2}', label=_auto_label(stem, suffix_a, ' (naive)'))

            signal_opt = comparison.get(f'{stem}_{suffix_b}')
            if signal_opt is not None and has_dual and secondary_data is not None:
                if signal_opt.ndim == 1:
                    signal_opt = signal_opt[:, np.newaxis]
                naive_opt = _naive_path(secondary_data[0], signal_opt, n_integrate)
                if len(naive_opt) > len(primary_data):
                    naive_opt = naive_opt[1:]
                _plot_traj(ax, naive_opt, dimensions, time[:len(naive_opt)],
                           color=f'C{i + 2}', linestyle='--',
                           label=_auto_label(stem, suffix_b, ' (naive)'))

    sig_lbl = _SIGNAL_LABELS.get(primary, primary.replace('_', ' '))
    if dimensions == 1:
        ax.set_xlabel(sig_lbl.capitalize())
        ax.set_ylabel('Time')
    elif dimensions == 2:
        ax.set_xlabel(f'X {sig_lbl}')
        ax.set_ylabel(f'Y {sig_lbl}')
    else:
        ax.set_xlabel(f'X {sig_lbl}')
        ax.set_ylabel(f'Y {sig_lbl}')
        ax.set_zlabel(f'Z {sig_lbl}')
    if primary == 'positions':
        boundary_points = boundary.get_points()
        if boundary_points:
            if dimensions == 1:
                ax.axvline(x=boundary_points[0], color='r', linestyle='--', linewidth=2, label='Boundary')
                ax.axvline(x=boundary_points[1], color='r', linestyle='--', linewidth=2)
            elif dimensions == 2:
                polygon = patches.Polygon(boundary_points, linestyle='--', linewidth=2, edgecolor='r', facecolor='none')
                ax.add_patch(polygon)

    ax.set_title('Constrained Foraging Path')
    ax.legend()
    plt.ion()

    out = Path(dir_path) / sub_dir
    out.mkdir(parents=True, exist_ok=True)
    prepend = file_prepend + '_' if file_prepend else ''
    primary_tag = '' if primary == 'positions' else f'-{primary}'
    overlay_tag = '-' + '+'.join(overlays) if overlays else ''
    file_name = f'{prepend}{dimensions}D-s={n_steps}{primary_tag}{overlay_tag}-{strategy}-{boundary}.svg'
    plt.savefig(out / file_name, bbox_inches='tight')
    plt.close(fig)

def plot_random_walk_model(dir_path, x: np.array, model, y: np.array):
    # TODO problems with normalization. Sum x over steps is not y if scaled differently
    m, s, d, _ = x.shape
    # incrementally consideres more steps to visualize error divergence
    # y is used to verify model correctness
    data = np.zeros((m, s, d, 2)) # add a dimension to contain both y_hat and y_ij
    for i in range(m) - 1:
        for j in range(s):
            x_ij = x[i:i+1, :j+1]
            y_ij = np.sum(x_ij, dim=1)   # TODO use cum sum instead outside of the loop!
            y_hat = model(x_ij)
            data[i, j, :, 0] = y_hat
            data[i, j, :, 1] = y_ij
        assert y_ij == y # when all steps are taken label should match sum of steps

    # plot two curves per path of lenth s, one with the y_hat and the other with y_ij
    # note that both y_hat and y_ij are a set of x, y coordinates when d = 2

    fig, ax = plt.subplots(figsize=(10, 10))
    for i in range(m): # plot each path
        ax.plot(data[i])

    ax.set_title('Incremental error in path integration')
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    plt.ion()

    path = Path(dir_path) / 'visualizations/random_walk'
    path.mkdir(parents=True, exist_ok=True)
    file_name = 'todo'
    plt.savefig(path / file_name, bbox_inches='tight')


if __name__ == '__main__':
    pass
