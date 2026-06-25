from benchmark.path_integration.visualization import plot_random_walk
from benchmark.path_integration.constrained_foraging_path import generate_dual_trajectory, extract_path
from project.boolean_reservoir.code.visualization import plot_train_history, plot_predictions_and_labels, plot_dynamics_history
from project.boolean_reservoir.code.parameter import load_yaml_config
import matplotlib
matplotlib.use('Agg')

def plot_random_walk_from_config(config_path, out_dir='/out/', overlays=None, m=1):
    P = load_yaml_config(config_path)
    D = P.D
    for i in range(m):
        d = generate_dual_trajectory(D.dimensions, D.steps, D.strategy_obj, D.boundary_obj)
        file_prepend = f'{i}_' if m > 1 else ''
        plot_random_walk(out_dir, extract_path(d, 'real'), D.strategy_obj, D.boundary_obj, overlays=overlays, file_prepend=file_prepend)

def plot_many_things(model, dataset, history):
    x_test = dataset.data['x_test'][:500]
    y_test = dataset.data['y_test'][:500]
    y_hat_test = model(x_test)
    plot_train_history(model.save_path, history)
    plot_predictions_and_labels(model.save_path, y_hat_test, y_test, tolerance=model.T.accuracy_threshold, axis_limits=[-1, 2])
    # plot_dynamics_history(model.save_path)
    # plot_graph_with_weight_coloring_3D(model.graph, model.readout)

if __name__ == '__main__':
    pass