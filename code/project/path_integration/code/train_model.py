from project.boolean_reservoir.code.train_model import train_single_model
from project.boolean_reservoir.code.visualization import plot_train_history, plot_dynamics_history, plot_activity_trace
from project.path_integration.code.visualization import plot_many_things
from project.boolean_reservoir.code.train_model_parallel import boolean_reservoir_grid_search
from project.boolean_reservoir.code.utils.utils import run_on_nodes
import logging
from project.boolean_reservoir.code.utils.utils import configure_logging
configure_logging()
logger = logging.getLogger(__name__)

if __name__ == '__main__':
    # Simple single-run examples (uncomment to use):
    # p, model, dataset, history = train_single_model('config/path_integration/1D/single_run/good_model.yaml')
    # plot_many_things(model, dataset, history)

    configs = [
        'config/path_integration/1D/grid_search/design_choices/continuous_redundancy.yaml',
        'config/path_integration/2D/grid_search/design_choices/continuous_redundancy.yaml',
        'config/path_integration/1D/grid_search/design_choices/continuous.yaml',
        'config/path_integration/2D/grid_search/design_choices/continuous.yaml',
    ]

    # submit with cpu_light.slurm as there are L3 cache misses, or cpu_max_workers works too (applies to high input models ie redundancy = 10)
    # run_on_nodes(configs, run_fn=lambda p: boolean_reservoir_grid_search(p, cpu_max_workers=18))
    run_on_nodes(configs, run_fn=lambda p: boolean_reservoir_grid_search(p))


