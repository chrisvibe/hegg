import numpy as np
from project.boolean_reservoir.code.utils.utils import set_seed
from benchmark.path_integration.constrained_foraging_path import generate_dual_trajectory, random_walk, to_polar, to_cartesian
from benchmark.path_integration.visualization import plot_random_walk
from benchmark.path_integration.parameter import PathIntegrationDatasetParams
from benchmark.utils.base_dataset import BaseDataset


class ConstrainedForagingPathDataset(BaseDataset):
    def __init__(self, D: PathIntegrationDatasetParams):
        super().__init__(D)
        set_seed(D.seed)

        if self.data_path.exists() and not D.generate_data:
            self.load_data()
        else:
            raw_data = self.generate_data(D.dimensions, D.samples, D.steps, D.strategy_obj, D.boundary_obj, D.coordinate, D.reset, D.mode, D.output_coordinate)
            self.set_data(raw_data)
            self.save_data()
        if D.shuffle:
            self.shuffle_data()

    @staticmethod
    def generate_data(dimensions, samples, n_steps, strategy, boundary, coordinate_system='cartesian', reset=True, mode='displacement', output_coordinate='cartesian'):
        data_x = []
        data_y = []
        origin = None

        for _ in range(samples):
            d = random_walk(dimensions, n_steps, strategy, boundary, origin=origin)
            positions = d['positions']
            p_final = positions[-1]

            if not reset:
                raise NotImplementedError("reset=False is not implemented yet. y label is coupled with boundary and this would cause drift")

            if mode == 'acceleration':
                x_data = d['a_net']
            elif mode == 'velocity':
                x_data = d['velocities']
            elif mode == 'displacement':
                x_data = np.diff(positions, axis=0)
            else:
                raise ValueError(f"Invalid mode: {mode}. Must be one of 'acceleration', 'velocity', or 'displacement'.")

            if coordinate_system == 'polar':
                x_data = to_polar(x_data)

            if output_coordinate == 'polar':
                p_final = to_polar(p_final)

            data_x.append(x_data.astype(np.float32))
            data_y.append(p_final.astype(np.float32))

        return {
            'x': np.stack(data_x),
            'y': np.stack(data_y),
        }
