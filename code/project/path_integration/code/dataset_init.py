import copy
from project.boolean_reservoir.code.encoding import BooleanEncoder, MinMaxNormalization, StandardScaler
from project.boolean_reservoir.code.parameter import Params
from project.boolean_reservoir.code.utils.utils import balance_dataset, l2_distance, l2_distance_squared
from project.boolean_reservoir.code.train_model import DatasetInit, KQGRInit
from benchmark.path_integration.constrained_foraging_path_dataset import ConstrainedForagingPathDataset


class PathIntegrationDatasetInit(KQGRInit, DatasetInit):

    def _create_raw_dataset(self, P: Params):
        cache_key = str(('raw', self.__class__.__name__, P.D.model_dump()))
        def _build():
            dataset = ConstrainedForagingPathDataset(P.D)
            if P.D.reset:
                if P.D.dimensions == 1:
                    dataset = balance_dataset(dataset, distance_fn=l2_distance, num_bins=100)
                elif P.D.dimensions == 2:
                    # l2 distance squared is more fair for 2D as area grows per radial slice
                    dataset = balance_dataset(dataset, distance_fn=l2_distance_squared, num_bins=100)
            return dataset
        return self._cache_dataset(cache_key, _build)

    def _process_dataset(self, dataset, P: Params):
        dataset = copy.deepcopy(dataset)
        dataset.set_normalizer_x(MinMaxNormalization())
        dataset.set_normalizer_y(StandardScaler())
        dataset.normalize()
        encoder = BooleanEncoder(P)
        dataset.set_encoder_x(encoder)
        dataset.encode_x()
        dataset.split_dataset()
        return dataset


if __name__ == '__main__':
    from project.boolean_reservoir.code.parameter import load_yaml_config
    from project.boolean_reservoir.code.utils.param_utils import generate_param_combinations
    P = load_yaml_config('config/path_integration/kq_and_gr/grid_search/test.yaml')
    P.U.kqgr.D.samples = 10
    P.M.I.redundancy = 2
    P.M.I.features = 2
    p = generate_param_combinations(P)[0]
    kq = PathIntegrationDatasetInit().kqgr(p.U.kqgr, kq=True)
    gr = PathIntegrationDatasetInit().kqgr(p.U.kqgr, kq=False)
    tau = p.U.kqgr.D.tau
    assert (gr.data['x'][0, ..., -tau:] != gr.data['x'][..., -tau:]).sum() == 0
    assert (kq.data['x'][..., :-tau] != gr.data['x'][..., :-tau]).sum() == 0
    pass
