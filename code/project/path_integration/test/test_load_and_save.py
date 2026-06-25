import pytest
import numpy as np
from shutil import rmtree
from pathlib import Path

from project.boolean_reservoir.code.reservoir import BooleanReservoir
from project.boolean_reservoir.code.train_model import train_single_model
from project.boolean_reservoir.code.utils.utils import configure_logging

CONFIG = 'config/test/path_integration/2D/single_run/test_model.yaml'
OUT = Path('/tmp/boolean_reservoir/test/path_integration/2D/single_run/test_model')


def _model_likeness_check(m1: BooleanReservoir, m2: BooleanReservoir, dataset):
    """Assert two models have identical structure, weights, and predictions."""
    m1.reset_reservoir(hard_reset=True)
    m2.reset_reservoir(hard_reset=True)

    assert m1.P.model == m2.P.model, "model parameters do not match"
    assert np.array_equal(m1.readout_W, m2.readout_W), "readout_W does not match"
    assert np.array_equal(m1.readout_b, m2.readout_b), "readout_b does not match"
    assert np.array_equal(m1.w_bi, m2.w_bi), "w_bi does not match"
    assert np.array_equal(m1.lut, m2.lut), "lut does not match"
    assert np.array_equal(m1.initial_states, m2.initial_states), "initial_states do not match"
    assert (
        list(m1.graph.edges(data=True)) == list(m2.graph.edges(data=True))
    ), "graph structure does not match"

    x_dev, y_dev = dataset.data['x_dev'], dataset.data['y_dev']
    m1.eval()
    m2.eval()
    assert np.array_equal(m1(x_dev), m2(x_dev)), "model predictions do not match"


def test_saving_and_loading_models():
    """Weights and predictions survive a save/load round-trip."""
    if OUT.exists():
        assert "/tmp/boolean_reservoir/test" in str(OUT)
        rmtree(OUT)

    _, model, dataset, _ = train_single_model(CONFIG)
    model2 = BooleanReservoir(load_path=model.P.L.last_checkpoint)

    _model_likeness_check(model, model2, dataset)


def test_reproducibility_of_training():
    """Two training runs with the same config and seeds produce identical weights."""
    if OUT.exists():
        assert "/tmp/boolean_reservoir/test" in str(OUT)
        rmtree(OUT)

    _, model1, dataset, _ = train_single_model(CONFIG, save_model=False)
    _, model2, _,       _ = train_single_model(CONFIG, save_model=False)

    _model_likeness_check(model1, model2, dataset)


if __name__ == '__main__':
    configure_logging()
    test_saving_and_loading_models()
    test_reproducibility_of_training()
