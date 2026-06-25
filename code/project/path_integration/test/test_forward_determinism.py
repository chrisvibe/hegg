"""Regression test: model forward pass output must be deterministic across refactors.

To recapture the hash after an intentional behaviour change:
    set EXPECTED_HASH = '5eb31b394425ed3593da5b2d3b507a4a'

Usage:
    pytest project/path_integration/test/test_model.py -v -s
"""
import hashlib
from project.boolean_reservoir.code.parameter import load_yaml_config
from project.boolean_reservoir.code.utils.param_utils import generate_param_combinations
from project.boolean_reservoir.code.reservoir import BooleanReservoir

CONFIG = 'config/path_integration/2D/grid_search/design_choices/continuous.yaml'

EXPECTED_HASH = '5eb31b394425ed3593da5b2d3b507a4a'


def test_model_forward_hash():
    P = load_yaml_config(CONFIG)
    P0 = generate_param_combinations(P)[0]

    dataset = P0.dataset_init_obj.train(P0)
    model = BooleanReservoir(P0)
    model.eval()

    y_hat = model(dataset.data['x'])

    h = hashlib.md5(y_hat.tobytes()).hexdigest()
    print(f"\nOutput hash: {h}")

    if EXPECTED_HASH is not None:
        assert h == EXPECTED_HASH, f"Output changed: {h!r} != {EXPECTED_HASH!r}"
    else:
        print("Paste the hash above into EXPECTED_HASH and re-run.")
