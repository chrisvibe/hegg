"""
Analytical verification tests for path integration.

Tests that displacement mode is solvable by a purely linear model:
    y[i] = sum_t( decode(x[i, t]) )

If a ridge model on analytically decoded inputs achieves >90% accuracy, it confirms:
  1. The base2 encoding is lossless enough at the configured resolution
  2. The task labels are correctly formulated
  3. The dataset pipeline produces consistent (x, y) pairs

These tests use no reservoir — they are mathematical sanity checks that any reviewer
familiar with path integration can verify by inspection.

NOTE — PathIntegrationVerificationModel (nn.Linear learnable decoder):
The original file also contained a gradient-based verification model using nn.Linear
to learn the encoding. That variant was removed when online/gradient training was
dropped (torch-removal branch). Restore it here once online learning returns.
"""
import numpy as np
import pytest

from project.boolean_reservoir.code.parameter import load_yaml_config
from project.boolean_reservoir.code.encoding import bin2dec

CONFIG_1D = 'config/test/path_integration/1D/grid_search/verification_model.yaml'
CONFIG_2D = 'config/test/path_integration/2D/grid_search/verification_model.yaml'
ACCURACY_THRESHOLD = 0.9


def _analytical_features(x: np.ndarray) -> np.ndarray:
    """Decode binary-encoded steps and sum across time: the analytical solution for displacement mode.

    x: (m, s, d, b)  →  returns (m, d)
    """
    m, s, d, b = x.shape
    decoded = bin2dec(x.reshape(m * s * d, b), b)   # (m*s*d,)
    return decoded.reshape(m, s, d).sum(axis=1)       # (m, d)


def _augment(F: np.ndarray) -> np.ndarray:
    """Append bias column to feature matrix."""
    return np.concatenate([F, np.ones((len(F), 1), dtype=np.float64)], axis=1)


def _fit_and_eval(dataset, T) -> float:
    """Fit ridge on analytical features (with bias) and return test accuracy."""
    F_train = _augment(_analytical_features(dataset.data['x']).astype(np.float64))
    y_train = dataset.data['y'].astype(np.float64)

    W = np.linalg.solve(
        F_train.T @ F_train + 1e-3 * np.eye(F_train.shape[1]),
        F_train.T @ y_train,
    )

    x_split = 'x_' + T.evaluation
    y_split = 'y_' + T.evaluation
    F_eval = _augment(_analytical_features(dataset.data[x_split]).astype(np.float64))
    y_hat  = F_eval @ W
    y_eval = dataset.data[y_split]

    distances = np.sqrt(np.sum((y_hat - y_eval) ** 2, axis=1))
    return float((distances < T.accuracy_threshold).mean())


@pytest.mark.parametrize("config_path", [CONFIG_1D, CONFIG_2D])
def test_displacement_analytically_solvable(config_path):
    """Displacement mode is analytically solvable by decode + sum: accuracy must be ≥ 90 %."""
    P = load_yaml_config(config_path)
    dataset = P.dataset_init_obj.train(P)
    accuracy = _fit_and_eval(dataset, P.M.T)
    assert accuracy >= ACCURACY_THRESHOLD, (
        f"Analytical displacement model accuracy {accuracy:.3f} < {ACCURACY_THRESHOLD} "
        f"for {config_path} — encoding may be lossy or task labels inconsistent."
    )


# NOTE — acceleration mode (verification_model_gravity.yaml):
# x = external accelerations, y = double integral of acceleration through physics.
# Not analytically solvable by decode+sum — requires a learnable decoder (nn.Linear).
# TODO: restore when online/gradient training returns.
