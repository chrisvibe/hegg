import numpy as np
import pytest
from benchmark.path_integration.constrained_foraging_path import to_polar, to_cartesian


def test_to_polar_2d():
    assert np.allclose(to_polar([1, 0]), [1, 0])
    assert np.allclose(to_polar([0, 1]), [1, np.pi/2])
    assert np.allclose(to_polar([-1, 0]), [1, np.pi])
    assert np.allclose(to_polar([1, 1]), [np.sqrt(2), np.pi/4])


def test_to_polar_3d():
    assert np.allclose(to_polar([1, 0, 0]), [1, 0, np.pi/2])
    assert np.allclose(to_polar([0, 1, 0]), [1, np.pi/2, np.pi/2])
    result = to_polar([0, 0, 1])
    assert np.allclose(result[[0, 2]], [1, 0], atol=1e-4)


@pytest.mark.parametrize("dim", [1, 2, 3])
def test_polar_cartesian_roundtrip(dim):
    np.random.seed(0)
    cart = np.random.randn(100, dim)
    polar = to_polar(cart)
    reconstructed = to_cartesian(polar)
    assert np.allclose(cart, reconstructed, atol=1e-6)
