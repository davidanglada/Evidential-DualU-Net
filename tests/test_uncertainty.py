import torch
import numpy as np
from evidential_dualunet.uncertainty import (
    dirichlet_probabilities,
    dirichlet_uncertainty,
    geometric_centroid_uncertainty,
    pool_instance_uncertainty,
)


def test_dirichlet_shapes_and_probability_sum():
    alpha = torch.rand(2, 4, 8, 9) + 1
    probabilities = dirichlet_probabilities(alpha)
    maps = dirichlet_uncertainty(alpha)
    assert probabilities.shape == alpha.shape
    assert torch.allclose(probabilities.sum(1), torch.ones(2, 8, 9), atol=1e-6)
    assert all(value.shape == (2, 8, 9) for value in maps.values())
    assert all(torch.isfinite(value).all() for value in maps.values())


def test_paper_instance_pooling_is_mean_and_excludes_background():
    alpha = torch.tensor(
        [
            [[20.0, 20.0], [1.0, 1.0]],  # background
            [[2.0, 4.0], [1.0, 1.0]],    # foreground class 1
            [[6.0, 8.0], [1.0, 1.0]],    # foreground class 2
        ]
    )
    instances = torch.tensor([[1, 1], [0, 0]])
    scores = pool_instance_uncertainty(alpha, instances)
    expected = dirichlet_uncertainty(torch.tensor([[3.0, 7.0]]), class_dim=1)
    assert scores[1]["class_id"] == 2
    assert scores[1]["strength"] == 10.0
    assert abs(scores[1]["vacuity"] - expected["vacuity"].item()) < 1e-6


def test_paper_centroid_geometric_scores():
    centroid = np.zeros((5, 5), dtype=np.float32)
    instances = np.zeros((5, 5), dtype=np.int32)
    instances[1:4, 1:4] = 1
    centroid[2, 2] = 0.8
    result = geometric_centroid_uncertainty(centroid, instances, sigma=1.0)
    assert abs(result[1]["peak"] - 0.2) < 1e-6
    expected_mass = abs(0.8 - 2 * np.pi) / (2 * np.pi)
    assert abs(result[1]["mass"] - expected_mass) < 1e-6
