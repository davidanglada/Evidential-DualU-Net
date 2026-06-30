import torch
from evidential_dualunet.uncertainty import dirichlet_probabilities, dirichlet_uncertainty


def test_dirichlet_shapes_and_probability_sum():
    alpha = torch.rand(2, 4, 8, 9) + 1
    probabilities = dirichlet_probabilities(alpha)
    maps = dirichlet_uncertainty(alpha)
    assert probabilities.shape == alpha.shape
    assert torch.allclose(probabilities.sum(1), torch.ones(2, 8, 9), atol=1e-6)
    assert all(value.shape == (2, 8, 9) for value in maps.values())
    assert all(torch.isfinite(value).all() for value in maps.values())

