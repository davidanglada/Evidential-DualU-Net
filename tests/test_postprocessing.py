import numpy as np
from evidential_dualunet.postprocessing import reconstruct_instances


def test_watershed_toy_input():
    foreground = np.zeros((32, 32), dtype=np.float32)
    foreground[4:14, 4:14] = 1; foreground[18:28, 18:28] = 1
    labels = reconstruct_instances(foreground)
    assert labels.shape == foreground.shape
    assert labels.dtype == np.int32
    assert labels.max() >= 2

