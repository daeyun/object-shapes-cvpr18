import random
from os import path

import numpy as np
from mvshape import pcl_utils


def test_alignment():
    source = np.random.randn(100, 3)
    target = source * 42 + np.array([1, 2, 3])

    offset, scale = pcl_utils.find_aligning_transformation(source, target)

    assert np.allclose(offset, np.array([1, 2, 3]))
    assert np.isclose(scale, 42)
