"""
FYS-STK4155 @UiO, PROJECT I.
Testing: Franke's function.
"""
import pytest
import numpy as np

from src.franke import franke_function


def test_franke() :
    """
    Validates Franke's function against data from [*].
    [*] https://www.sfu.ca/~ssurjano/Code/franke2dm.html
    """
    x = np.array(
        [
            0.5185949425105382,
            0.6489914927123561,
            0.6596052529083072,
            0.8003305753524015,
            0.9729745547638625,
            ]
    )
    y = np.array(
        [
            0.0834698148589140,
            0.1331710076071617,
            0.4323915037834617,
            0.4537977087269195,
            0.8253137954020456,
        ]
    )
    f = np.array(
        [
            0.4488306370927234,
            0.4063778975108695,
            0.4875600327917881,
            0.4834151909828157,
            0.0479911637101943,
        ]
    )
    assert np.allclose(franke_function(x, y), f)
