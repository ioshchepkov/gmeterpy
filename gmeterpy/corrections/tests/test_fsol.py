"""
Test finite speed of light corrections module.

"""

import pytest
import numpy as np

import gmeterpy.units as u

from gmeterpy.corrections.fsol import finite_speed_of_light_correction


def test_finite_speed_of_light_correction():
    """Test finite speed of light est correction.

    """
    g0 = 9.81 * u.m / u.s**2
    v0 = 0.4 * u.m / u.s
    T = 0.17 * u.s

    with pytest.raises(ValueError):
        finite_speed_of_light_correction(g0, v0, T, schema='klmn')

    fsof = finite_speed_of_light_correction(g0, v0, T, schema='esd').to('uGal')

    # test units
    assert isinstance(fsof, u.Quantity)

    # TEST ESD

    # test sign
    fsof_above = finite_speed_of_light_correction(g0, v0, T,
                                                  beam_splitter_above=True,
                                                  schema='esd').to('uGal')

    assert np.all(fsof_above.value >= 0.0)
    assert np.all(fsof.value <= 0.0)

    np.testing.assert_array_almost_equal(fsof.value, -fsof_above.value)

    # test numerical value
    fsof_ref = -12.67  # uGal

    np.testing.assert_array_almost_equal(fsof.value, fsof_ref, decimal=2)

    # TEST EST

    g0 = 980483871.46 * u.uGal
    v0 = 0.620 * u.m / u.s
    T = 0.204 * u.s

    fsof = finite_speed_of_light_correction(g0, v0, T, schema='est').to('uGal')

    # test sign
    fsof_above = finite_speed_of_light_correction(g0, v0, T,
                                                  beam_splitter_above=True,
                                                  schema='est').to('uGal')

    assert np.all(fsof_above.value >= 0.0)
    assert np.all(fsof.value <= 0.0)

    np.testing.assert_array_almost_equal(fsof.value, -fsof_above.value)

    # test numerical value
    fsof_ref = -15.90  # uGal

    np.testing.assert_array_almost_equal(fsof.value, fsof_ref, decimal=2)
