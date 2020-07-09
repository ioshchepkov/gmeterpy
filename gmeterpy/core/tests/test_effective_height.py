"""
Test finite speed of light corrections module.

"""

import pytest
import numpy as np

import gmeterpy.units as u

from gmeterpy.core.freefall.effective_height import free_fall_effective_measurement_height
from gmeterpy.core.freefall.effective_height import tod_to_t0_distance


def test_free_fall_effective_measurement_height():
    """Test finite speed of light est correction.

    """

    # g0 = 9.81570660 * u.m / u.s**2
    initial_velocity = 0.4412051 * u.m / u.s
    drop_duration = 0.2366589 * u.s

    h_eff_ref = 0.144878 * u.m

    h_eff = free_fall_effective_measurement_height(initial_velocity,
                                                   drop_duration)

    # test units
    assert isinstance(h_eff, u.Quantity)

    np.testing.assert_array_almost_equal(h_eff.value, h_eff_ref.value)


def test_tod_to_t0_distance():
    """Test TOD to t=0 distance calculation.

    """

    g0 = 9.81570660 * u.m / u.s**2
    initial_velocity = 0.4412051 * u.m / u.s

    h0_ref = 0.009916 * u.m

    h0 = tod_to_t0_distance(initial_velocity, g0)

    # test units
    assert isinstance(h0, u.Quantity)

    np.testing.assert_array_almost_equal(h0.value, h0_ref.value)
