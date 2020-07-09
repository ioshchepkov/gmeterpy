# -*- coding: utf-8 -*-
"""Effective height calclulation for absolute gravimeters.

This module contatins functions for calculation the effective height, i.e. the
position in a free fall trajectory where the solution is independend from the
vertical gravity gradient.

"""

import gmeterpy.units as u


@u.quantity_input(initial_velocity=u.m / u.s, drop_duration=u.s)
def free_fall_effective_measurement_height(initial_velocity, drop_duration):
    """Effective measurements height of free-fall absolute gravimeters.

    This function calculates the distance from the starting point
    of data acquisition (t=0) in the free-fall experiment (drop) to the
    point where the solution of the free-fall equation is independent
    from the vertical gravity gradient.

    Parameters
    ----------
    initial_velocity : ~astropy.units.Quantity
        Initial velocity of the drop.
    drop_duration : ~astropy.units.Quantity
        Drop duration.

    Returns
    -------
    h_eff: ~astropy.units.Quantity
        The effective measurement height of free-fall.

    Notes
    -----
    This is an ananlytical determination based on the work of
    Ludger Timmen [1]_.

    Reference
    ---------
    .. [1] Timmen, L. (2003). Precise definition of the effective measurement
       height of free-fall absolute gravimeters. Metrologia, 40(2), 62.

    """

    g0 = 9.81 * u.m / u.s**2
    v0 = initial_velocity
    T = drop_duration

    v02 = v0**2
    v03 = v0**3
    g02 = g0**2
    g03 = g0**3
    T2 = T**2
    T3 = T**3

    h_eff_numerator = 56. * v0**4 * T + 102.4 * v03 * g0 * T2 +\
        61.2 * v02 * g02 * T3 + 14.0 * v0 * g03 * T**4 + g0**4 * T**5

    h_eff_denominator = 112. * v03 + 168. * v02 * g0 * T + 67.2 * v0 * g02 * T2 +\
        5.6 * g03 * T3

    h_eff = h_eff_numerator / h_eff_denominator

    return h_eff


@u.quantity_input(initial_velocity=u.m / u.s, gravity=u.m / u.s**2)
def tod_to_t0_distance(initial_velocity, gravity):
    r"""Distance to the starting position of the drop (z=0) from t=0.

    The solution of the free-fall equation is determined at the
    starting point of data acquisition and not at the "top of the drop" (TOD).
    This function will return the distance between these two points.

    Parameters
    ----------
    initial_velocity : ~astropy.units.Quantity
        Initial velocity of the drop.
    gravity : ~astropy.units.Quantity
        Gravity from the solution of free fall equation.

    Returns
    -------
    distance: ~astropy.units.Quantity
        The distance to the starting position of the drop from the starting
        position of data acquisition.

    Notes
    -----
    The distance :math:`s` between two points in uniform acceleration is

    .. math::
        \Delta z = \dfrac{v_0^2}{2g},

    where :math:`v_0` is the initial velocity at :math:`t = 0`, :math:`g` is
    the gravity at :math:`t = 0`.

    """

    distance = 0.5 * initial_velocity**2 / gravity

    return distance
