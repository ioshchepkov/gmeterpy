# -*- coding: utf-8 -*-
"""Correction due to the finite speed of light for absolute gravimeters.

This module contatins functions for calculation of the correction due to the
finite speed of light to the free-fall absolute gravity measurements.
It is also known as a Doppler-shift correction.

"""

import gmeterpy.units as u
import gmeterpy.constants as const


@u.quantity_input(gravity=u.m / u.s**2, initial_velocity=u.m / u.s,
                  drop_duration=u.s)
def finite_speed_of_light_correction(gravity, initial_velocity,
                                     drop_duration, beam_splitter_above=False,
                                     schema='est'):
    r"""Correction due to the finite speed of light.

    This is the correction due to the finite speed of light
    either for levels equally spaced in time (EST) or in
    distance (ESD).

    Parameters
    ----------
    gravity : ~astropy.units.Quantity
        Gravity from the solution of free fall equation.
    initial_velocity : ~astropy.units.Quantity
        Initial velocity of the drop.
    drop_duration : ~astropy.units.Quantity
        Drop duration.
    beam_splitter_above : bool, optional
        Indicates whether beam splitter positioned above
        (True) or below (False) the test body.
        Default is False.
    schema : {'est', 'esd'}, optional
        Controls what kind of schema is used in gravimeter.

        * 'est' for multi-levels equally spaced in time.
        * 'esd' for multi-levels equally spaced in distance.

        Default is `est`.

    Returns
    -------
    fsol: ~astropy.units.Quantity
        Correction due to the finite speed of light.

    Notes
    -----
    The correction due to the finite speed of light for levels equally spaced in
    time (EST) equals [1]_:

    .. math:: \Delta g = \pm\dfrac{3}{c} g_0 (v_0 + \dfrac{1}{2} g_0 * T)

    where :math:`c` is the speed of light, :math:`g_0,v_0` are the gravity
    value and initial velocity derived from the solution of free fall equation,
    :math:`T` is the drop duration. The upper or lower sign before the equation
    corresponds to the beam splitter positioned above or below the test body,
    respectively.

    For other schemas see the reference article [1]_.

    Reference
    ---------
    .. [1] Nagornyi VD, Zanimonskiy YM, Zanimonskiy YY (2011) Correction due to
       the finite speed of light in absolute gravimeters. Metrologia 48:101–113.
       doi: 10.1088/0026-1394/48/3/004

    """
    sign = 1 if beam_splitter_above else -1

    T = drop_duration
    g0 = gravity
    v0 = initial_velocity

    if schema == 'est':
        c1 = 0.5 * T
    elif schema == 'esd':
        c1_u = T * (4 * g0**3 * T**3 + 45 * g0**2 * T**2 * v0
                    + 108 * g0 * T * v0**2 + 70 * v0**3)
        c1_d = 7 * (g0**3 * T**3 + 12 * g0**2 * T**2 * v0
                    + 30 * g0 * T * v0**2 + 20 * v0**3)
        c1 = c1_u / c1_d
    else:
        raise ValueError('{} is not valid schema'.format(schema))

    fsol = sign * 3 * g0 / const.c * (v0 + g0 * c1)

    return fsol
