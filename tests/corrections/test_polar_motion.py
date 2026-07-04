"""
Test atmospheric pressure correction.

"""

import numpy as np
from astropy.time import Time
from astropy.utils import iers

import gmeterpy.corrections.polar_motion as pm
import gmeterpy.units as u


def test_get_polar_motion(monkeypatch):
    """Test get_polar_motion function."""
    monkeypatch.setattr(iers.conf, "auto_download", False)

    # Bulletin B with status
    time = Time("2013-01-01")
    pm_xy = pm.get_polar_motion(time, return_status=True)
    assert len(pm_xy) == 3
    assert pm_xy[-1][0] == "IERS_B"
    assert np.isfinite(pm_xy[0].value)
    assert np.isfinite(pm_xy[1].value)

    # Bulletin B with no status
    pm_xy_without_status = pm.get_polar_motion(time, return_status=False)
    assert len(pm_xy_without_status) == 2
    np.testing.assert_array_equal(pm_xy_without_status[0], pm_xy[0])
    np.testing.assert_array_equal(pm_xy_without_status[1], pm_xy[1])

    # Out of range
    _, _, status = pm.get_polar_motion(Time("2100-01-01"), return_status=True)
    assert status[0] == "OUT_OF_RANGE"


def test_polar_motion_correction():
    """Test polar motion correction."""
    xp, yp = 0.1375 * u.arcsec, 0.3944 * u.arcsec
    lat, lon = 55.855 * u.deg, 37.516 * u.deg

    grav_corr = pm.polar_motion_correction(xp, yp, lat, lon)

    assert grav_corr.round(2) == 2.33 * u.uGal
