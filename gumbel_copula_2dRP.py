'''
Gumbel Copula Utilities for Multivariate Extreme Value Analysis
===============================================================

Backward-compatibility shim.  All implementation has moved to
``archimedean_copulas``, which adds Clayton and Frank support alongside
the generalized API.  This module re-exports the original Gumbel-specific
names and signatures so that existing notebooks and scripts continue to work
without changes.

Prefer importing from ``archimedean_copulas`` directly for new code.
'''

# ── Re-export self-contained functions ────────────────────────────────────────
from archimedean_copulas import (
    gumbel_copula,
    gumbel_copula_pdf,
    gumbel_contour,
    gumbel_kendall_isoline,
    kendall_distribution_gumbel,
    kendall_level_confidence_bounds,
    fit_gumbel_theta,
    sample_gumbel,
    pseudo_observations,
    best_fit_rv,
)

# ── Gumbel-specific wrappers for functions whose signatures now include family ─

import numpy as np
from archimedean_copulas import (
    invert_kendall_level  as _invert_kendall_level,
    return_period_AND     as _rp_AND,
    return_period_OR      as _rp_OR,
    return_period_conditional as _rp_cond,
    iso_rp_AND            as _iso_AND,
    iso_rp_OR             as _iso_OR,
    joint_density_OR      as _jd_OR,
    likelihood_along_contour as _lac,
    kendall_contour_bands as _kcb,
    bootstrap_kendall_contours  as _bkc,
    bootstrap_kendall_contours_lines as _bkcl,
    central_kendall_contour as _ckc,
)


def invert_kendall_level(K_target, theta):
    '''Find t in (0,1) such that K_C(t) = K_target for the Gumbel copula.'''
    return _invert_kendall_level(K_target, theta, 'gumbel')


def return_period_AND(u, v, theta):
    '''Joint AND return period for the Gumbel copula.'''
    return _rp_AND(u, v, theta, 'gumbel')


def return_period_OR(u, v, theta):
    '''Joint OR return period for the Gumbel copula.'''
    return _rp_OR(u, v, theta, 'gumbel')


def return_period_conditional(u, v, theta):
    '''Conditional return period T(U > u | V > v) for the Gumbel copula.'''
    return _rp_cond(u, v, theta, 'gumbel')


def iso_rp_AND(T, theta, u, v, n):
    '''AND-type iso-return-period contour meshgrid for the Gumbel copula.'''
    return _iso_AND(T, theta, u, v, n, family='gumbel')


def iso_rp_OR(u, T, theta):
    '''OR-type iso-return-period curve v(u) for the Gumbel copula.'''
    return _iso_OR(u, T, theta, 'gumbel')


def joint_density_OR(u, v, theta, fx, fy):
    '''Joint physical-space density on an OR contour for the Gumbel copula.'''
    return _jd_OR(u, v, theta, fx, fy, family='gumbel')


def likelihood_along_contour(c0, u, theta):
    '''Copula density along the level curve C(u,v)=c0 for the Gumbel copula.'''
    return _lac(c0, u, theta, 'gumbel')


def kendall_contour_bands(theta, T, n_eff, grid_size=200, alpha=0.05):
    '''Gumbel Kendall contour with analytical confidence bands.'''
    return _kcb(theta, T, n_eff, grid_size=grid_size, alpha=alpha, family='gumbel')


def bootstrap_kendall_contours(u_reg, v_reg, T,
                                n_boot=500, grid_size=200,
                                alpha=0.05, random_state=None):
    '''Bootstrap Kendall contour bands (grid form) for the Gumbel copula.'''
    return _bkc(u_reg, v_reg, T, family='gumbel',
                n_boot=n_boot, grid_size=grid_size,
                alpha=alpha, random_state=random_state)


def bootstrap_kendall_contours_lines(
    u_reg, v_reg, T_K,
    n_boot=500, n_mc=200_000,
    u_min=1e-3, u_max=1 - 1e-3,
    n_points=200, alpha=0.05,
):
    '''Bootstrap Kendall contour bands (line form) for the Gumbel copula.'''
    return _bkcl(u_reg, v_reg, T_K, family='gumbel',
                 n_boot=n_boot, n_mc=n_mc,
                 u_min=u_min, u_max=u_max,
                 n_points=n_points, alpha=alpha)


def central_kendall_contour(u_reg, v_reg, T, grid_size=200):
    '''Central Gumbel Kendall T-year contour from data (no bootstrap).'''
    return _ckc(u_reg, v_reg, T, grid_size=grid_size, family='gumbel')


# ── Internal helper (used by bootstrap_kendall_contours_lines above) ──────────

def _invert_gumbel_v(u, t, theta):
    '''Solve C(u, v) = t for v, given scalar u.'''
    A = (-np.log(t)) ** theta - (-np.log(u)) ** theta
    if A <= 0:
        return np.nan
    return np.exp(-A ** (1.0 / theta))
