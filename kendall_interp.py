'''
Kendall Contour Interpretation Utilities
=========================================

Tools for characterizing which variable (duration or severity) dominates
along a Kendall return period contour. Useful for understanding the
shape and asymmetry of bivariate drought risk.
'''

import numpy as np




def dominance_index_copula(u, v):
    '''
    Copula-space dominance index along a Kendall contour.

    Computes the arc-length-weighted mean of (u - v) along the contour.
    Positive values indicate that U (duration) tends to be larger than V
    (severity) at the same copula level; negative values the reverse.

    Parameters
    ----------
    u, v : array-like
        Copula coordinates of the Kendall contour, ordered along the curve.

    Returns
    -------
    D : float
        Dominance index.
        D > 0  -> U dominates
        D < 0  -> V dominates
        D = 0  -> symmetric
    '''
    u = np.asarray(u)
    v = np.asarray(v)

    du = np.diff(u)
    dv = np.diff(v)
    ds = np.sqrt(du**2 + dv**2)

    integrand = 0.5 * ((u[:-1] - v[:-1]) + (u[1:] - v[1:]))
    return np.sum(integrand * ds) / np.sum(ds)




def tail_dominance_ratio(u, v, q=0.95):
    '''
    Tail dominance ratio using the upper portion of a Kendall contour.

    Restricts to points where both u > q and v > q, then compares the
    spread of U and V in that tail region.

    Parameters
    ----------
    u, v : array-like
        Copula coordinates of the Kendall contour.
    q : float
        Quantile threshold defining the upper tail (default 0.95).

    Returns
    -------
    R : float
        Tail dominance ratio.
        R > 1  -> U dominates in the tail
        R < 1  -> V dominates in the tail
        NaN    -> fewer than 2 points in the tail region
    '''
    u = np.asarray(u)
    v = np.asarray(v)

    mask = (u > q) & (v > q)
    if np.sum(mask) < 2:
        return np.nan

    u_tail = u[mask]
    v_tail = v[mask]

    return (
        (u_tail.max() - np.median(u_tail)) /
        (v_tail.max() - np.median(v_tail))
    )




def dominance_physical(u, v, Fx_inv, Fy_inv):
    '''
    Dominance index computed in physical space (e.g., days vs. percentile-days).

    Applies the inverse CDFs to transform the contour from copula space
    to physical space before computing the arc-length-weighted mean difference.

    Parameters
    ----------
    u, v : array-like
        Copula coordinates of the Kendall contour.
    Fx_inv, Fy_inv : scipy.stats frozen distributions
        Fitted marginal distributions (must support .ppf()).

    Returns
    -------
    D_phys : float
        Physical-space dominance index (units depend on X and Y scales).
    '''
    x = Fx_inv.ppf(u)
    y = Fy_inv.ppf(v)

    dx = np.diff(x)
    dy = np.diff(y)
    ds = np.sqrt(dx**2 + dy**2)

    integrand = 0.5 * ((x[:-1] - y[:-1]) + (x[1:] - y[1:]))
    return np.sum(integrand * ds) / np.sum(ds)
