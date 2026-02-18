import numpy as np

def dominance_index_copula(u, v):
    """
    Copula-space dominance index for a Kendall contour.

    Parameters
    ----------
    u, v : array-like
        Copula coordinates of the Kendall contour (same length, ordered).

    Returns
    -------
    D : float
        Dominance index:
        D > 0  -> U dominates
        D < 0  -> V dominates
        D = 0  -> symmetric dominance
    """
    u = np.asarray(u)
    v = np.asarray(v)

    # Arc-length weights
    du = np.diff(u)
    dv = np.diff(v)
    ds = np.sqrt(du**2 + dv**2)

    integrand = 0.5 * ((u[:-1] - v[:-1]) + (u[1:] - v[1:]))

    return np.sum(integrand * ds) / np.sum(ds)




def tail_dominance_ratio(u, v, q=0.95):
    """
    Tail dominance ratio using upper portion of Kendall contour.

    Parameters
    ----------
    u, v : array-like
        Copula coordinates of the Kendall contour.
    q : float
        Quantile threshold for defining the upper tail.

    Returns
    -------
    R : float
        Tail dominance ratio:
        R > 1  -> U dominates
        R < 1  -> V dominates
    """
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
    """
    Dominance index in physical space.

    Parameters
    ----------
    u, v : array-like
        Copula coordinates.
    Fx_inv, Fy_inv : callable
        Inverse CDFs for X and Y.

    Returns
    -------
    D_phys : float
        Physical-space dominance index.
    """
    x = Fx_inv.ppf(u)
    y = Fy_inv.ppf(v)

    dx = np.diff(x)
    dy = np.diff(y)
    ds = np.sqrt(dx**2 + dy**2)

    integrand = 0.5 * ((x[:-1] - y[:-1]) + (x[1:] - y[1:]))

    return np.sum(integrand * ds) / np.sum(ds)
