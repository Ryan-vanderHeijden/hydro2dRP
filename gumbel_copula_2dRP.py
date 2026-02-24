'''
ChatGPT was used to create uniform docstrings for the functions in this file.
'''

'''
Gumbel Copula Utilities for Multivariate Extreme Value Analysis
===============================================================

This module provides tools for working with the Gumbel copula, including:

- Copula PDF
- OR iso–return-period contours
- Joint density along OR contours
- Helper utilities for distribution fitting

Notation
--------
theta : float
    Gumbel copula dependence parameter (theta >= 1).
u, v : array-like
    Pseudo-observations in [0, 1] (copula scale).
T : float
    Design return period (years).
fx, fy : scipy.stats distribution objects
    Marginal distributions (PPF internally used).

References
----------
- Nelsen, R. B. (2006). *An Introduction to Copulas*.
- Salvadori et al. (2011). *Multivariate Return Periods*.
'''

import numpy as np
import scipy as sp



def gumbel_copula_pdf(u, v, theta):
    '''
    Probability density function of the Gumbel copula.

    Parameters
    ----------
    u, v : array-like
        Values in (0, 1).
    theta : float
        Dependence parameter (theta >= 1).

    Returns
    -------
    c_uv : array-like
        Copula density evaluated at (u, v).
    '''
    u = np.asarray(u)
    v = np.asarray(v)

    logu = -np.log(u)
    logv = -np.log(v)

    A = logu**theta + logv**theta
    C = np.exp(-A**(1 / theta))

    part1 = C / (u * v)
    part2 = A**(2 / theta - 2)
    part3 = theta - 1 + A**(1 / theta)
    part4 = (logu * logv)**(theta - 1)

    return part1 * part2 * part3 * part4




def iso_rp_OR(u, T, theta):
    '''
    OR-type iso–return-period curve v(u).

    Parameters
    ----------
    u : array-like
        Copula grid.
    T : float
        Return period.
    theta : float
        Copula parameter.

    Returns
    -------
    v : array-like
        Corresponding v-values on OR contour.
    '''
    A = (-np.log(1 - 1 / T))**theta
    term = A - (-np.log(u))**theta
    term[term < 0] = np.nan

    return np.exp(-(term)**(1 / theta))




def joint_density_OR(u, v, theta, fx, fy):
    '''
    Joint density under OR-conditioning.

    Returns
    -------
    f_xy : array-like
        Joint density in physical space.
    '''
    c_uv = gumbel_copula_pdf(u, v, theta)
    return c_uv * fx.ppf(u) * fy.ppf(v)




def best_fit_rv(data, dist_names, print_out=False):
    '''
    Fit candidate distributions and select best by AIC.

    Returns
    -------
    best_dist : scipy.stats distribution
    best_params : tuple
    best_aic : float
    '''
    best_aic = np.inf
    best_dist = None
    best_params = None

    for dist_name in dist_names:
        dist = getattr(sp.stats, dist_name)
        params = dist.fit(data)
        log_likelihood = np.sum(dist.logpdf(data, *params))
        k = len(params)
        aic = 2 * k - 2 * log_likelihood

        if aic < best_aic:
            best_aic = aic
            best_dist = dist
            best_params = params

        if print_out:
            print(f'Distribution: {best_dist}, AIC: {aic}')

    return best_dist, best_params, best_aic
