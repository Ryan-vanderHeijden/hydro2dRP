'''
ChatGPT was used to create uniform docstrings for the functions in this file.
'''

'''
Gumbel Copula Utilities for Multivariate Extreme Value Analysis
===============================================================

This module provides tools for working with the Gumbel copula, including:

- Copula PDF
- Kendall iso–return-period contours
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
- Salvadori et al. (2010, 2011). *Multivariate Return Periods*.
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





def gumbel_kendall_isoline(u, c_T, theta):
    A0 = (-np.log(c_T))**theta
    B = (-np.log(u))**theta

    v = np.full_like(u, np.nan)
    valid = B < A0
    v[valid] = np.exp(-(A0 - B[valid])**(1/theta))

    return v





def joint_density(u, v, theta, fx, fy):
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
