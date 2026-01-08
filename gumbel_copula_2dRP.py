'''
ChatGPT was used to create uniform docstrings for the functions in this file.
'''

'''
Gumbel Copula Utilities for Multivariate Extreme Value Analysis
===============================================================

This module provides tools for working with the Gumbel copula, including:

- Copula CDF and PDF
- Joint return periods (AND, OR, conditional)
- Iso–return-period contours
- Kendall risk contours and bootstrap confidence bands
- Sampling from the Gumbel copula
- Likelihood evaluation along copula contours
- Helper utilities for colored plotting and distribution fitting

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
from scipy.stats import levy_stable, uniform



# Core Functions

def gumbel_copula(u, v, theta):
    '''
    Gumbel copula cumulative distribution function.

    Parameters
    ----------
    u, v : array-like
        Values in [0, 1].
    theta : float
        Dependence parameter (theta >= 1).

    Returns
    -------
    C : array-like
        Copula CDF evaluated at (u, v).
    '''
    return np.exp(-(((-np.log(u))**theta +
                      (-np.log(v))**theta)**(1 / theta)))




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




def gumbel_copula_cdf(u, v, theta):
    '''
    Vectorized Gumbel copula CDF with parameter validation.

    Parameters
    ----------
    u, v : array-like
        Values in [0, 1].
    theta : float
        Dependence parameter (theta >= 1).

    Returns
    -------
    C : array-like
        Copula CDF.
    '''
    if theta < 1:
        raise ValueError('theta must be >= 1')

    u = np.asarray(u)
    v = np.asarray(v)

    term = ((-np.log(u))**theta + (-np.log(v))**theta)**(1 / theta)
    return np.exp(-term)




# Return Period Functions
def return_period_AND(u, v, theta):
    '''
    Joint return period for the AND case (U > u AND V > v).

    Returns
    -------
    T_AND : array-like
        AND-type return period.
    '''
    C = gumbel_copula(u, v, theta)
    return 1 / (1 - u - v + C)




def return_period_OR(u, v, theta):
    '''
    Joint return period for the OR case (U > u OR V > v).

    Returns
    -------
    T_OR : array-like
        OR-type return period.
    '''
    C = gumbel_copula(u, v, theta)
    return 1 / (1 - C)




def return_period_conditional(u, v, theta):
    '''
    Conditional return period T(U > u | V > v).

    Returns
    -------
    T_cond : array-like
        Conditional return period.
    '''
    C = gumbel_copula(u, v, theta)
    return (1 - v) / (1 - u - v + C)




# Iso-line Return Period Functions
def iso_rp_AND(T, theta, u, v, n):
    '''
    AND-type iso–return-period contour.

    Returns
    -------
    U, V : 2D arrays
        Meshgrid of copula space.
    Z : 2D array
        Zero-contour corresponds to T-year return period.
    '''
    U, V = np.meshgrid(u, v)
    C = gumbel_copula(U, V, theta)
    Z = 1 - U - V + C

    return U, V, Z - 1 / T




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




# Density and Marginal Utilities
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




def best_fit_rv(data, dist_names, print=False):
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

        if print:
            print(f'Distribution: {dist_name}, AIC: {aic}')

        if aic < best_aic:
            best_aic = aic
            best_dist = dist
            best_params = params

    return best_dist, best_params, best_aic




# Kendall Risk Contours
def sample_gumbel(n, theta):
    '''
    Sample from a bivariate Gumbel copula using the
    Marshall–Olkin algorithm.

    Returns
    -------
    U, V : array-like
        Copula samples.
    '''
    alpha = 1 / theta
    S = levy_stable.rvs(alpha, 1, size=n)
    E1 = -np.log(uniform.rvs(size=n))
    E2 = -np.log(uniform.rvs(size=n))

    U = np.exp(-(E1 / S)**alpha)
    V = np.exp(-(E2 / S)**alpha)

    return U, V




def bootstrap_kendall_levels(C_sim, c_hat, return_periods, B=500, alpha=0.05):
    '''
    Bootstrap confidence intervals for Kendall copula levels.

    Returns
    -------
    lower, center, upper : arrays
        Confidence bands.
    '''
    n = len(C_sim)
    q = 1 - 1 / return_periods
    c_boot = np.empty((B, len(return_periods)))

    for b in range(B):
        C_star = np.random.choice(C_sim, size=n, replace=True)
        c_boot[b] = np.quantile(C_star, q)

    lower = np.quantile(c_boot, alpha / 2, axis=0)
    upper = np.quantile(c_boot, 1 - alpha / 2, axis=0)

    return lower, c_hat, upper




def kendall_level(C_sorted, T):
    '''
    Kendall copula level corresponding to return period T.
    '''
    q = 1 - 1 / T
    return np.quantile(C_sorted, q)




# Gumbel Contours and Likelihood
def gumbel_contour(u, c0, theta):
    '''
    Compute v(u) for a fixed copula level c0.
    '''
    A0 = (-np.log(c0))**theta
    B = (-np.log(u))**theta

    v = np.full_like(u, np.nan)
    valid = B < A0
    v[valid] = np.exp(-(A0 - B[valid])**(1 / theta))

    return v




def likelihood_along_contour(c0, u, theta):
    '''
    Evaluate copula density along a Kendall contour.

    Returns
    -------
    u, v : array-like
    c_uv : array-like
        Copula density values.
    '''
    v = gumbel_contour(u, c0, theta)
    mask = ~np.isnan(v)

    u, v = u[mask], v[mask]
    copula_density = gumbel_copula_pdf(u, v, theta)

    return u, v, copula_density




