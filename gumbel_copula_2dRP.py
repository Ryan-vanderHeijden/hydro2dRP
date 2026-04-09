'''
Gumbel Copula Utilities for Multivariate Extreme Value Analysis
===============================================================

This module provides tools for working with the Gumbel copula, including:

- Copula CDF and PDF
- Joint return periods (AND, OR, conditional)
- Iso–return-period contours
- Kendall risk contours and bootstrap confidence bands
- Likelihood evaluation along copula contours
- Sampling from the Gumbel copula
- Helper utilities for distribution fitting and pseudo-observations

Notation
--------
theta : float
    Gumbel copula dependence parameter (theta >= 1).
u, v : array-like
    Pseudo-observations in [0, 1] (copula scale).
T : float
    Design return period (years).
fx, fy : scipy.stats frozen distribution objects
    Fitted marginal distributions.

References
----------
- Nelsen, R. B. (2006). An Introduction to Copulas.
- Salvadori et al. (2011). Multivariate Return Periods.
- Joe, H. (1997). Multivariate Models and Dependence Concepts.
'''

import numpy as np
import scipy as sp
from scipy.stats import levy_stable, uniform, norm, kendalltau, rankdata
from scipy.optimize import brentq




# ── Core Copula Functions ─────────────────────────────────────────────────────

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
    u = np.asarray(u)
    v = np.asarray(v)
    return np.exp(-((-np.log(u))**theta + (-np.log(v))**theta)**(1 / theta))




def gumbel_copula_pdf(u, v, theta):
    '''
    Probability density function of the Gumbel copula.

    Derived from the Gumbel generator phi(t) = (-ln t)^theta via
    the Archimedean copula PDF formula (Nelsen 2006).

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
    part2 = A**(1 / theta - 2)        # exponent is 1/theta - 2, not 2/theta - 2
    part3 = theta - 1 + A**(1 / theta)
    part4 = (logu * logv)**(theta - 1)

    return part1 * part2 * part3 * part4




# ── Return Period Functions ───────────────────────────────────────────────────

def return_period_AND(u, v, theta):
    '''
    Joint return period for the AND case: P(U > u AND V > v).

    Parameters
    ----------
    u, v : array-like
        Copula-space values.
    theta : float
        Copula parameter.

    Returns
    -------
    T_AND : array-like
    '''
    C = gumbel_copula(u, v, theta)
    return 1 / (1 - u - v + C)




def return_period_OR(u, v, theta):
    '''
    Joint return period for the OR case: P(U > u OR V > v).

    Parameters
    ----------
    u, v : array-like
        Copula-space values.
    theta : float
        Copula parameter.

    Returns
    -------
    T_OR : array-like
    '''
    C = gumbel_copula(u, v, theta)
    return 1 / (1 - C)




def return_period_conditional(u, v, theta):
    '''
    Conditional return period T(U > u | V > v).

    Parameters
    ----------
    u, v : array-like
        Copula-space values.
    theta : float
        Copula parameter.

    Returns
    -------
    T_cond : array-like
    '''
    C = gumbel_copula(u, v, theta)
    return (1 - v) / (1 - u - v + C)




# ── Iso-Contour Functions ─────────────────────────────────────────────────────

def iso_rp_AND(T, theta, u, v, n):
    '''
    AND-type iso–return-period contour on a meshgrid.

    Parameters
    ----------
    T : float
        Return period.
    theta : float
        Copula parameter.
    u, v : array-like
        1D grids for U and V axes.
    n : int
        Grid resolution (unused; for API symmetry).

    Returns
    -------
    U, V : 2D arrays
        Meshgrid in copula space.
    Z : 2D array
        Zero-contour (Z == 0) is the T-year AND contour.
    '''
    U, V = np.meshgrid(u, v)
    C = gumbel_copula(U, V, theta)
    Z = 1 - U - V + C
    return U, V, Z - 1 / T




def iso_rp_OR(u, T, theta):
    '''
    OR-type iso–return-period curve v(u).

    Solves C(u, v) = 1 - 1/T for v as a function of u.

    Parameters
    ----------
    u : array-like
        Grid of u values.
    T : float
        Return period.
    theta : float
        Copula parameter.

    Returns
    -------
    v : array-like
        Corresponding v-values on the OR contour (NaN outside valid range).
    '''
    A = (-np.log(1 - 1 / T))**theta
    term = A - (-np.log(u))**theta
    term[term < 0] = np.nan
    return np.exp(-term**(1 / theta))




def gumbel_contour(u, c0, theta):
    '''
    Compute v(u) along the copula level curve C(u, v) = c0.

    Parameters
    ----------
    u : array-like
        Grid of u values.
    c0 : float
        Target copula level.
    theta : float
        Copula parameter.

    Returns
    -------
    v : array-like
        Corresponding v-values (NaN outside valid range).
    '''
    A0 = (-np.log(c0))**theta
    B = (-np.log(u))**theta

    v = np.full_like(u, np.nan)
    valid = B < A0
    v[valid] = np.exp(-(A0 - B[valid])**(1 / theta))
    return v




def gumbel_kendall_isoline(u, c_T, theta):
    '''
    Alias for gumbel_contour. Computes v(u) on the copula level curve
    C(u, v) = c_T.

    When c_T = 1 - 1/T this corresponds to the OR return period contour.
    For the Kendall return period contour, use invert_kendall_level to
    obtain c_T from T first.

    Parameters
    ----------
    u : array-like
    c_T : float
        Target copula level.
    theta : float

    Returns
    -------
    v : array-like
    '''
    return gumbel_contour(u, c_T, theta)




# ── Density and Likelihood ────────────────────────────────────────────────────

def joint_density_OR(u, v, theta, fx, fy):
    '''
    Joint density of (X, Y) in physical space, evaluated along an
    OR return period contour.

    Uses Sklar's theorem: f(x,y) = c(F_X(x), F_Y(y)) * f_X(x) * f_Y(y),
    where x = F_X^{-1}(u) and y = F_Y^{-1}(v).

    Parameters
    ----------
    u, v : array-like
        Copula-space coordinates on the contour.
    theta : float
        Copula parameter.
    fx, fy : scipy.stats frozen distributions
        Fitted marginal distributions for X and Y.

    Returns
    -------
    f_xy : array-like
        Joint density at each (u, v) point on the contour.
    '''
    c_uv = gumbel_copula_pdf(u, v, theta)
    return c_uv * fx.pdf(fx.ppf(u)) * fy.pdf(fy.ppf(v))




def likelihood_along_contour(c0, u, theta):
    '''
    Evaluate copula density along a copula level curve C(u,v) = c0.

    Parameters
    ----------
    c0 : float
        Target copula level.
    u : array-like
        Grid of u values.
    theta : float
        Copula parameter.

    Returns
    -------
    u, v : array-like
        Valid points on the contour (NaN rows removed).
    c_uv : array-like
        Copula density values along the contour.
    '''
    v = gumbel_contour(u, c0, theta)
    mask = ~np.isnan(v)
    u, v = u[mask], v[mask]
    c_uv = gumbel_copula_pdf(u, v, theta)
    return u, v, c_uv




# ── Kendall Distribution ──────────────────────────────────────────────────────

def kendall_distribution_gumbel(t, theta):
    '''
    Kendall distribution K_C(t) for the Gumbel copula.

    For an Archimedean copula with generator phi, K(t) = t - phi(t)/phi'(t).
    For the Gumbel generator phi(t) = (-ln t)^theta this gives:
        K(t) = t * (1 - ln(t) / theta)

    Parameters
    ----------
    t : array-like
        Copula values in (0, 1).
    theta : float
        Copula parameter.

    Returns
    -------
    K : array-like
        Kendall CDF values.
    '''
    return t - (t / theta) * np.log(t)




def invert_kendall_level(K_target, theta):
    '''
    Find t in (0, 1) such that K_C(t) = K_target.

    Parameters
    ----------
    K_target : float
        Target Kendall CDF value.
    theta : float
        Copula parameter.

    Returns
    -------
    t : float
        Copula level corresponding to K_target.
    '''
    f = lambda t: kendall_distribution_gumbel(t, theta) - K_target
    return brentq(f, 1e-10, 1 - 1e-10)




def kendall_level_confidence_bounds(T, n_eff, alpha=0.05):
    '''
    Confidence interval for the Kendall probability K(t_T) = 1 - 1/T,
    using the effective sample size to account for dependence.

    Parameters
    ----------
    T : float
        Return period.
    n_eff : float
        Effective sample size.
    alpha : float
        Significance level (default 0.05 for 95% CI).

    Returns
    -------
    K_L, K_U : float
        Lower and upper confidence bounds on K(t_T).
    '''
    K_hat = 1.0 - 1.0 / T
    z = norm.ppf(1 - alpha / 2)
    var_K = (K_hat * (1 - K_hat)) / n_eff
    delta = z * np.sqrt(var_K)
    K_L = max(1e-6, K_hat - delta)
    K_U = min(1 - 1e-6, K_hat + delta)
    return K_L, K_U




def kendall_contour_bands(theta, T, n_eff, grid_size=200, alpha=0.05):
    '''
    Kendall return period contour with analytical confidence bands
    based on effective sample size.

    Parameters
    ----------
    theta : float
        Fitted copula parameter.
    T : float
        Return period.
    n_eff : float
        Effective sample size.
    grid_size : int
        Resolution of the copula-space grid.
    alpha : float
        Significance level.

    Returns
    -------
    U, V : 2D arrays
        Meshgrid in copula space.
    contours : dict
        Keys: 'central', 'lower', 'upper'.
        Zero-contour of each gives the corresponding band boundary.
    '''
    K_L, K_U = kendall_level_confidence_bounds(T, n_eff, alpha)

    t_c = invert_kendall_level(1 - 1 / T, theta)
    t_L = invert_kendall_level(K_L, theta)
    t_U = invert_kendall_level(K_U, theta)

    u = np.linspace(1e-4, 1 - 1e-4, grid_size)
    v = np.linspace(1e-4, 1 - 1e-4, grid_size)
    U, V = np.meshgrid(u, v)
    C = gumbel_copula(U, V, theta)

    return U, V, {
        'central': C - t_c,
        'lower':   C - t_L,
        'upper':   C - t_U,
    }




# ── Bootstrap Confidence Bands ────────────────────────────────────────────────

def fit_gumbel_theta(u, v):
    '''
    Fit Gumbel copula parameter theta using Kendall's tau inversion.

    The relationship tau = 1 - 1/theta gives theta = 1 / (1 - tau).

    Parameters
    ----------
    u, v : array-like
        Pseudo-observations in (0, 1).

    Returns
    -------
    theta : float
        Estimated copula parameter (>= 1).
    '''
    tau, _ = kendalltau(u, v)
    return max(1.0, 1.0 / (1.0 - tau))




def bootstrap_kendall_contours(u_reg, v_reg, T,
                                n_boot=500,
                                grid_size=200,
                                alpha=0.05,
                                random_state=None):
    '''
    Bootstrap Kendall contour confidence bands on a 2D copula-space grid.

    Resamples regional events, refits theta, and inverts the Kendall
    distribution analytically to get contour levels.

    Parameters
    ----------
    u_reg, v_reg : array-like
        Pseudo-observations of regional drought (duration, severity).
    T : float
        Return period.
    n_boot : int
        Bootstrap replicates.
    grid_size : int
        Copula-space grid resolution.
    alpha : float
        Significance level for confidence bands.
    random_state : int or None
        Random seed.

    Returns
    -------
    U, V : 2D arrays
        Meshgrid in copula space.
    contours : dict
        Keys: 'median', 'lower', 'upper'. Zero-contour of each gives the band.
    '''
    rng = np.random.default_rng(random_state)
    n_eff = len(u_reg)

    u_grid = np.linspace(1e-4, 1 - 1e-4, grid_size)
    v_grid = np.linspace(1e-4, 1 - 1e-4, grid_size)
    U, V = np.meshgrid(u_grid, v_grid)

    contour_stack = []

    for _ in range(n_boot):
        idx = rng.choice(n_eff, size=n_eff, replace=True)
        u_b = u_reg[idx]
        v_b = v_reg[idx]

        theta_b = fit_gumbel_theta(u_b, v_b)
        t_b = invert_kendall_level(1 - 1 / T, theta_b)
        C_b = gumbel_copula(U, V, theta_b)
        contour_stack.append(C_b - t_b)

    contour_stack = np.asarray(contour_stack)

    return U, V, {
        'median': np.quantile(contour_stack, 0.5,         axis=0),
        'lower':  np.quantile(contour_stack, alpha / 2,   axis=0),
        'upper':  np.quantile(contour_stack, 1 - alpha/2, axis=0),
    }




def bootstrap_kendall_contours_lines(
    u_reg,
    v_reg,
    T_K,
    n_boot=500,
    n_mc=200_000,
    u_min=1e-3,
    u_max=1 - 1e-3,
    n_points=200,
    alpha=0.05,
):
    '''
    Bootstrap Kendall contour confidence bands as (u, v) line arrays.

    Uses Monte Carlo estimation of the Kendall CDF rather than analytical
    inversion. Suitable for plotting contours directly as lines.

    Parameters
    ----------
    u_reg, v_reg : array-like
        Pseudo-observations of regional drought events.
    T_K : float
        Kendall return period.
    n_boot : int
        Bootstrap replicates.
    n_mc : int
        Monte Carlo samples for Kendall CDF estimation per replicate.
    u_min, u_max : float
        Range of u values for the output contour grid.
    n_points : int
        Number of points along each contour line.
    alpha : float
        Significance level for confidence bands.

    Returns
    -------
    result : dict
        Keys: 'u', 'v_median', 'v_lower', 'v_upper', 'all_contours'.
    '''
    u_grid = np.linspace(u_min, u_max, n_points)
    contours = []

    for _ in range(n_boot):
        idx = np.random.randint(0, len(u_reg), len(u_reg))
        u_b = u_reg[idx]
        v_b = v_reg[idx]

        theta_b = fit_gumbel_theta(u_b, v_b)

        # MC estimate of Kendall CDF
        u_mc = np.random.rand(n_mc)
        v_mc = np.random.rand(n_mc)
        C_mc = np.sort(gumbel_copula(u_mc, v_mc, theta_b))
        t_b = np.quantile(C_mc, 1 - 1 / T_K)

        v_curve = np.array([_invert_gumbel_v(u, t_b, theta_b) for u in u_grid])
        contours.append(v_curve)

    contours = np.array(contours)

    return {
        'u':            u_grid,
        'v_median':     np.nanmedian(contours, axis=0),
        'v_lower':      np.nanquantile(contours, alpha / 2, axis=0),
        'v_upper':      np.nanquantile(contours, 1 - alpha / 2, axis=0),
        'all_contours': contours,
    }




def central_kendall_contour(u_reg, v_reg, T, grid_size=200):
    '''
    Compute the central Kendall contour from data (no bootstrap).

    Parameters
    ----------
    u_reg, v_reg : array-like
        Pseudo-observations.
    T : float
        Return period.
    grid_size : int
        Copula-space grid resolution.

    Returns
    -------
    U, V : 2D arrays
        Meshgrid.
    Z : 2D array
        Zero-contour is the Kendall T-year return period contour.
    '''
    theta_hat = fit_gumbel_theta(u_reg, v_reg)
    t_hat = invert_kendall_level(1 - 1 / T, theta_hat)

    u = np.linspace(1e-4, 1 - 1e-4, grid_size)
    v = np.linspace(1e-4, 1 - 1e-4, grid_size)
    U, V = np.meshgrid(u, v)
    C = gumbel_copula(U, V, theta_hat)

    return U, V, C - t_hat




# ── Sampling ──────────────────────────────────────────────────────────────────

def sample_gumbel(n, theta):
    '''
    Sample from a bivariate Gumbel copula using the Marshall–Olkin algorithm.

    Parameters
    ----------
    n : int
        Number of samples.
    theta : float
        Copula parameter.

    Returns
    -------
    U, V : array-like, shape (n,)
        Copula samples in [0, 1].
    '''
    alpha = 1 / theta
    S = levy_stable.rvs(alpha, 1, size=n)
    E1 = -np.log(uniform.rvs(size=n))
    E2 = -np.log(uniform.rvs(size=n))

    U = np.exp(-(E1 / S)**alpha)
    V = np.exp(-(E2 / S)**alpha)

    return U, V




def pseudo_observations(x):
    '''
    Convert a data vector to pseudo-observations in (0, 1).

    Uses the scaled rank transform: r_i / (n + 1).

    Parameters
    ----------
    x : array-like
        Data vector.

    Returns
    -------
    u : array-like
        Pseudo-observations in (0, 1).
    '''
    r = rankdata(x, method='average')
    return r / (len(x) + 1)




# ── Distribution Fitting ──────────────────────────────────────────────────────

def best_fit_rv(data, dist_names, print_out=False):
    '''
    Fit candidate distributions to data and select the best by AIC.

    Parameters
    ----------
    data : array-like
        Observed values.
    dist_names : list of str
        scipy.stats distribution names to try (e.g. ['gamma', 'weibull_min']).
    print_out : bool
        If True, print AIC for each candidate.

    Returns
    -------
    best_dist : scipy.stats distribution class
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
            print(f'Distribution: {dist_name}, AIC: {aic:.2f}')

    return best_dist, best_params, best_aic




# ── Internal Helpers ──────────────────────────────────────────────────────────

def _invert_gumbel_v(u, t, theta):
    '''Solve C(u, v) = t for v, given scalar u.'''
    A = (-np.log(t))**theta - (-np.log(u))**theta
    if A <= 0:
        return np.nan
    return np.exp(-A**(1 / theta))
