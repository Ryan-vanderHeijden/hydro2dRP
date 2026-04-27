'''
Archimedean Copula Utilities: Gumbel, Clayton, and Frank
=========================================================

Self-contained module for bivariate Archimedean copula analysis.
Supersedes ``gumbel_copula_2dRP`` (which is now a compatibility shim).

Supported families
------------------
Gumbel:  C = exp(-((-ln u)^θ + (-ln v)^θ)^{1/θ}),  θ ≥ 1
Clayton: C = max(u^{-θ} + v^{-θ} - 1, 0)^{-1/θ},  θ > 0
Frank:   C = -1/θ ln(1 + (e^{-θu}-1)(e^{-θv}-1)/(e^{-θ}-1)),  θ ≠ 0

Notation
--------
theta : float
    Copula dependence parameter (family-specific bounds above).
    θ → lower bound gives independence; θ → ∞ gives comonotonicity.
u, v : array-like
    Pseudo-observations in (0, 1) (copula scale).
family : str
    One of ``'gumbel'``, ``'clayton'``, ``'frank'``.
T : float
    Design return period (years).

References
----------
- Nelsen, R. B. (2006). An Introduction to Copulas.
- Salvadori et al. (2011). Multivariate Return Periods.
- Joe, H. (1997). Multivariate Models and Dependence Concepts.
- Genest & MacKay (1986). The joy of copulas.
- Aas et al. (2009). Pair-copula constructions of multiple dependence.
'''

import numpy as np
import scipy as sp
from scipy.stats import levy_stable, uniform, norm, kendalltau, rankdata
from scipy.optimize import brentq, minimize_scalar
from scipy.integrate import quad


_FAMILIES = ('gumbel', 'clayton', 'frank')


# ── Gumbel Copula ──────────────────────────────────────────────────────────────

def gumbel_copula(u, v, theta):
    '''
    Gumbel copula CDF.

    C(u, v) = exp(-((-ln u)^theta + (-ln v)^theta)^{1/theta})

    Parameters
    ----------
    u, v : array-like  Values in (0, 1).
    theta : float      Dependence parameter (theta >= 1).

    Returns
    -------
    C : array-like
    '''
    u = np.asarray(u)
    v = np.asarray(v)
    return np.exp(-((-np.log(u)) ** theta + (-np.log(v)) ** theta) ** (1.0 / theta))


def gumbel_copula_pdf(u, v, theta):
    '''
    Gumbel copula PDF.

    Derived from the generator phi(t) = (-ln t)^theta via the Archimedean
    copula density formula (Nelsen 2006).

    Parameters
    ----------
    u, v : array-like  Values in (0, 1).
    theta : float      Dependence parameter (theta >= 1).

    Returns
    -------
    c_uv : array-like
    '''
    u = np.asarray(u)
    v = np.asarray(v)
    logu = -np.log(u)
    logv = -np.log(v)
    A = logu ** theta + logv ** theta
    C = np.exp(-A ** (1.0 / theta))
    return (C / (u * v)
            * A ** (1.0 / theta - 2.0)
            * (theta - 1.0 + A ** (1.0 / theta))
            * (logu * logv) ** (theta - 1.0))


def kendall_distribution_gumbel(t, theta):
    '''
    Kendall distribution K_C(t) for the Gumbel copula.

    K(t) = t * (1 - ln(t) / theta)

    Parameters
    ----------
    t : array-like  Copula values in (0, 1).
    theta : float

    Returns
    -------
    K : array-like
    '''
    return t - (t / theta) * np.log(t)


# ── Clayton Copula ─────────────────────────────────────────────────────────────

def clayton_copula(u, v, theta):
    '''
    Clayton copula CDF.

    C(u, v) = max(u^{-theta} + v^{-theta} - 1, 0)^{-1/theta}

    Parameters
    ----------
    u, v : array-like  Values in (0, 1).
    theta : float      Dependence parameter (theta > 0).

    Returns
    -------
    C : array-like
    '''
    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)
    A = u ** (-theta) + v ** (-theta) - 1.0
    return np.where(A > 0, A ** (-1.0 / theta), 0.0)


def clayton_copula_pdf(u, v, theta):
    '''
    Clayton copula PDF.

    c(u, v) = (1+theta) * (uv)^{-1-theta} * (u^{-theta}+v^{-theta}-1)^{-2-1/theta}

    Parameters
    ----------
    u, v : array-like  Values in (0, 1).
    theta : float

    Returns
    -------
    c_uv : array-like
    '''
    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)
    A = u ** (-theta) + v ** (-theta) - 1.0
    return (1.0 + theta) * (u * v) ** (-1.0 - theta) * A ** (-2.0 - 1.0 / theta)


def kendall_distribution_clayton(t, theta):
    '''
    Kendall distribution K_C(t) for the Clayton copula.

    K(t) = [t*(theta+1) - t^{theta+1}] / theta

    Parameters
    ----------
    t : array-like  Copula values in (0, 1).
    theta : float

    Returns
    -------
    K : array-like
    '''
    t = np.asarray(t, dtype=float)
    return (t * (theta + 1.0) - t ** (theta + 1.0)) / theta


# ── Frank Copula ───────────────────────────────────────────────────────────────

def frank_copula(u, v, theta):
    '''
    Frank copula CDF.

    C(u, v) = -1/theta * ln(1 + (e^{-theta*u}-1)(e^{-theta*v}-1)/(e^{-theta}-1))

    Parameters
    ----------
    u, v : array-like  Values in (0, 1).
    theta : float      Dependence parameter (theta != 0).

    Returns
    -------
    C : array-like
    '''
    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)
    D = np.expm1(-theta)           # e^{-theta} - 1
    a = np.expm1(-theta * u)       # e^{-theta*u} - 1
    b = np.expm1(-theta * v)       # e^{-theta*v} - 1
    return -np.log1p(a * b / D) / theta


def frank_copula_pdf(u, v, theta):
    '''
    Frank copula PDF.

    c(u, v) = -theta * D * e^{-theta*(u+v)} / (D + ab)^2
    where D = e^{-theta}-1, a = e^{-theta*u}-1, b = e^{-theta*v}-1.

    Parameters
    ----------
    u, v : array-like  Values in (0, 1).
    theta : float

    Returns
    -------
    c_uv : array-like
    '''
    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)
    D = np.expm1(-theta)
    a = np.expm1(-theta * u)
    b = np.expm1(-theta * v)
    return -theta * D * np.exp(-theta * (u + v)) / (D + a * b) ** 2


def kendall_distribution_frank(t, theta):
    '''
    Kendall distribution K_C(t) for the Frank copula.

    K(t) = t + ln[(e^{-theta*t}-1)/(e^{-theta}-1)] * (e^{-theta*t}-1) / (theta*e^{-theta*t})

    Parameters
    ----------
    t : array-like  Copula values in (0, 1).
    theta : float

    Returns
    -------
    K : array-like
    '''
    t = np.asarray(t, dtype=float)
    D = np.expm1(-theta)
    a = np.expm1(-theta * t)
    with np.errstate(divide='ignore', invalid='ignore'):
        log_ratio = np.log(a / D)
    K = t + log_ratio * a / (theta * np.exp(-theta * t))
    K = np.where(t <= 0.0, 0.0, K)
    K = np.where(t >= 1.0, 1.0, K)
    return K


# ── h-functions (conditional CDFs) ────────────────────────────────────────────

def h_gumbel(u, v, theta):
    '''
    Gumbel h-function: h(u | v) = dC(u, v) / dv.

    h = C(u,v) * A^{1/theta - 1} * (-log v)^{theta-1} / v
    where A = (-log u)^theta + (-log v)^theta.
    '''
    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)
    logu = -np.log(u)
    logv = -np.log(v)
    A = logu ** theta + logv ** theta
    C = np.exp(-A ** (1.0 / theta))
    return C * A ** (1.0 / theta - 1.0) * logv ** (theta - 1.0) / v


def h_inv_gumbel(w, v, theta, tol=1e-12):
    '''
    Inverse Gumbel h-function. Solved numerically via brentq.

    Parameters
    ----------
    w : array-like  Target probability in (0, 1).
    v : array-like or float  Conditioning value.
    theta : float
    tol : float

    Returns
    -------
    u : array-like
    '''
    w = np.asarray(w, dtype=float)
    v = np.asarray(v, dtype=float)
    scalar = w.ndim == 0 and v.ndim == 0
    w = np.atleast_1d(w)
    v = np.atleast_1d(v) if v.ndim > 0 else np.full_like(w, float(v))

    result = np.empty_like(w)
    for i in range(len(w)):
        vi = v[i] if v.size > 1 else v[0]
        def obj(u_): return h_gumbel(u_, vi, theta) - w[i]
        try:
            result[i] = brentq(obj, 1e-10, 1 - 1e-10, xtol=tol)
        except ValueError:
            result[i] = np.nan

    return float(result[0]) if scalar else result


def h_clayton(u, v, theta):
    '''
    Clayton h-function: h(u | v) = dC(u, v) / dv.

    h = v^{-theta-1} * (u^{-theta}+v^{-theta}-1)^{-1-1/theta}
    '''
    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)
    A = u ** (-theta) + v ** (-theta) - 1.0
    return v ** (-theta - 1.0) * A ** (-1.0 - 1.0 / theta)


def h_inv_clayton(w, v, theta):
    '''
    Inverse Clayton h-function. Analytical.

    u = [(w * v^{theta+1})^{-theta/(1+theta)} - v^{-theta} + 1]^{-1/theta}
    '''
    w = np.asarray(w, dtype=float)
    v = np.asarray(v, dtype=float)
    A = (w * v ** (theta + 1.0)) ** (-theta / (1.0 + theta)) - v ** (-theta) + 1.0
    return np.clip(A ** (-1.0 / theta), 1e-10, 1 - 1e-10)


def h_frank(u, v, theta):
    '''
    Frank h-function: h(u | v) = dC(u, v) / dv.

    h = e^{-theta*v} * (e^{-theta*u}-1) / [D + (e^{-theta*u}-1)(e^{-theta*v}-1)]
    '''
    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)
    D = np.expm1(-theta)
    a = np.expm1(-theta * u)
    b = np.expm1(-theta * v)
    return np.exp(-theta * v) * a / (D + a * b)


def h_inv_frank(w, v, theta):
    '''
    Inverse Frank h-function. Analytical.

    u = -1/theta * ln(1 + w*(e^{-theta}-1) / [e^{-theta*v}*(1-w) + w])
    '''
    w = np.asarray(w, dtype=float)
    v = np.asarray(v, dtype=float)
    D = np.expm1(-theta)
    denom = np.exp(-theta * v) * (1.0 - w) + w
    return np.clip(-np.log1p(w * D / denom) / theta, 1e-10, 1 - 1e-10)


# ── Dispatch Functions ─────────────────────────────────────────────────────────

def copula_cdf(u, v, theta, family):
    '''
    Copula CDF dispatcher.

    Parameters
    ----------
    u, v : array-like
    theta : float
    family : {'gumbel', 'clayton', 'frank'}
    '''
    if family == 'gumbel':   return gumbel_copula(u, v, theta)
    if family == 'clayton':  return clayton_copula(u, v, theta)
    if family == 'frank':    return frank_copula(u, v, theta)
    raise ValueError(f"family must be one of {_FAMILIES}")


def copula_pdf(u, v, theta, family):
    '''
    Copula PDF dispatcher.

    Parameters
    ----------
    u, v : array-like
    theta : float
    family : {'gumbel', 'clayton', 'frank'}
    '''
    if family == 'gumbel':   return gumbel_copula_pdf(u, v, theta)
    if family == 'clayton':  return clayton_copula_pdf(u, v, theta)
    if family == 'frank':    return frank_copula_pdf(u, v, theta)
    raise ValueError(f"family must be one of {_FAMILIES}")


def h_func(u, v, theta, family):
    '''
    h-function dispatcher: h(u | v) = dC(u, v) / dv.

    Returns the conditional CDF of U given V = v.

    Parameters
    ----------
    u : array-like  Target variable.
    v : array-like  Conditioning variable.
    theta : float
    family : {'gumbel', 'clayton', 'frank'}
    '''
    if family == 'gumbel':   return h_gumbel(u, v, theta)
    if family == 'clayton':  return h_clayton(u, v, theta)
    if family == 'frank':    return h_frank(u, v, theta)
    raise ValueError(f"family must be one of {_FAMILIES}")


def h_func_inv(w, v, theta, family):
    '''
    Inverse h-function dispatcher. Solves h(u | v; theta) = w for u.

    Parameters
    ----------
    w : array-like  Target probability in (0, 1).
    v : array-like  Conditioning value in (0, 1).
    theta : float
    family : {'gumbel', 'clayton', 'frank'}
    '''
    if family == 'gumbel':   return h_inv_gumbel(w, v, theta)
    if family == 'clayton':  return h_inv_clayton(w, v, theta)
    if family == 'frank':    return h_inv_frank(w, v, theta)
    raise ValueError(f"family must be one of {_FAMILIES}")


# ── Kendall Distributions ──────────────────────────────────────────────────────

def kendall_distribution(t, theta, family):
    '''
    Kendall distribution K_C(t) dispatcher.

    Parameters
    ----------
    t : array-like
    theta : float
    family : {'gumbel', 'clayton', 'frank'}
    '''
    if family == 'gumbel':   return kendall_distribution_gumbel(t, theta)
    if family == 'clayton':  return kendall_distribution_clayton(t, theta)
    if family == 'frank':    return kendall_distribution_frank(t, theta)
    raise ValueError(f"family must be one of {_FAMILIES}")


def invert_kendall_level(K_target, theta, family=None):
    '''
    Find t in (0, 1) such that K_C(t; theta, family) = K_target.

    Parameters
    ----------
    K_target : float
    theta : float
    family : {'gumbel', 'clayton', 'frank'}
        Defaults to 'gumbel' when omitted (backward-compatible with
        the old gumbel_copula_2dRP signature).
    '''
    if family is None:
        family = 'gumbel'
    f = lambda t: kendall_distribution(t, theta, family) - K_target
    return brentq(f, 1e-10, 1 - 1e-10)


# ── Parameter Estimation ───────────────────────────────────────────────────────

def _debye1(theta):
    '''First-order Debye function D_1(theta) = (1/theta)*∫_0^theta t/(e^t-1) dt.'''
    if abs(theta) < 1e-10:
        return 1.0
    integrand = lambda t: t / np.expm1(t) if t > 1e-10 else 1.0 - t / 2.0
    val, _ = quad(integrand, 0.0, abs(theta), limit=100)
    return val / abs(theta)


def tau_to_theta(tau, family):
    '''
    Convert Kendall's tau to copula parameter via method of moments.

    Gumbel:  theta = 1/(1-tau)          [tau ∈ [0, 1)]
    Clayton: theta = 2*tau/(1-tau)       [tau ∈ (0, 1)]
    Frank:   numerical inversion         [tau ∈ (0, 1)]
    '''
    if family == 'gumbel':
        return max(1.0, 1.0 / (1.0 - tau))
    if family == 'clayton':
        return max(1e-6, 2.0 * tau / (1.0 - tau))
    if family == 'frank':
        if abs(tau) < 1e-6:
            return 1e-6
        f = lambda th: 1.0 + 4.0 * (_debye1(th) - 1.0) / th - tau
        try:
            return brentq(f, 1e-6, 100.0)
        except ValueError:
            return 1e-6
    raise ValueError(f"family must be one of {_FAMILIES}")


def fit_copula_mle(u, v, family):
    '''
    Fit copula parameter theta by maximum likelihood (bounded scalar optimisation).

    Parameters
    ----------
    u, v : array-like   Pseudo-observations in (0, 1).
    family : str

    Returns
    -------
    theta : float
    log_likelihood : float
    '''
    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)

    _bounds = {'gumbel': (1.0 + 1e-6, 30.0),
               'clayton': (1e-6, 30.0),
               'frank':   (1e-6, 50.0)}

    def neg_loglik(theta):
        vals = np.clip(copula_pdf(u, v, theta, family), 1e-300, None)
        ll   = np.sum(np.log(vals))
        return -ll if np.isfinite(ll) else 1e20

    lb, ub = _bounds[family]
    res = minimize_scalar(neg_loglik, bounds=(lb, ub), method='bounded')
    return res.x, -res.fun


def fit_copula(u, v, family, method='mle'):
    '''
    Fit a single Archimedean copula family to pseudo-observations.

    Parameters
    ----------
    u, v : array-like
    family : {'gumbel', 'clayton', 'frank'}
    method : {'mle', 'tau'}
        'mle' — maximum likelihood (default).
        'tau' — method of moments via Kendall's tau inversion (faster).

    Returns
    -------
    theta : float
    log_likelihood : float
    '''
    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)

    if method == 'tau':
        tau, _ = kendalltau(u, v)
        theta  = tau_to_theta(tau, family)
        ll     = float(np.sum(np.log(np.clip(copula_pdf(u, v, theta, family), 1e-300, None))))
        return theta, ll

    return fit_copula_mle(u, v, family)


def fit_gumbel_theta(u, v):
    '''
    Fit Gumbel theta via Kendall's tau inversion: theta = 1/(1-tau).

    Retained for backward compatibility with gumbel_copula_2dRP.
    Equivalent to ``fit_copula(u, v, 'gumbel', method='tau')[0]``.
    '''
    tau, _ = kendalltau(u, v)
    return max(1.0, 1.0 / (1.0 - tau))


# ── Model Selection ────────────────────────────────────────────────────────────

def select_copula(u, v, families=None, criterion='aicc', method='mle'):
    '''
    Fit candidate families and select the best by AICc (or AIC / BIC).

    Each Archimedean family has k=1 free parameter.

    Parameters
    ----------
    u, v : array-like
    families : list of str, optional   Defaults to all three families.
    criterion : {'aicc', 'aic', 'bic'}
    method : {'mle', 'tau'}

    Returns
    -------
    results : list of dict
        Sorted best → worst by criterion.  Keys: family, theta,
        log_likelihood, aic, aicc, bic.
    best : dict
    '''
    if families is None:
        families = list(_FAMILIES)

    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)
    n, k = len(u), 1

    results = []
    for fam in families:
        theta, ll = fit_copula(u, v, fam, method=method)
        aic  = 2.0 * k - 2.0 * ll
        aicc = aic + 2.0 * k * (k + 1.0) / max(n - k - 1, 1)
        bic  = k * np.log(n) - 2.0 * ll
        results.append({'family': fam, 'theta': theta,
                        'log_likelihood': ll, 'aic': aic, 'aicc': aicc, 'bic': bic})

    results.sort(key=lambda r: r[criterion])
    return results, results[0]


# ── Return Period Functions ────────────────────────────────────────────────────

def return_period_AND(u, v, theta, family='gumbel'):
    '''
    Joint AND return period: T_AND = 1 / P(U > u, V > v).

    Parameters
    ----------
    u, v : array-like
    theta : float
    family : str  (default 'gumbel' for backward compatibility)
    '''
    C = copula_cdf(u, v, theta, family)
    return 1.0 / (1.0 - u - v + C)


def return_period_OR(u, v, theta, family='gumbel'):
    '''
    Joint OR return period: T_OR = 1 / P(U > u OR V > v).

    Parameters
    ----------
    u, v : array-like
    theta : float
    family : str  (default 'gumbel' for backward compatibility)
    '''
    C = copula_cdf(u, v, theta, family)
    return 1.0 / (1.0 - C)


def return_period_conditional(u, v, theta, family='gumbel'):
    '''
    Conditional return period T(U > u | V > v).

    T_cond = (1 - v) / P(U > u, V > v) = (1 - v) / (1 - u - v + C)

    Parameters
    ----------
    u, v : array-like
    theta : float
    family : str  (default 'gumbel' for backward compatibility)
    '''
    C = copula_cdf(u, v, theta, family)
    return (1.0 - v) / (1.0 - u - v + C)


# ── Contour Functions ─────────────────────────────────────────────────────────

def copula_contour(u, c0, theta, family):
    '''
    Compute v(u) along the copula level curve C(u, v) = c0.

    Returns NaN outside the range where a solution exists.

    Parameters
    ----------
    u : array-like  Grid of u values.
    c0 : float      Target copula level.
    theta : float
    family : str
    '''
    u = np.asarray(u, dtype=float)

    if family == 'gumbel':
        A0 = (-np.log(c0)) ** theta
        B  = (-np.log(u))  ** theta
        v  = np.full_like(u, np.nan)
        valid = B < A0
        v[valid] = np.exp(-(A0 - B[valid]) ** (1.0 / theta))
        return v

    if family == 'clayton':
        A = c0 ** (-theta) - u ** (-theta) + 1.0
        return np.where(A > 0, A ** (-1.0 / theta), np.nan)

    if family == 'frank':
        D     = np.expm1(-theta)
        numer = D * np.expm1(-theta * c0)
        denom = np.expm1(-theta * u)
        arg   = 1.0 + numer / denom
        return np.where(arg > 0, -np.log(arg) / theta, np.nan)

    raise ValueError(f"family must be one of {_FAMILIES}")


# Backward-compatible aliases
def gumbel_contour(u, c0, theta):
    '''Alias for copula_contour(..., family='gumbel').'''
    return copula_contour(u, c0, theta, 'gumbel')


def gumbel_kendall_isoline(u, c_T, theta):
    '''Alias for copula_contour(..., family='gumbel').'''
    return copula_contour(u, c_T, theta, 'gumbel')


def kendall_isoline(u, c_T, theta, family):
    '''
    Compute v(u) on the copula level curve C(u, v) = c_T.

    Use ``invert_kendall_level(1 - 1/T, theta, family)`` to obtain c_T first.
    '''
    return copula_contour(u, c_T, theta, family)


def iso_rp_AND(T, theta, u, v, n=None, family='gumbel'):
    '''
    AND-type iso-return-period contour on a meshgrid.

    Parameters
    ----------
    T : float
    theta : float
    u, v : array-like   1D grid arrays.
    n : int             Unused (kept for API symmetry with old module).
    family : str        Default 'gumbel'.

    Returns
    -------
    U, V : 2D arrays
    Z : 2D array   Zero-contour (Z==0) is the T-year AND contour.
    '''
    U, V = np.meshgrid(u, v)
    C    = copula_cdf(U, V, theta, family)
    return U, V, 1.0 - U - V + C - 1.0 / T


def iso_rp_OR(u, T, theta, family='gumbel'):
    '''
    OR-type iso-return-period curve v(u).

    Solves C(u, v) = 1 - 1/T for v as a function of u.

    Parameters
    ----------
    u : array-like
    T : float
    theta : float
    family : str   Default 'gumbel'.

    Returns
    -------
    v : array-like
    '''
    return copula_contour(np.asarray(u, dtype=float), 1.0 - 1.0 / T, theta, family)


# ── Density and Likelihood ────────────────────────────────────────────────────

def joint_density_OR(u, v, theta, fx, fy, family='gumbel'):
    '''
    Joint physical-space density at copula-space coordinates (u, v) on an
    OR contour, via Sklar's theorem.

    f(x,y) = c(u, v) * f_X(F_X^{-1}(u)) * f_Y(F_Y^{-1}(v))

    Parameters
    ----------
    u, v : array-like   Copula coordinates already on the contour.
    theta : float
    fx, fy : frozen scipy.stats distributions
    family : str        Default 'gumbel'.
    '''
    c_uv = copula_pdf(u, v, theta, family)
    return c_uv * fx.pdf(fx.ppf(u)) * fy.pdf(fy.ppf(v))


def likelihood_along_contour(c0, u, theta, family='gumbel'):
    '''
    Evaluate copula density along the level curve C(u, v) = c0.

    Parameters
    ----------
    c0 : float
    u : array-like
    theta : float
    family : str   Default 'gumbel'.

    Returns
    -------
    u, v : array-like   Valid contour points (NaN rows removed).
    c_uv : array-like   Copula density along the contour.
    '''
    v    = copula_contour(u, c0, theta, family)
    mask = ~np.isnan(v)
    u    = u[mask]
    v    = v[mask]
    c_uv = copula_pdf(u, v, theta, family)
    return u, v, c_uv


def joint_density_on_contour(c0, u, theta, family, fx, fy):
    '''
    Joint physical-space density along the copula level curve C(u, v) = c0.

    Parameters
    ----------
    c0 : float
    u : array-like
    theta : float
    family : str
    fx, fy : frozen scipy.stats distributions

    Returns
    -------
    u, v : array-like   Valid points.
    f_xy : array-like   Joint density.
    '''
    u, v, c_uv = likelihood_along_contour(c0, u, theta, family)
    return u, v, c_uv * fx.pdf(fx.ppf(u)) * fy.pdf(fy.ppf(v))


# ── Uncertainty Quantification ─────────────────────────────────────────────────

def kendall_level_confidence_bounds(T, n_eff, alpha=0.05):
    '''
    Normal-approximation confidence interval for K(t_T) = 1 - 1/T.

    Parameters
    ----------
    T : float       Return period.
    n_eff : float   Effective sample size.
    alpha : float   Significance level (default 0.05 → 95% CI).

    Returns
    -------
    K_L, K_U : float
    '''
    K_hat = 1.0 - 1.0 / T
    z     = norm.ppf(1.0 - alpha / 2.0)
    delta = z * np.sqrt(K_hat * (1.0 - K_hat) / n_eff)
    return max(1e-6, K_hat - delta), min(1.0 - 1e-6, K_hat + delta)


def kendall_contour_bands(theta, T, n_eff, grid_size=200, alpha=0.05, family='gumbel'):
    '''
    Kendall return period contour with analytical confidence bands.

    Parameters
    ----------
    theta : float
    T : float
    n_eff : float
    grid_size : int
    alpha : float
    family : str   Default 'gumbel'.

    Returns
    -------
    U, V : 2D arrays
    contours : dict with keys 'central', 'lower', 'upper'.
    '''
    K_L, K_U = kendall_level_confidence_bounds(T, n_eff, alpha)
    t_c = invert_kendall_level(1.0 - 1.0 / T, theta, family)
    t_L = invert_kendall_level(K_L, theta, family)
    t_U = invert_kendall_level(K_U, theta, family)

    u = np.linspace(1e-4, 1 - 1e-4, grid_size)
    v = np.linspace(1e-4, 1 - 1e-4, grid_size)
    U, V = np.meshgrid(u, v)
    C    = copula_cdf(U, V, theta, family)

    return U, V, {'central': C - t_c, 'lower': C - t_L, 'upper': C - t_U}


def kendall_contour_grid(theta, T, family, grid_size=200):
    '''
    Kendall T-year contour as a meshgrid zero-surface.

    Parameters
    ----------
    theta : float
    T : float
    family : str
    grid_size : int

    Returns
    -------
    U, V : 2D arrays
    Z : 2D array   Zero-contour gives the Kendall T-year contour.
    '''
    t_c = invert_kendall_level(1.0 - 1.0 / T, theta, family)
    u   = np.linspace(1e-4, 1 - 1e-4, grid_size)
    v   = np.linspace(1e-4, 1 - 1e-4, grid_size)
    U, V = np.meshgrid(u, v)
    return U, V, copula_cdf(U, V, theta, family) - t_c


def central_kendall_contour(u_reg, v_reg, T, grid_size=200, family=None, method='mle'):
    '''
    Compute the central Kendall T-year contour from data (no bootstrap).

    Parameters
    ----------
    u_reg, v_reg : array-like
    T : float
    grid_size : int
    family : str or None   If None, selects the best family by AICc.
    method : {'mle', 'tau'}

    Returns
    -------
    U, V : 2D arrays
    Z : 2D array   Zero-contour is the Kendall T-year contour.
    '''
    if family is None:
        _, best  = select_copula(u_reg, v_reg, method=method)
        family   = best['family']
        theta    = best['theta']
    else:
        theta, _ = fit_copula(u_reg, v_reg, family, method=method)

    return kendall_contour_grid(theta, T, family, grid_size)


def bootstrap_kendall_contours(
    u_reg,
    v_reg,
    T,
    family=None,
    n_boot=500,
    grid_size=200,
    alpha=0.05,
    random_state=None,
    method='mle',
):
    '''
    Bootstrap Kendall return-period contour confidence bands (grid form).

    Parameters
    ----------
    u_reg, v_reg : array-like
    T : float
    family : str or None
        None → re-selects best family on each replicate (data-driven).
    n_boot, grid_size, alpha, random_state, method : as usual.

    Returns
    -------
    U, V : 2D arrays
    contours : dict  Keys: 'median', 'lower', 'upper'.
    '''
    rng   = np.random.default_rng(random_state)
    u_reg = np.asarray(u_reg)
    v_reg = np.asarray(v_reg)
    n     = len(u_reg)

    u_grid = np.linspace(1e-4, 1 - 1e-4, grid_size)
    v_grid = np.linspace(1e-4, 1 - 1e-4, grid_size)
    U, V   = np.meshgrid(u_grid, v_grid)

    stack = []
    for _ in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        u_b, v_b = u_reg[idx], v_reg[idx]

        if family is None:
            _, best = select_copula(u_b, v_b, method=method)
            fam_b, theta_b = best['family'], best['theta']
        else:
            fam_b   = family
            theta_b, _ = fit_copula(u_b, v_b, fam_b, method=method)

        t_b = invert_kendall_level(1.0 - 1.0 / T, theta_b, fam_b)
        stack.append(copula_cdf(U, V, theta_b, fam_b) - t_b)

    stack = np.asarray(stack)
    return U, V, {
        'median': np.quantile(stack, 0.50,           axis=0),
        'lower':  np.quantile(stack, alpha / 2,       axis=0),
        'upper':  np.quantile(stack, 1.0 - alpha / 2, axis=0),
    }


def bootstrap_kendall_contours_lines(
    u_reg,
    v_reg,
    T_K,
    family=None,
    n_boot=500,
    n_mc=200_000,
    u_min=1e-3,
    u_max=1 - 1e-3,
    n_points=200,
    alpha=0.05,
    method='mle',
):
    '''
    Bootstrap Kendall contour confidence bands returned as (u, v) line arrays.

    Uses Monte Carlo estimation of the Kendall CDF rather than analytical
    inversion. Suitable for plotting contours directly as lines.

    Parameters
    ----------
    u_reg, v_reg : array-like
    T_K : float             Kendall return period.
    family : str or None    None → re-selects per replicate.
    n_boot : int
    n_mc : int              MC samples for Kendall CDF estimation per replicate.
    u_min, u_max : float
    n_points : int
    alpha : float
    method : {'mle', 'tau'}

    Returns
    -------
    result : dict
        Keys: 'u', 'v_median', 'v_lower', 'v_upper', 'all_contours'.
    '''
    u_grid    = np.linspace(u_min, u_max, n_points)
    rng       = np.random.default_rng()
    contours  = []

    for _ in range(n_boot):
        idx  = rng.choice(len(u_reg), size=len(u_reg), replace=True)
        u_b  = u_reg[idx]
        v_b  = v_reg[idx]

        if family is None:
            _, best = select_copula(u_b, v_b, method=method)
            fam_b, theta_b = best['family'], best['theta']
        else:
            fam_b   = family
            theta_b, _ = fit_copula(u_b, v_b, fam_b, method=method)

        # MC estimate of Kendall level for T_K
        u_mc = rng.uniform(0, 1, n_mc)
        v_mc = rng.uniform(0, 1, n_mc)
        C_mc = np.sort(copula_cdf(u_mc, v_mc, theta_b, fam_b))
        t_b  = np.quantile(C_mc, 1.0 - 1.0 / T_K)

        v_curve = copula_contour(u_grid, t_b, theta_b, fam_b)
        contours.append(v_curve)

    contours = np.array(contours)
    return {
        'u':            u_grid,
        'v_median':     np.nanmedian(contours, axis=0),
        'v_lower':      np.nanquantile(contours, alpha / 2,       axis=0),
        'v_upper':      np.nanquantile(contours, 1.0 - alpha / 2, axis=0),
        'all_contours': contours,
    }


# ── Sampling ──────────────────────────────────────────────────────────────────

def sample_gumbel(n, theta):
    '''
    Sample from a bivariate Gumbel copula using the Marshall–Olkin algorithm.

    Parameters
    ----------
    n : int
    theta : float

    Returns
    -------
    U, V : ndarray, shape (n,)
    '''
    alpha = 1.0 / theta
    S  = levy_stable.rvs(alpha, 1, size=n)
    E1 = -np.log(uniform.rvs(size=n))
    E2 = -np.log(uniform.rvs(size=n))
    return np.exp(-(E1 / S) ** alpha), np.exp(-(E2 / S) ** alpha)


def sample_copula(n, theta, family, random_state=None):
    '''
    Sample from a bivariate Archimedean copula via conditional inversion.

    Algorithm:
        1. u ~ U(0,1)
        2. w ~ U(0,1)
        3. v = h_inv(w | u; theta, family)   [conditional on u]

    For Gumbel, ``sample_gumbel`` (Marshall–Olkin) is faster for large n.
    For Clayton and Frank, this is the natural sampler.

    Parameters
    ----------
    n : int
    theta : float
    family : str
    random_state : int or None

    Returns
    -------
    U, V : ndarray, shape (n,)
    '''
    rng = np.random.default_rng(random_state)
    u   = rng.uniform(0.0, 1.0, n)
    w   = rng.uniform(0.0, 1.0, n)
    v   = h_func_inv(w, u, theta, family)
    return u, v


# ── Utilities ─────────────────────────────────────────────────────────────────

def pseudo_observations(x):
    '''
    Convert a data vector to pseudo-observations in (0, 1) via scaled ranks.

    r_i / (n + 1)

    Parameters
    ----------
    x : array-like

    Returns
    -------
    u : array-like
    '''
    r = rankdata(x, method='average')
    return r / (len(x) + 1)


def best_fit_rv(data, dist_names, print_out=False):
    '''
    Fit candidate marginal distributions and select the best by AICc.

    Parameters
    ----------
    data : array-like
    dist_names : list of str   scipy.stats distribution names.
    print_out : bool

    Returns
    -------
    best_dist : scipy.stats distribution class
    best_params : tuple
    best_aicc : float
    '''
    best_aicc  = np.inf
    best_dist  = None
    best_params = None
    n = len(data)

    for name in dist_names:
        dist   = getattr(sp.stats, name)
        params = dist.fit(data)
        ll     = np.sum(dist.logpdf(data, *params))
        k      = len(params)
        aic    = 2 * k - 2 * ll
        aicc   = aic + (2 * k * (k + 1)) / (n - k - 1)

        if aicc < best_aicc:
            best_aicc   = aicc
            best_dist   = dist
            best_params = params

        if print_out:
            print(f'Distribution: {name}, AICc: {aicc:.2f}')

    return best_dist, best_params, best_aicc
