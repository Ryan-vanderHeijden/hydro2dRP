import numpy as np
import scipy as sp
from scipy.stats import levy_stable, uniform, norm
from scipy.optimize import minimize
from scipy.stats import kendalltau




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




def kendall_distribution_gumbel(t, theta):
    '''
    Kendall distribution K_C(t) for the Gumbel copula
    '''
    return t - (t / theta) * np.log(t)




def invert_kendall_level(K_target, theta):
    '''
    Find t such that K_C(t) = K_target
    '''
    f = lambda t: kendall_distribution_gumbel(t, theta) - K_target
    return sp.brentq(f, 1e-10, 1 - 1e-10)




def fit_gumbel_theta(u, v):
    """
    Fit Gumbel copula using Kendall's tau inversion
    (robust and fast for bootstrap)
    """
    tau, _ = kendalltau(u, v)
    return 1.0 / (1.0 - tau)




def bootstrap_kendall_contours(u_reg, v_reg, T,
                               n_boot=500,
                               grid_size=200,
                               alpha=0.05,
                               random_state=None):
    """
    Bootstrap Kendall contour confidence bands using ESS
    """

    rng = np.random.default_rng(random_state)
    n_eff = len(u_reg)

    # Grid
    u = np.linspace(1e-4, 1 - 1e-4, grid_size)
    v = np.linspace(1e-4, 1 - 1e-4, grid_size)
    U, V = np.meshgrid(u, v)

    contour_stack = []

    for b in range(n_boot):
        # ESS-aware resampling
        idx = rng.choice(n_eff, size=n_eff, replace=True)
        u_b = u_reg[idx]
        v_b = v_reg[idx]

        # Refit copula
        theta_b = fit_gumbel_theta(u_b, v_b)

        # Kendall level
        t_b = invert_kendall_level(1 - 1/T, theta_b)

        # Copula surface
        C_b = gumbel_copula(U, V, theta_b)

        contour_stack.append(C_b - t_b)

    contour_stack = np.asarray(contour_stack)

    # Empirical quantiles
    lower = np.quantile(contour_stack, alpha / 2, axis=0)
    upper = np.quantile(contour_stack, 1 - alpha / 2, axis=0)
    median = np.quantile(contour_stack, 0.5, axis=0)

    return U, V, {
        "median": median,
        "lower": lower,
        "upper": upper
    }




def central_kendall_contour(u_reg, v_reg, T, grid_size=200):
    theta_hat = fit_gumbel_theta(u_reg, v_reg)
    t_hat = invert_kendall_level(1 - 1/T, theta_hat)

    u = np.linspace(1e-4, 1 - 1e-4, grid_size)
    v = np.linspace(1e-4, 1 - 1e-4, grid_size)
    U, V = np.meshgrid(u, v)

    C = gumbel_copula(U, V, theta_hat)
    return U, V, C - t_hat









