import numpy as np
import scipy as sp
import pandas as pd
from scipy.stats import levy_stable, uniform, norm, rankdata
from scipy.optimize import minimize, brentq
from scipy.stats import kendalltau




# Temporal Clustering
'''
site_id | start_time | end_time | severity | duration
'''

def event_midpoint(start, end):
    return start + (end - start) / 2




def temporal_clustering(events, delta_t):
    '''
    Cluster site-level drought events into independent regional events
    using a fixed reference time for each cluster.

    Parameters
    ----------
    events : pandas.DataFrame
        Must be sorted by 'event_time'
    delta_t : pandas.Timedelta

    Returns
    -------
    clusters : list of pandas.DataFrame
    '''

    clusters = []

    # Initialize first cluster
    current_cluster = [events.iloc[0]]
    t_ref = events.iloc[0]['event_time']

    for i in range(1, len(events)):
        t_cur = events.iloc[i]['event_time']

        # Compare to fixed reference time
        if (t_cur - t_ref) <= delta_t:
            current_cluster.append(events.iloc[i])
        else:
            # Close current cluster
            clusters.append(pd.DataFrame(current_cluster))

            # Start new cluster
            current_cluster = [events.iloc[i]]
            t_ref = t_cur

    # Append final cluster
    clusters.append(pd.DataFrame(current_cluster))

    return clusters





def aggregate_cluster(cluster,
                      duration_rule='max',
                      severity_rule='max'):
    '''
    Aggregate a regional drought cluster into one multivariate event.
    '''
    if severity_rule == 'max':
        S = cluster['severity'].max()
    elif severity_rule == 'sum':
        S = cluster['severity'].sum()
    elif severity_rule == 'mean':
        S = cluster['severity'].mean()
    elif severity_rule == 'median':
        S = cluster['severity'].median()
    else:
        raise ValueError('Unknown severity rule')

    if duration_rule == 'max':
        D = cluster['duration'].max()
    elif duration_rule=='sum':
        D = cluster['duration'].sum()
    elif duration_rule == 'mean':
        D = cluster['duration'].mean()
    elif duration_rule=='median':
        D = cluster['duration'].median()
    else:
        raise ValueError('Unknown duration rule')

    return D, S




def pseudo_observations(x):
    '''
    Convert data to pseudo-observations in (0,1)
    '''
    r = rankdata(x, method='average')
    return r / (len(x) + 1)




# Kendall bootstrap
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




def invert_gumbel_v(u, t, theta):
    """Solve C(u,v)=t for v"""
    A = (-np.log(t))**theta - (-np.log(u))**theta
    if A <= 0:
        return np.nan
    return np.exp(-(A)**(1/theta))




def kendall_function(theta, n_mc=200_000):
    u = np.random.rand(n_mc)
    v = np.random.rand(n_mc)
    C = gumbel_copula(u, v, theta)
    return np.sort(C)




def kendall_threshold(C_sorted, T_K):
    p = 1 - 1 / T_K
    return np.quantile(C_sorted, p)




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
    return brentq(f, 1e-10, 1 - 1e-10)




def fit_gumbel_theta(u, v):
    '''
    Fit Gumbel copula using Kendall's tau inversion
    (robust and fast for bootstrap)
    '''
    tau, _ = kendalltau(u, v)
    return 1.0 / (1.0 - tau)




def bootstrap_kendall_contours(u_reg, v_reg, T,
                               n_boot=500,
                               grid_size=200,
                               alpha=0.05,
                               random_state=None):
    '''
    Bootstrap Kendall contour confidence bands using ESS
    '''

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
        'median': median,
        'lower': lower,
        'upper': upper
    }




def central_kendall_contour(u_reg, v_reg, T, grid_size=200):
    theta_hat = fit_gumbel_theta(u_reg, v_reg)
    t_hat = invert_kendall_level(1 - 1/T, theta_hat)

    u = np.linspace(1e-4, 1 - 1e-4, grid_size)
    v = np.linspace(1e-4, 1 - 1e-4, grid_size)
    U, V = np.meshgrid(u, v)

    C = gumbel_copula(U, V, theta_hat)
    return U, V, C - t_hat




def bootstrap_kendall_contours_lines(
    u_reg,
    v_reg,
    T_K,
    n_boot=500,
    n_mc=200_000,
    u_min=1e-3,
    u_max=1-1e-3,
    n_points=200,
    alpha=0.05,
):
    """
    Returns Kendall contour median and confidence bands as lists of (u,v)
    """

    u_grid = np.linspace(u_min, u_max, n_points)

    contours = []

    for b in range(n_boot):

        # ── Bootstrap resample regional events ──
        idx = np.random.randint(0, len(u_reg), len(u_reg))
        u_b = u_reg[idx]
        v_b = v_reg[idx]

        # ── Fit copula parameter ──
        theta_b = fit_gumbel_theta(u_b, v_b)

        # ── Kendall threshold ──
        C_sorted = kendall_function(theta_b, n_mc)
        t_b = kendall_threshold(C_sorted, T_K)

        # ── Compute contour (u,v) ──
        v_curve = np.array([
            invert_gumbel_v(u, t_b, theta_b) for u in u_grid
        ])

        contours.append(v_curve)

    contours = np.array(contours)

    # ── Confidence bands ──
    v_med = np.nanmedian(contours, axis=0)
    v_lo  = np.nanquantile(contours, alpha/2, axis=0)
    v_hi  = np.nanquantile(contours, 1-alpha/2, axis=0)

    return {
        "u": u_grid,
        "v_median": v_med,
        "v_lower": v_lo,
        "v_upper": v_hi,
        "all_contours": contours  # optional, but very useful
    }








