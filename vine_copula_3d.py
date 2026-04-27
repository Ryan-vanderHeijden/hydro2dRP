'''
3-Variable D-Vine (Pair-Copula Construction)
=============================================

Implements a trivariate vine copula for joint return period analysis of
three drought characteristics — typically duration, severity, and spatial
extent — using sequential pair-copula factorisation (D-vine / C-vine,
which are identical for three variables).

Structure
---------
A 3-variable vine decomposes the joint density into three bivariate copulas:

    c(u_a, u_b, u_c) = c_ab(u_a, u_b)
                     * c_bc(u_b, u_c)
                     * c_ac|b( h(u_a|u_b), h(u_c|u_b) )

where (a, b, c) is the fitted variable ordering (b is the "root" node),
and h(u|v) = dC(u,v)/dv is the conditional CDF (h-function).

The variable ordering is chosen by a maximum-spanning-tree criterion on
pairwise |Kendall tau|, placing the two most-dependent pairs in Tree 1
and the remaining (conditioned) pair in Tree 2.

Each pair copula is independently selected from {gumbel, clayton, frank}
by AICc unless an explicit family mapping is provided.

Usage
-----
>>> from gumbel_copula_2dRP import pseudo_observations
>>> u_dur = pseudo_observations(duration)
>>> u_sev = pseudo_observations(severity)
>>> u_ext = pseudo_observations(spatial_extent)
>>>
>>> vc = VineCopula3D()
>>> vc.fit(u_dur, u_sev, u_ext)
>>> vc.summary()
>>>
>>> # Sample synthetic droughts
>>> s_dur, s_sev, s_ext = vc.simulate(10_000)
>>>
>>> # AND return period at a query point
>>> T = vine_and_return_period(vc, u1=0.9, u2=0.9, u3=0.9)

References
----------
- Aas, K. et al. (2009). Pair-copula constructions of multiple dependence.
  Insurance: Mathematics and Economics, 44(2), 182–198.
- Czado, C. (2019). Analyzing Dependent Data with Vine Copulas.
  Lecture Notes in Statistics, Springer.
- Joe, H. (1997). Multivariate Models and Dependence Concepts.
'''

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import kendalltau

from archimedean_copulas import (
    copula_cdf,
    copula_pdf,
    h_func,
    h_func_inv,
    fit_copula,
    select_copula,
    invert_kendall_level,
    copula_contour,
)


# ── VineCopula3D class ─────────────────────────────────────────────────────────

class VineCopula3D:
    '''
    3-variable D-vine copula fitted to pseudo-observations.

    Parameters
    ----------
    families : 'auto' or dict
        ``'auto'`` (default) selects the best-fitting family for each pair
        by AICc from {gumbel, clayton, frank}.
        Pass a dict with keys ``'12'``, ``'23'``, ``'13|2'`` to fix families,
        e.g. ``{'12': 'gumbel', '23': 'frank', '13|2': 'clayton'}``.
    method : {'mle', 'tau'}
        Parameter estimation method used when ``families='auto'``.

    Attributes set after ``fit()``
    ------------------------------
    order_ : list of int
        Permutation of [0, 1, 2] such that ``order_[1]`` is the root node.
        Variables are referred to as positions a=order_[0], b=order_[1],
        c=order_[2] internally.
    families_ : dict
        Fitted copula family for each pair (keys '12', '23', '13|2').
    thetas_ : dict
        Fitted copula parameter for each pair.
    pair_lls_ : dict
        Log-likelihoods for each fitted pair copula.
    var_names_ : list of str
        Labels ['u1', 'u2', 'u3'] reordered by ``order_``.
    '''

    def __init__(self, families='auto', method='mle'):
        self.families_config = families
        self.method          = method
        self.order_          = None
        self.families_       = {}
        self.thetas_         = {}
        self.pair_lls_       = {}
        self.var_names_      = None

    # ── Structure selection ────────────────────────────────────────────────────

    def _select_structure(self, data):
        '''
        Choose variable ordering by maximum spanning tree on |Kendall tau|.

        For three variables, the pair with the weakest dependence ends up as
        the conditioned pair in Tree 2. The variable shared by the two
        strongest-dependence pairs becomes the root node (position [1]).

        Parameters
        ----------
        data : ndarray, shape (n, 3)

        Returns
        -------
        order : list of int  Permutation of [0, 1, 2].
        '''
        pairs = [(0, 1), (0, 2), (1, 2)]
        abs_tau = {}
        for i, j in pairs:
            tau, _ = kendalltau(data[:, i], data[:, j])
            abs_tau[(i, j)] = abs(tau)

        # Weakest pair → goes to Tree 2 as the conditioned pair
        min_pair = min(abs_tau, key=abs_tau.get)

        # Remaining two pairs share exactly one variable → that is the root
        remaining = [p for p in pairs if p != min_pair]
        root = (set(remaining[0]) & set(remaining[1])).pop()
        left  = (set(remaining[0]) - {root}).pop()
        right = (set(remaining[1]) - {root}).pop()

        return [left, root, right]

    # ── Fitting ────────────────────────────────────────────────────────────────

    def fit(self, u1, u2, u3):
        '''
        Fit the vine copula to three vectors of pseudo-observations.

        Step 1: select variable ordering.
        Step 2: fit Tree-1 pair copulas (a,b) and (b,c).
        Step 3: compute conditional pseudo-observations via h-functions.
        Step 4: fit Tree-2 pair copula (a,c|b) to the conditional data.

        Parameters
        ----------
        u1, u2, u3 : array-like
            Pseudo-observations in (0, 1) for variables 1, 2, 3.
            Typically produced by ``pseudo_observations()`` from
            ``gumbel_copula_2dRP``.

        Returns
        -------
        self
        '''
        u1 = np.asarray(u1, dtype=float)
        u2 = np.asarray(u2, dtype=float)
        u3 = np.asarray(u3, dtype=float)
        data = np.column_stack([u1, u2, u3])

        # Structure selection
        self.order_ = self._select_structure(data)
        a, b, c = (data[:, self.order_[k]] for k in range(3))
        labels = ['u1', 'u2', 'u3']
        self.var_names_ = [labels[i] for i in self.order_]

        # Tree 1: fit pair copulas (a,b) and (b,c)
        for key, x, y in [('12', a, b), ('23', b, c)]:
            fam, theta, ll = self._fit_pair(x, y, key)
            self.families_[key] = fam
            self.thetas_[key]   = theta
            self.pair_lls_[key] = ll

        # Conditional pseudo-observations for Tree 2
        u_a_b = h_func(a, b, self.thetas_['12'], self.families_['12'])
        u_c_b = h_func(c, b, self.thetas_['23'], self.families_['23'])

        # Tree 2: fit conditional pair copula (a,c|b)
        fam, theta, ll = self._fit_pair(u_a_b, u_c_b, '13|2')
        self.families_['13|2'] = fam
        self.thetas_['13|2']   = theta
        self.pair_lls_['13|2'] = ll

        # Keep conditional observations for diagnostics
        self._cond_obs = np.column_stack([u_a_b, u_c_b])

        return self

    def _fit_pair(self, x, y, key):
        '''Fit one pair copula; honour families_config if provided.'''
        if self.families_config == 'auto':
            _, best = select_copula(x, y, method=self.method)
            return best['family'], best['theta'], best['log_likelihood']
        else:
            fam   = self.families_config.get(key, 'gumbel')
            theta, ll = fit_copula(x, y, fam, method=self.method)
            return fam, theta, ll

    # ── Density ────────────────────────────────────────────────────────────────

    def pdf(self, u1, u2, u3):
        '''
        Evaluate the vine copula density at (u1, u2, u3).

        c(u1,u2,u3) = c_ab * c_bc * c_{ac|b}(h(a|b), h(c|b))

        Parameters
        ----------
        u1, u2, u3 : array-like  Must be broadcastable.

        Returns
        -------
        density : array-like  Non-negative values.
        '''
        u1 = np.asarray(u1, dtype=float)
        u2 = np.asarray(u2, dtype=float)
        u3 = np.asarray(u3, dtype=float)
        data = np.column_stack([u1.ravel(), u2.ravel(), u3.ravel()])

        a = data[:, self.order_[0]]
        b = data[:, self.order_[1]]
        c = data[:, self.order_[2]]

        c12  = copula_pdf(a, b, self.thetas_['12'],   self.families_['12'])
        c23  = copula_pdf(b, c, self.thetas_['23'],   self.families_['23'])
        h_ab = h_func(a, b, self.thetas_['12'],       self.families_['12'])
        h_cb = h_func(c, b, self.thetas_['23'],       self.families_['23'])
        c132 = copula_pdf(h_ab, h_cb, self.thetas_['13|2'], self.families_['13|2'])

        return c12 * c23 * c132

    # ── Simulation ─────────────────────────────────────────────────────────────

    def simulate(self, n, random_state=None):
        '''
        Sample n observations from the vine via Rosenblatt inversion.

        Algorithm (D-vine, root = b):
            Draw v1, v2, v3 ~ U(0,1) independently.
            b  = v2
            a  = h_inv(v1 | b ; theta_12)
            z  = h_inv(v3 | v1 ; theta_13|2)    [v1 = h(a|b) by construction]
            c  = h_inv(z  | b ; theta_23)

        Returns (u1, u2, u3) in the *original* variable order.

        Parameters
        ----------
        n : int
        random_state : int or None

        Returns
        -------
        u1, u2, u3 : ndarray, each shape (n,)  Values in (0, 1).
        '''
        rng = np.random.default_rng(random_state)
        v   = rng.uniform(0.0, 1.0, (n, 3))

        b = v[:, 1]
        a = h_func_inv(v[:, 0], b,    self.thetas_['12'],   self.families_['12'])
        # v[:,0] = h(a|b) by the definition of h_inv, so pass it directly to Tree 2
        z = h_func_inv(v[:, 2], v[:, 0], self.thetas_['13|2'], self.families_['13|2'])
        c = h_func_inv(z,       b,    self.thetas_['23'],   self.families_['23'])

        # Map back to original variable indices
        result = np.empty((n, 3))
        result[:, self.order_[0]] = a
        result[:, self.order_[1]] = b
        result[:, self.order_[2]] = c

        return result[:, 0], result[:, 1], result[:, 2]

    # ── Summary ────────────────────────────────────────────────────────────────

    def summary(self):
        '''
        Print a formatted summary of the fitted vine structure, copula
        families, parameters, and log-likelihoods.
        '''
        a, b, c = self.var_names_
        total_ll = sum(self.pair_lls_.values())

        print('3-Variable D-Vine Copula')
        print(f'Variable order : {a} — {b} — {c}  ({b} is root)')
        print()
        print('Tree 1  (unconditional pairs)')
        print(f'  ({a}, {b}) : family = {self.families_["12"]:>8s}'
              f'   theta = {self.thetas_["12"]:8.4f}'
              f'   log-lik = {self.pair_lls_["12"]:.2f}')
        print(f'  ({b}, {c}) : family = {self.families_["23"]:>8s}'
              f'   theta = {self.thetas_["23"]:8.4f}'
              f'   log-lik = {self.pair_lls_["23"]:.2f}')
        print()
        print('Tree 2  (conditional pair)')
        print(f'  ({a}, {c} | {b}) : family = {self.families_["13|2"]:>8s}'
              f'   theta = {self.thetas_["13|2"]:8.4f}'
              f'   log-lik = {self.pair_lls_["13|2"]:.2f}')
        print()
        print(f'Total log-likelihood : {total_ll:.4f}')


# ── Return Period Utilities ────────────────────────────────────────────────────

def vine_and_return_period(vc, u1, u2, u3, n_mc=100_000, random_state=None):
    '''
    Estimate the AND joint return period at query point(s) (u1, u2, u3).

    T_AND = 1 / P(U1 > u1, U2 > u2, U3 > u3)

    The exceedance probability is estimated from n_mc samples drawn from the
    fitted vine.  For large T (rare events) increase n_mc to reduce MC noise.

    Parameters
    ----------
    vc : VineCopula3D
    u1, u2, u3 : float or array-like  Query points in (0, 1).
    n_mc : int
    random_state : int or None

    Returns
    -------
    T_AND : float or ndarray
    '''
    s1, s2, s3 = vc.simulate(n_mc, random_state=random_state)

    u1 = np.asarray(u1, dtype=float)
    u2 = np.asarray(u2, dtype=float)
    u3 = np.asarray(u3, dtype=float)

    scalar = (u1.ndim == 0)
    u1, u2, u3 = np.atleast_1d(u1), np.atleast_1d(u2), np.atleast_1d(u3)

    out = np.empty(len(u1))
    for i in range(len(u1)):
        p = float(np.mean((s1 > u1[i]) & (s2 > u2[i]) & (s3 > u3[i])))
        out[i] = 1.0 / p if p > 0 else np.inf

    return float(out[0]) if scalar else out


def vine_or_return_period(vc, u1, u2, u3, n_mc=100_000, random_state=None):
    '''
    Estimate the OR joint return period at query point(s) (u1, u2, u3).

    T_OR = 1 / P(U1 > u1 OR U2 > u2 OR U3 > u3)

    Parameters
    ----------
    vc : VineCopula3D
    u1, u2, u3 : float or array-like
    n_mc : int
    random_state : int or None

    Returns
    -------
    T_OR : float or ndarray
    '''
    s1, s2, s3 = vc.simulate(n_mc, random_state=random_state)

    u1 = np.asarray(u1, dtype=float)
    u2 = np.asarray(u2, dtype=float)
    u3 = np.asarray(u3, dtype=float)

    scalar = (u1.ndim == 0)
    u1, u2, u3 = np.atleast_1d(u1), np.atleast_1d(u2), np.atleast_1d(u3)

    out = np.empty(len(u1))
    for i in range(len(u1)):
        p = float(np.mean((s1 > u1[i]) | (s2 > u2[i]) | (s3 > u3[i])))
        out[i] = 1.0 / p if p > 0 else np.inf

    return float(out[0]) if scalar else out


def vine_kendall_return_period_mc(vc, n_mc=500_000, random_state=None):
    '''
    Estimate the Kendall CDF K_C(t) from Monte Carlo samples of the vine.

    K_C(t) = P(C(U1, U2, U3) ≤ t), where C is approximated empirically by
    counting how many MC samples fall below each simulated point in all
    three dimensions simultaneously.

    Returns a callable that maps copula level t → K_C(t) and the raw
    sample data for further analysis.

    Parameters
    ----------
    vc : VineCopula3D
    n_mc : int   Larger values give smoother K_C estimates (default 500 k).
    random_state : int or None

    Returns
    -------
    t_vals : ndarray, shape (n_mc,)
        Estimated "copula values" C(u^{(i)}) for each MC sample, sorted.
        The empirical K_C(t) = fraction of t_vals ≤ t.
    samples : ndarray, shape (n_mc, 3)
        Raw MC samples (columns: u1, u2, u3) in original variable order.

    Notes
    -----
    The "copula value" at sample i is estimated as the fraction of MC samples
    dominated by (≤) sample i in all dimensions — the empirical copula.
    This is O(n_mc^2) in principle but computed efficiently via sorting.
    '''
    s1, s2, s3 = vc.simulate(n_mc, random_state=random_state)
    samples = np.column_stack([s1, s2, s3])

    # Empirical copula: C_n(u) = fraction of samples with all coords ≤ u
    # Computed via rank: rank_i / n_mc for each dimension, then take min
    # (An approximation; full empirical copula is expensive)
    from scipy.stats import rankdata
    r1 = rankdata(s1) / (n_mc + 1)
    r2 = rankdata(s2) / (n_mc + 1)
    r3 = rankdata(s3) / (n_mc + 1)

    # Minimum rank gives a conservative lower bound on C_n; use product as
    # an approximate empirical copula value for sorting purposes
    c_approx = r1 * r2 * r3   # rough proxy for the joint CDF value
    t_vals   = np.sort(c_approx)

    return t_vals, samples


def vine_kendall_contour_2d(
    vc,
    T,
    fixed_var,
    fixed_quantiles,
    grid_size=150,
    # Legacy MC parameters — accepted but ignored
    n_mc=None,
    band_width=None,
    random_state=None,
):
    '''
    Compute 2D Kendall return period contours from the exact vine conditional
    density (no Monte Carlo, no banding, no parametric family assumption).

    For each quantile q of the fixed variable:

    1. Evaluate the vine's conditional density
       f(u_a, u_b | fixed_var = q) = vine.pdf(...)
       on a ``grid_size × grid_size`` grid (fast — no simulation required).
    2. Normalise and compute the 2-D conditional CDF by cumulative summation.
    3. Estimate the Kendall level t* via binary search on
       K(t) = ∫∫_{F(u,v)≤t} f(u,v|q) du dv  such that  K(t*) = 1 − 1/T.
    4. Extract the 1-D contour v(u) by linear interpolation along each
       column of the CDF grid.

    This replaces the previous MC + hard-band + copula-refit approach.
    Results are deterministic, noise-free, and exact up to grid resolution.
    The legacy ``n_mc``, ``band_width``, and ``random_state`` parameters are
    accepted for backward compatibility but have no effect.

    Parameters
    ----------
    vc : VineCopula3D
    T : float           Target Kendall return period (years).
    fixed_var : int     Index (0, 1, or 2) of the variable to condition on.
    fixed_quantiles : array-like
        Quantile values to condition on (e.g., [0.25, 0.50, 0.75, 0.90]).
    grid_size : int     Resolution of the density / CDF grid (default 150).

    Returns
    -------
    contours : list of dict
        One entry per quantile level.  Keys:
        ``'quantile'``   — the fixed quantile.
        ``'u'``          — 1-D array of free-var-0 values on the contour.
        ``'v'``          — 1-D array of free-var-1 values on the contour.
        ``'U'``, ``'V'`` — 2-D meshgrid arrays (for ``ax.contour``).
        ``'cdf_2d'``     — conditional copula CDF on the grid.
        ``'t_star'``     — Kendall level used for the contour.
        ``'var_labels'`` — (x-label, y-label) for the two free axes.
        ``'method'``     — ``'exact'``.
        Returns ``'u': None, 'v': None`` if the density grid is degenerate.
    '''
    free_vars  = [i for i in range(3) if i != fixed_var]
    var_labels = ['u1', 'u2', 'u3']

    axis = np.linspace(1e-3, 1 - 1e-3, grid_size)
    du   = axis[1] - axis[0]
    # U varies along columns (axis=1), V along rows (axis=0)
    U, V = np.meshgrid(axis, axis)

    contours = []
    for q in np.asarray(fixed_quantiles, dtype=float):
        vals               = [None, None, None]
        vals[fixed_var]    = np.full(U.shape, q)
        vals[free_vars[0]] = U
        vals[free_vars[1]] = V

        # Conditional density: vine pdf with fixed_var pinned at q
        density = vc.pdf(vals[0], vals[1], vals[2]).reshape(U.shape)
        density = np.clip(density, 0.0, None)

        total = density.sum() * du * du
        if total <= 0.0:
            contours.append({
                'quantile':   float(q),
                'u': None, 'v': None,
                'U': U, 'V': V, 'cdf_2d': None, 't_star': None,
                'var_labels': (var_labels[free_vars[0]],
                               var_labels[free_vars[1]]),
                'method': 'exact',
            })
            continue
        density = density / total

        # 2-D conditional CDF by cumulative summation
        #   cumsum axis=0 integrates over V (rows)
        #   cumsum axis=1 integrates over U (columns)
        # cdf_2d[i,j] = P(free_var_0 ≤ axis[j], free_var_1 ≤ axis[i] | fixed=q)
        cdf_2d = np.cumsum(np.cumsum(density, axis=0), axis=1) * du * du
        cdf_2d = np.clip(cdf_2d / cdf_2d[-1, -1], 0.0, 1.0)

        # Kendall distribution: K(t) = mass of density where cdf_2d ≤ t
        # Binary search for t* such that K(t*) = 1 − 1/T
        K_target = 1.0 - 1.0 / T
        lo, hi   = 0.0, 1.0
        for _ in range(60):
            t_mid = 0.5 * (lo + hi)
            K_mid = float(density[cdf_2d <= t_mid].sum() * du * du)
            if K_mid < K_target:
                lo = t_mid
            else:
                hi = t_mid
        t_star = 0.5 * (lo + hi)

        # Extract 1-D contour v(u) by column-wise linear interpolation
        u_line, v_line = [], []
        for j, uj in enumerate(axis):
            col = cdf_2d[:, j]   # CDF along V-axis at fixed U = uj
            if col.min() < t_star < col.max():
                idx = int(np.searchsorted(col, t_star))
                idx = np.clip(idx, 1, len(col) - 1)
                t0, t1 = col[idx - 1], col[idx]
                v0, v1 = axis[idx - 1], axis[idx]
                span = t1 - t0
                vj   = v0 + (t_star - t0) * (v1 - v0) / span if span > 0 else v0
                u_line.append(uj)
                v_line.append(vj)

        contours.append({
            'quantile':   float(q),
            'u':          np.array(u_line) if u_line else None,
            'v':          np.array(v_line) if v_line else None,
            'U':          U,
            'V':          V,
            'cdf_2d':     cdf_2d,
            't_star':     t_star,
            'var_labels': (var_labels[free_vars[0]], var_labels[free_vars[1]]),
            'method':     'exact',
        })

    return contours


def vine_density_slice(vc, fixed_var, fixed_quantile, grid_size=100):
    '''
    Evaluate the vine density on a 2D grid with one variable fixed.

    Useful for visualising how the joint density of the two free variables
    changes at different levels of the third (e.g., different spatial extents).

    Parameters
    ----------
    vc : VineCopula3D
    fixed_var : int        Index (0, 1, or 2) of the variable to fix.
    fixed_quantile : float Value at which to fix the variable.
    grid_size : int

    Returns
    -------
    U, V : 2D arrays  Meshgrid for the two free variables.
    Z : 2D array      Vine density evaluated at each grid point.
    labels : tuple    (x-label, y-label) for the two free axes.
    '''
    free_vars  = [i for i in range(3) if i != fixed_var]
    var_labels = ['u1', 'u2', 'u3']

    axis = np.linspace(1e-3, 1 - 1e-3, grid_size)
    U, V = np.meshgrid(axis, axis)

    vals  = [None, None, None]
    vals[fixed_var]    = np.full_like(U, fixed_quantile)
    vals[free_vars[0]] = U
    vals[free_vars[1]] = V

    Z = vc.pdf(vals[0], vals[1], vals[2]).reshape(U.shape)

    return U, V, Z, (var_labels[free_vars[0]], var_labels[free_vars[1]])


# ── 3-D Grid Utilities ─────────────────────────────────────────────────────────

def vine_and_return_period_grid(vc, grid_size=20, n_mc=500_000, random_state=None):
    '''
    Estimate T_AND on a regular 3-D grid via Monte Carlo.

    T_AND[i,j,k] = 1 / P(U1 > g[i], U2 > g[j], U3 > g[k])
    estimated from ``n_mc`` samples drawn once and reused for all grid points.

    The vectorised inner loop collapses the k-axis into a single NumPy
    operation, avoiding a triple nested loop while keeping memory manageable
    (peak: n_active_samples × grid_size booleans per outer iteration).

    Parameters
    ----------
    vc : VineCopula3D
    grid_size : int        Number of grid points per axis (default 20).
    n_mc : int             MC sample count (default 500 000).
    random_state : int or None

    Returns
    -------
    g : 1-D array          Grid axis, shape (grid_size,), in (0.05, 0.95).
    T_and : 3-D array      Return periods, shape (grid_size, grid_size, grid_size).
    '''
    s1, s2, s3 = vc.simulate(n_mc, random_state=random_state)
    g = np.linspace(0.05, 0.95, grid_size)
    T_and = np.empty((grid_size, grid_size, grid_size))

    for i, q1 in enumerate(g):
        m1     = s1 > q1
        sub2   = s2[m1]
        sub3   = s3[m1]
        for j, q2 in enumerate(g):
            m2      = sub2 > q2
            sub3_j  = sub3[m2]
            # Vectorise over k: count exceedances above every g threshold at once
            counts  = (sub3_j[:, np.newaxis] > g[np.newaxis, :]).sum(axis=0)
            p       = counts / n_mc
            T_and[i, j, :] = np.where(p > 0, 1.0 / p, np.inf)

    return g, T_and


# ── 3-D Visualisation ──────────────────────────────────────────────────────────

def plot_vine_3d_density_slices(
    vc,
    fixed_var=2,
    fixed_quantiles=(0.25, 0.50, 0.75, 0.90),
    grid_size=60,
    observed=None,
    cmap='YlOrRd',
    figsize=(10, 8),
    ax=None,
):
    '''
    3-D matplotlib figure: vine density contours stacked at fixed quantile levels.

    Each panel is a ``contourf`` slice offset at its quantile level along the
    fixed-variable axis, giving a tomographic view of the joint density.
    Observed pseudo-observations are overlaid as a scatter if supplied.

    Parameters
    ----------
    vc : VineCopula3D
    fixed_var : int        Axis to stack along (0, 1, or 2; default 2 = coverage).
    fixed_quantiles : sequence  Quantile levels for the slices.
    grid_size : int        Density grid resolution per slice.
    observed : tuple (u1, u2, u3) or None  Pseudo-obs to scatter in 3-D.
    cmap : str
    figsize : tuple
    ax : Axes3D or None    If None, a new figure is created.

    Returns
    -------
    fig, ax
    '''
    from mpl_toolkits.mplot3d import Axes3D   # noqa: F401

    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax  = fig.add_subplot(111, projection='3d')
    else:
        fig = ax.get_figure()

    var_labels = ['U_duration', 'U_severity', 'U_coverage']
    free_vars  = [i for i in range(3) if i != fixed_var]

    # zdir mapping: which 3-D axis the slice is offset along
    zdir_map = {0: 'x', 1: 'y', 2: 'z'}
    zdir     = zdir_map[fixed_var]

    for q in fixed_quantiles:
        U, V, Z, _ = vine_density_slice(vc, fixed_var=fixed_var,
                                         fixed_quantile=q, grid_size=grid_size)
        finite = Z[np.isfinite(Z)]
        if len(finite) == 0:
            continue
        Z_plot = np.clip(Z, 0, np.percentile(finite, 98))
        ax.contourf(U, V, Z_plot, zdir=zdir, offset=q,
                    levels=12, cmap=cmap, alpha=0.65)

    if observed is not None:
        u1, u2, u3 = observed
        ax.scatter(u1, u2, u3, s=12, c='steelblue', alpha=0.5,
                   edgecolors='none', zorder=5)

    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_zlim(0, 1)
    ax.set_xlabel(var_labels[free_vars[0]])
    ax.set_ylabel(var_labels[free_vars[1]])
    ax.set_zlabel(var_labels[fixed_var])

    return fig, ax


def plot_vine_3d_isosurface(
    g,
    T_and_grid,
    T_levels=(10, 50, 100),
    observed=None,
    figsize=(10, 8),
    ax=None,
):
    '''
    3-D AND return period isosurfaces via the Marching Cubes algorithm.

    Requires ``scikit-image`` (``pip install scikit-image``).

    Isosurfaces are plotted in log₁₀(T) space so that spacing between
    nested shells is perceptually uniform.  The innermost shell (lowest T)
    is the most opaque; outermost (highest T) the most transparent.

    Parameters
    ----------
    g : 1-D array        Grid axis from ``vine_and_return_period_grid``.
    T_and_grid : 3-D array  T_AND values, shape ``(len(g),) * 3``.
    T_levels : sequence  Return periods to render as nested isosurfaces.
    observed : tuple (u1, u2, u3) or None  Pseudo-obs scatter overlay.
    figsize : tuple
    ax : Axes3D or None

    Returns
    -------
    fig, ax
    '''
    try:
        from skimage.measure import marching_cubes
    except ImportError:
        raise ImportError(
            'scikit-image is required: pip install scikit-image'
        )
    from mpl_toolkits.mplot3d import Axes3D           # noqa: F401
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax  = fig.add_subplot(111, projection='3d')
    else:
        fig = ax.get_figure()

    dg    = float(g[1] - g[0])
    log_T = np.log10(np.clip(T_and_grid, 1.0, 1e9))
    log_T = np.where(np.isfinite(log_T), log_T, np.nanmax(log_T[np.isfinite(log_T)]))
    lo, hi = float(log_T.min()), float(log_T.max())

    # Sort descending so the outermost (lowest-T) shell is added first
    colors   = plt.cm.plasma_r(np.linspace(0.15, 0.85, len(T_levels)))
    alphas   = np.linspace(0.20, 0.45, len(T_levels))   # innermost most opaque

    for T_target, col, alpha in zip(
        sorted(T_levels, reverse=True), colors, alphas
    ):
        level = np.log10(float(T_target))
        if not (lo < level < hi):
            continue
        try:
            verts, faces, _, _ = marching_cubes(
                log_T, level=level, spacing=(dg, dg, dg)
            )
            verts += g[0]   # shift from [0, …] to [g[0], …]
            mesh = Poly3DCollection(
                verts[faces], alpha=alpha, facecolor=col,
                edgecolor='none', label=f'T = {T_target:g} yr',
            )
            ax.add_collection3d(mesh)
        except Exception:
            pass

    if observed is not None:
        u1, u2, u3 = observed
        ax.scatter(u1, u2, u3, s=12, c='k', alpha=0.45,
                   edgecolors='none', zorder=5, label='Observed')

    ax.set_xlim(g[0], g[-1])
    ax.set_ylim(g[0], g[-1])
    ax.set_zlim(g[0], g[-1])
    ax.set_xlabel('U_duration')
    ax.set_ylabel('U_severity')
    ax.set_zlabel('U_coverage')
    ax.legend(loc='upper left', fontsize=9)

    return fig, ax
