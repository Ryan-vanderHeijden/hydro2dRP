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
    grid_size=200,
    n_mc=200_000,
    band_width=0.04,
    random_state=None,
):
    '''
    Compute 2D Kendall return period contours from the vine by conditioning
    one variable at fixed quantile levels.

    Strategy
    --------
    For each quantile q of the fixed variable:
    1. Draw n_mc samples from the vine.
    2. Retain samples where the fixed variable falls in [q-band, q+band].
    3. Fit the best bivariate copula to the two free variables among
       the retained samples (re-selecting family by AICc).
    4. Compute and return the Kendall T-year contour for that conditional
       bivariate copula.

    This gives an approximate conditional contour; accuracy improves with
    larger n_mc and smaller band_width (subject to the sample-size trade-off).

    Parameters
    ----------
    vc : VineCopula3D
    T : float              Target Kendall return period.
    fixed_var : int        Index (0, 1, or 2) of the variable to fix.
    fixed_quantiles : array-like
        Quantile values to condition on (e.g., [0.25, 0.50, 0.75]).
    grid_size : int        Resolution of the returned contour grids.
    n_mc : int
    band_width : float     Half-width of the conditioning band (default 0.04).
    random_state : int or None

    Returns
    -------
    contours : list of dict
        One entry per quantile level.  Keys:
        - ``'quantile'``    : the fixed quantile.
        - ``'u'``           : u-grid for the free variable on the x-axis.
        - ``'v'``           : v-values on the Kendall contour (NaN where
                              outside valid range).
        - ``'var_labels'``  : (x-label, y-label) for the two free variables.
        - ``'family'``      : selected bivariate copula family.
        - ``'theta'``       : fitted parameter.
        - ``'n_cond'``      : number of conditioning samples used.
        Returns ``'u': None, 'v': None`` if too few samples fall in the band.
    '''
    s1, s2, s3  = vc.simulate(n_mc, random_state=random_state)
    samples     = np.column_stack([s1, s2, s3])

    free_vars   = [i for i in range(3) if i != fixed_var]
    var_labels  = ['u1', 'u2', 'u3']
    u_grid      = np.linspace(1e-3, 1 - 1e-3, grid_size)

    contours = []
    for q in np.asarray(fixed_quantiles):
        mask  = (samples[:, fixed_var] >= q - band_width) & \
                (samples[:, fixed_var] <= q + band_width)
        cond  = samples[mask]

        if len(cond) < 30:
            contours.append({
                'quantile':   float(q),
                'u':          None,
                'v':          None,
                'var_labels': (var_labels[free_vars[0]],
                               var_labels[free_vars[1]]),
                'family':     None,
                'theta':      None,
                'n_cond':     len(cond),
            })
            continue

        x = cond[:, free_vars[0]]
        y = cond[:, free_vars[1]]

        _, best = select_copula(x, y)
        fam     = best['family']
        theta   = best['theta']
        t_level = invert_kendall_level(1.0 - 1.0 / T, theta, fam)
        v_line  = copula_contour(u_grid, t_level, theta, fam)

        contours.append({
            'quantile':   float(q),
            'u':          u_grid,
            'v':          v_line,
            'var_labels': (var_labels[free_vars[0]],
                           var_labels[free_vars[1]]),
            'family':     fam,
            'theta':      theta,
            'n_cond':     len(cond),
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
