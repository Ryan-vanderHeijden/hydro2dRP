import pandas as pd
import numpy as np




# ── Temporal Clustering ───────────────────────────────────────────────────────

def event_midpoint(start, end):
    '''
    Compute the midpoint datetime of an event interval.

    Parameters
    ----------
    start, end : datetime-like

    Returns
    -------
    midpoint : datetime-like
    '''
    return start + (end - start) / 2




def temporal_clustering(events, delta_t):
    '''
    Cluster site-level drought events into independent regional events
    using a fixed reference time for each cluster.

    Events are sorted by their midpoint time. A new cluster starts whenever
    the current event's midpoint falls more than delta_t after the reference
    time of the previous cluster.

    Parameters
    ----------
    events : pandas.DataFrame
        Must contain an 'event_time' column (e.g., event midpoint) and be
        sorted by 'event_time'.
    delta_t : pandas.Timedelta
        Maximum separation between events in the same cluster.

    Returns
    -------
    clusters : list of pandas.DataFrame
        One DataFrame per cluster.
    '''
    clusters = []
    current_cluster = [events.iloc[0]]
    t_ref = events.iloc[0]['event_time']

    for i in range(1, len(events)):
        t_cur = events.iloc[i]['event_time']

        if (t_cur - t_ref) <= delta_t:
            current_cluster.append(events.iloc[i])
        else:
            clusters.append(pd.DataFrame(current_cluster))
            current_cluster = [events.iloc[i]]
            t_ref = t_cur

    clusters.append(pd.DataFrame(current_cluster))
    return clusters




def aggregate_cluster(cluster, duration_rule='max', severity_rule='max'):
    '''
    Aggregate a cluster of site-level drought events into one regional event.

    Parameters
    ----------
    cluster : pandas.DataFrame
        Must contain 'severity' and 'duration' columns.
    duration_rule : {'max', 'sum', 'mean', 'median'}
        Aggregation rule for duration.
    severity_rule : {'max', 'sum', 'mean', 'median'}
        Aggregation rule for severity.

    Returns
    -------
    D : float
        Aggregated regional duration.
    S : float
        Aggregated regional severity.
    '''
    rules = {'max', 'sum', 'mean', 'median'}

    if severity_rule not in rules:
        raise ValueError(f"severity_rule must be one of {rules}")
    if duration_rule not in rules:
        raise ValueError(f"duration_rule must be one of {rules}")

    S = getattr(cluster['severity'], severity_rule)()
    D = getattr(cluster['duration'], duration_rule)()

    return D, S




def regional_concurrence_intervals(
    df,
    frac_thresh=0.2,
    buffer=7,
    end_gap=7
):
    """
    Identify regional drought events using buffered site-level intervals,
    with persistence-based event termination.

    Parameters
    ----------
    df : DataFrame
        Columns: ['site', 'start', 'end']
    frac_thresh : float
        Fraction of sites required for a regional event (e.g., 0.2)
    buffer_days : int
        Temporal buffer applied to site intervals for concurrence detection
    end_gap_days : int
        Required continuous time below threshold to terminate an event

    Returns
    -------
    events : list of (start, end)
        Regional event intervals
    """

    sites = df.index.unique()
    n_sites = len(sites)
    k_min = int(np.ceil(frac_thresh * n_sites))

    buffer = pd.Timedelta(days=buffer)
    end_gap = pd.Timedelta(days=end_gap)

    # ── Build sweep-line boundaries with site identity ──
    boundaries = []

    for _, row in df.iterrows():
        s = row.start - buffer
        e = row.end + buffer

        boundaries.append((s, row.site, +1))
        boundaries.append((e + pd.Timedelta(seconds=1), row.site, -1))

    boundaries.sort(key=lambda x: x[0])

    active_sites = set()
    events = []

    in_event = False
    t_start = None
    t_below = None  # when concurrence first drops below threshold

    for t, site, delta in boundaries:

        if delta == +1:
            active_sites.add(site)
        else:
            active_sites.discard(site)

        n_active = len(active_sites)

        # ── Start condition ──
        if (not in_event) and n_active >= k_min:
            t_start = t
            in_event = True
            t_below = None

        # ── Below-threshold handling ──
        if in_event:
            if n_active < k_min:
                if t_below is None:
                    t_below = t
                elif t - t_below >= end_gap:
                    # terminate event
                    events.append((t_start, t_below))
                    in_event = False
                    t_start = None
                    t_below = None
            else:
                # recovered above threshold
                t_below = None

    # close trailing event
    if in_event:
        events.append((t_start, boundaries[-1][0]))

    return events




def regional_metrics_from_intervals(
    df,
    events,
    severity_method,
    duration_method
):
    """
    Compute regional event severity and duration from site-level intervals.

    Parameters
    ----------
    df : DataFrame
        Columns: ['site', 'start', 'end', 'severity']
    events : list of (start, end)
        Regional event intervals
    severity_method : {"sum", "mean", "max"}
        Aggregation of site-level severities
    duration_method : {"union", "mean", "max", "total"}
        Aggregation of site-level durations

    Returns
    -------
    durations : ndarray
        Regional durations
    severities : ndarray
        Regional severities
    """

    if severity_method not in {"sum", "mean", "max"}:
        raise ValueError("severity_method must be 'sum', 'mean', or 'max'")

    if duration_method not in {"union", "mean", "max", "total"}:
        raise ValueError(
            "duration_method must be 'union', 'mean', 'max', or 'total'"
        )

    durations = []
    severities = []

    for t0, t1 in events:

        # site droughts overlapping the regional event
        mask = (df.start <= t1) & (df.end >= t0)
        overlapping = df.loc[mask]

        if overlapping.empty:
            continue

        # ── Duration aggregation ──
        if duration_method == "union":
            duration = (t1 - t0).days + 1

        elif duration_method == "mean":
            duration = overlapping['duration'].mean()

        elif duration_method == "max":
            duration = overlapping['duration'].max()

        elif duration_method == "total":
            duration = overlapping['duration'].sum()

        # ── Severity aggregation ──
        if severity_method == "sum":
            severity = overlapping["severity"].sum()

        elif severity_method == "mean":
            severity = overlapping["severity"].mean()

        elif severity_method == "max":
            severity = overlapping["severity"].max()

        durations.append(int(duration))
        severities.append(int(severity))

    return np.asarray(durations), np.asarray(severities)





def reduce_clusters(cluster_id, severity):
    '''
    Select the most severe event per cluster.
    '''
    unique_clusters = np.unique(cluster_id)
    keep_idx = []

    for c in unique_clusters:
        idx = np.where(cluster_id == c)[0]
        i_max = idx[np.argmax(severity[idx])]
        keep_idx.append(i_max)

    return np.array(keep_idx)
