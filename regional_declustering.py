import pandas as pd
import numpy as np





def regional_concurrence_intervals(
    df,
    frac_thresh=0.2,
    buffer=7
):
    '''
    Identify regional drought events using buffered site-level intervals.

    Parameters
    ----------
    df : DataFrame
        Columns: ['start', 'end'] (index is 'site')
        start/end must be datetime-like
    frac_thresh : float
        Fraction of sites required for a regional event (e.g., 0.2)
    buffer_days : int
        Temporal buffer (days) added before start and after end

    Returns
    -------
    events : list of (start, end)
        Buffered regional event intervals
    '''

    sites = df.index.unique()
    n_sites = len(sites)
    k_min = int(np.ceil(frac_thresh * n_sites))

    buffer = pd.Timedelta(days=buffer)

    boundaries = []

    for _, row in df.iterrows():
        s = row.start - buffer
        e = row.end + buffer

        boundaries.append((s, +1))
        # decrement just after end to keep end inclusive
        boundaries.append((e + pd.Timedelta(seconds=1), -1))

    boundaries.sort(key=lambda x: x[0])

    events = []
    active = 0
    in_event = False

    for t, delta in boundaries:
        prev_active = active
        active += delta

        # Start of regional event
        if (not in_event) and active >= k_min:
            t_start = t
            in_event = True

        # End of regional event
        if in_event and active < k_min:
            t_end = t
            events.append((t_start, t_end))
            in_event = False

    return events




def regional_metrics_from_intervals(
    df,
    events,
    severity_method='sum',
    duration_method='union'
):
    '''
    Compute regional event severity and duration from site-level intervals.

    Parameters
    ----------
    df : DataFrame
        Columns: ['site', 'start', 'end', 'severity']
    events : list of (start, end)
        Regional event intervals (typically buffered for detection)
    severity_method : {'sum', 'mean', 'max'}
        Aggregation of site-level severities
    duration_method : {'union', 'mean', 'max'}
        Aggregation of site-level durations

    Returns
    -------
    durations : ndarray
        Regional durations
    severities : ndarray
        Regional severities
    
    
    Guidance on aggregation methods:
        | Severity | Duration | Interpretation                                |
        | -------- | -------- | --------------------------------------------- |
        | sum      | union    | Total regional drought load (most common)     |
        | mean     | mean     | Typical site drought                          |
        | max      | max      | Worst-case site behavior                      |
        | sum      | max      | Severe footprint + persistence                |

    
    '''

    if severity_method not in {'sum', 'mean', 'max'}:
        raise ValueError("severity_method must be 'sum', 'mean', or 'max'")

    if duration_method not in {'union', 'mean', 'max'}:
        raise ValueError("duration_method must be 'union', 'mean', or 'max'")

    durations = []
    severities = []

    for t0, t1 in events:

        # overlapping site-level droughts
        mask = (df.start <= t1) & (df.end >= t0)
        overlapping = df.loc[mask]

        if overlapping.empty:
            continue

        # ── Duration aggregation ──
        if duration_method == 'union':
            # full regional envelope duration
            duration = (t1 - t0).days + 1

        else:
            site_durations = (
                (overlapping.end.clip(upper=t1) -
                 overlapping.start.clip(lower=t0))
                .dt.days + 1
            )

            if duration_method == 'mean':
                duration = site_durations.mean()

            elif duration_method == 'max':
                duration = site_durations.max()

        # ── Severity aggregation ──
        if severity_method == 'sum':
            severity = overlapping['severity'].sum()

        elif severity_method == 'mean':
            severity = overlapping['severity'].mean()

        elif severity_method == 'max':
            severity = overlapping['severity'].max()

        durations.append(duration)
        severities.append(severity)

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
