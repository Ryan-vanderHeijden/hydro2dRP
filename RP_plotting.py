import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np


# Plotting Utilities
def colored_line(
    x, y, c,
    cmap='viridis',
    norm=None,
    linewidth=2,
):
    '''
    ChatGPT assisted 1/3/2026
    Create a LineCollection with color varying along the line.

    Masks NaN and ±Inf in x, y, and c.
    Segments touching invalid values are dropped (gaps preserved).

    Parameters
    ----------
    x, y : array-like, shape (N,)
        Line coordinates.
    c : array-like, shape (N,)
        Values used for coloring (one per point).
    cmap : str or Colormap
        Matplotlib colormap.
    norm : matplotlib.colors.Normalize, optional
        Color normalization. If None, uses min/max of valid c.
    linewidth : float
        Line width.

    Returns
    -------
    lc : LineCollection
        Configured LineCollection object.
    '''

    x = np.asarray(x)
    y = np.asarray(y)
    c = np.asarray(c)

    if not (x.shape == y.shape == c.shape):
        raise ValueError('x, y, and c must have the same shape')

    # Finite mask (NaN + ±Inf)
    finite = np.isfinite(x) & np.isfinite(y) & np.isfinite(c)

    # Build segments
    points = np.column_stack([x, y]).reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    # Keep segments whose endpoints are valid
    seg_mask = finite[:-1] & finite[1:]
    segments = segments[seg_mask]
    colors = c[:-1][seg_mask]

    if norm is None:
        norm = plt.Normalize(colors.min(), colors.max())

    lc = LineCollection(
        segments,
        cmap=cmap,
        norm=norm,
        linewidth=linewidth,
    )
    lc.set_array(colors)

    return lc