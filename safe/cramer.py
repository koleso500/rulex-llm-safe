import numpy as np

def lorenz_curve(y):
    """
    Compute the Lorenz curve for a given array.

    Parameters
    ----------
    y : array-like
        Input values

    Returns
    -------
    np.ndarray
        Normalized cumulative sum (Lorenz curve)
    """
    y = np.asarray(y, dtype=float).reshape(-1)
    y = y[~np.isnan(y)]
    if len(y) == 0:
        return np.array([])
    y_sorted = np.sort(y)
    cum = np.cumsum(y_sorted)
    sum_y = cum[-1]
    if sum_y == 0:
        return np.full_like(cum, np.nan)
    return cum / sum_y


def concordance_curve(y, yhat):
    """
    Compute the concordance curve between true and predicted values.

    Parameters
    ----------
    y : array-like
        True values
    yhat : array-like
        Predicted values

    Returns
    -------
    np.ndarray
        Concordance curve
    """
    y = np.asarray(y, dtype=float).reshape(-1)
    yhat = np.asarray(yhat, dtype=float).reshape(-1)
    mask = ~np.isnan(y) & ~np.isnan(yhat)
    y = y[mask]
    yhat = yhat[mask]

    if len(y) == 0:
        return np.array([])

    ord_idx = np.argsort(yhat, kind='stable')
    cum = np.cumsum(y[ord_idx])
    return cum / cum[-1]


def gini_via_lorenz(y):
    """
    Calculate Gini coefficient.

    Parameters
    ----------
    y : array-like
        Input values

    Returns
    -------
    float
        Gini coefficient
    """
    y = np.asarray(y, dtype=float).reshape(-1)
    l = lorenz_curve(y)
    n = len(l)
    if n == 0:
        return np.nan
    u = np.linspace(1 / n, 1, n)
    return 2 * np.mean(np.abs(u - l))


def cvm1_concordance_weighted(y, yhat):
    """
    Weighted Cramer von Mises distance between Lorenz and Concordance curves.

    Parameters
    ----------
    y : array-like
        True values
    yhat : array-like
        Predicted values

    Returns
    -------
    float
        Weighted CvM distance
    """
    y = np.asarray(y, dtype=float).reshape(-1)
    yhat = np.asarray(yhat, dtype=float).reshape(-1)
    mask = ~np.isnan(y) & ~np.isnan(yhat)
    y = y[mask]
    yhat = yhat[mask]

    n = len(y)
    if n == 0:
        return np.nan

    # Lorenz curve
    ord_y = np.argsort(y, kind='stable')
    l = np.cumsum(y[ord_y]) / np.sum(y)

    # Concordance curve
    ord_yhat = np.argsort(yhat, kind='stable')
    c = np.cumsum(y[ord_yhat]) / np.sum(y)

    # Weights
    weights = y[ord_y] / np.sum(y)

    return np.sum(np.abs(c - l) * weights)