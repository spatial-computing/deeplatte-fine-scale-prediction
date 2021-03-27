import numpy as np
from scipy.optimize import curve_fit


def gaussian(h, r, s, n=0):
    return n + s * (1. - np.exp(- (h ** 2 / (r / 2.) ** 2)))


def exponential(h, r, s, n=0):
    return n + s * (1. - np.exp(-(h / (r / 3.))))


def get_fit_bounds(x, y):
    n = np.nanmin(y)
    r = np.nanmax(x)
    s = np.nanmax(y)
    return 0, [r, s, n]


def get_fit_func(x, y, model):
    try:
        bounds = get_fit_bounds(x, y)
        popt, _ = curve_fit(model, x, y, method='trf', p0=bounds[1], bounds=bounds)
        return popt
    except Exception as e:
        return [0, 0, 0]


def gen_semivariogram(distances, variances, bins, thr):

    valid_variances, valid_bins = [], []
    for b in range(len(bins) - 1):
        left, right = bins[b], bins[b + 1]
        mask = (distances >= left) & (distances < right)
        if np.count_nonzero(mask) > 10:
            v = np.nanmean(variances[mask])
            d = np.nanmean(distances[mask])
            valid_variances.append(v)
            valid_bins.append(d)

    x, y = np.array(valid_bins), np.array(valid_variances)
    popt = get_fit_func(x, y, model=gaussian)
    return popt


def compute_variogram_loss():
    pass