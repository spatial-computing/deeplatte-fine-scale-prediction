import numpy as np
import math
from sklearn.metrics import r2_score


def compute_error(ground_truth, prediction):
    rmse = r_m_s_e_error(ground_truth, prediction)
    r2 = r_2_error(ground_truth, prediction)
    mape = m_a_p_e_error(ground_truth, prediction)
    return rmse, mape, r2


def r_2_error(ground_truth, prediction):
    return r2_score(ground_truth[~np.isnan(ground_truth)], prediction[~np.isnan(ground_truth)])


def m_a_p_e_error(ground_truth, prediction):
    return np.nanmean(np.abs((ground_truth - prediction) / ground_truth)) * 100


def r_m_s_e_error(ground_truth, prediction):
    mse = np.nanmean((ground_truth - prediction) ** 2)
    return math.sqrt(mse)


class Norm:

    def __init__(self, x):
        self.mean = np.nanmean(x, axis=0)
        self.std = np.nanstd(x, axis=0)

    def transform(self, x):
        return np.divide(x - self.mean, self.std, out=np.zeros_like(x - self.mean), where=self.std != 0.0)


def normalize_mat(input_mat, if_retain_last_dim=True):
    """ normalize the input feature matrix

    Params:
        feature_mat: (n_times, n_features, n_rows, n_cols)
        if_retain_last_dim: if keeping the last dimension as original
    Return:
        norm_mat: (n_times, n_features, n_rows, n_cols)
    """

    n_times, n_features, n_rows, n_cols = input_mat.shape
    x_2d = np.moveaxis(input_mat, 1, -1)  # => (n_times, n_rows, n_cols, n_features)
    x_2d = x_2d.reshape(-1, n_features)  # => (n_samples, n_features)
    norm = Norm(x_2d)
    norm_mat = norm.transform(x_2d)

    norm_mat = norm_mat.reshape(n_times, n_rows, n_cols, n_features)  # => (n_times, n_rows, n_cols, n_features)
    if if_retain_last_dim:
        norm_mat = np.moveaxis(norm_mat, -1, 1)  # => (n_times, n_features, n_rows, n_cols)

    # print('Shape of the normalized matrix = {}'.format(norm_mat.shape))
    return norm_mat
