import json
import logging
import numpy as np
import torch.utils.data as dat
import torch


class DataObj:

    def __init__(self, label_mat, dynamic_mat, static_mat,
                 tar_label_mat, tar_dynamic_mat, tar_static_mat,
                 dynamic_features, static_features, mapping_mat):

        self.label_mat = label_mat
        self.dynamic_mat = dynamic_mat
        self.static_mat = static_mat

        self.tar_label_mat = tar_label_mat
        self.tar_dynamic_mat = tar_dynamic_mat
        self.tar_static_mat = tar_static_mat

        self.dynamic_feature_names = dynamic_features
        self.static_feature_names = static_features
        self.mapping_mat = mapping_mat

        self.n_features = len(self.dynamic_feature_names) + len(self.static_feature_names)
        self.n_dynamic_features = len(self.dynamic_feature_names)
        self.n_static_features = len(self.static_feature_names)
        self.n_times, _, self.n_rows, self.n_cols = self.label_mat.shape

        """ initialization """
        self.train_loc, self.val_loc, self.test_loc = [], [], []
        self.train_y, self.val_y, self.test_y = None, None, None
        self.dynamic_x, self.static_x = None, None  # input x after normalization
        self.tar_dynamic_x, self.tar_static_x = None, None  # input x after normalization

    def gen_train_val_test_label(self, label_mat, locations):
        new_label_mat = np.full(label_mat.shape, np.nan)
        for loc in locations:
            r, c = np.where(self.mapping_mat == loc)
            new_label_mat[..., r[0], c[0]] = label_mat[..., r[0], c[0]]
        return new_label_mat


def load_train_val_test(train_val_test_file, args):
    """ load the training, validation, and testing locations """

    f = open(train_val_test_file, 'r')
    train_val_test = json.loads(f.read())

    if len(args.dates) == 1:
        train_loc = train_val_test[args.tar_date]['train_loc']
        val_loc = train_val_test[args.tar_date]['val_loc']
        test_loc = train_val_test[args.tar_date]['test_loc']
    else:
        # treat the testing locations from the last date as the target testing locations
        val_loc = train_val_test[args.tar_date]['val_loc']
        test_loc = train_val_test[args.tar_date]['test_loc']

        # remove the selected testing locations from other dates
        train_loc = []
        for date in args.dates:
            train_loc += [i for i in train_val_test[date]['train_loc'] if i not in test_loc and i not in val_loc]

    train_loc = sorted(list(set(train_loc)))

    logging.info('Number of given pm locations = {}.'.format(len(train_loc + val_loc + test_loc)))
    logging.info('Number of training locations = {}.'.format(len(train_loc)))
    logging.info('Number of validation locations = {}.'.format(len(val_loc)))
    logging.info('Number of testing locations = {}.'.format(len(test_loc)))

    return train_loc, val_loc, test_loc
