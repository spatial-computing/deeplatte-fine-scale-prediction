import os
import logging

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class DataObj:

    def __init__(self, label_mat, dynamic_x, static_x,
                 dynamic_feature_names, static_feature_names,
                 mapping_mat, **kwargs):

        self.label_mat = label_mat
        self.dynamic_x = self.norm(dynamic_x)
        self.static_x = self.norm(static_x)
    
        self.dynamic_feature_names = dynamic_feature_names
        self.static_feature_names = static_feature_names
        self.features_names = dynamic_feature_names + static_feature_names
        self.mapping_mat = mapping_mat

        self.num_features = len(self.dynamic_feature_names) + len(self.static_feature_names)
        self.num_dynamic_features = len(self.dynamic_feature_names)
        self.num_static_features = len(self.static_feature_names)
        self.num_times, _, self.num_rows, self.num_cols = self.dynamic_x.shape

        # set up training
        self.use_test = kwargs.get('use_test', False)
        self.train_size = kwargs.get('train_size', 0.8)
        self.test_size = kwargs.get('test_size', 0.2) * self.use_test

        self.train_loc, self.val_loc, self.test_loc = self.split_train_val_test_locations()
        self.train_y = self.set_target_label_mat(self.train_loc)
        self.val_y = self.set_target_label_mat(self.val_loc)
        self.test_y = self.set_target_label_mat(self.test_loc)

    def set_target_label_mat(self, locations):
        """ return a new label mat containing only given locations """

        mat = np.full(self.label_mat.shape, np.nan)
        for loc in locations:
            r, c = np.where(self.mapping_mat == loc)
            mat[..., r[0], c[0]] = self.label_mat[..., r[0], c[0]]
        return mat

    def split_train_val_test_locations(self):
        """ split labeled locations into train, val, (test) set

        return: train_loc, val_loc, test_loc -> list, list, list """

        #  find locations that have more than 0.01 x num_times labels
        candidate_mat = np.sum(~np.isnan(self.label_mat), axis=(0, 1)) == self.num_times

        sub_region_locations = [
            self.mapping_mat[np.where(candidate_mat[0:self.num_rows // 2, 0:self.num_cols // 2])].tolist(),
            self.mapping_mat[np.where(candidate_mat[0:self.num_rows // 2, self.num_cols // 2:])].tolist(),
            self.mapping_mat[np.where(candidate_mat[self.num_rows // 2:, 0:self.num_cols // 2])].tolist(),
            self.mapping_mat[np.where(candidate_mat[self.num_rows // 2:, self.num_cols // 2:])].tolist()
        ]

        train_loc, val_loc, test_loc = [], [], []
        for loc in sub_region_locations:
            if self.use_test:
                train, test = train_test_split(loc, test_size=self.test_size, random_state=1234)
                train, val = train_test_split(train, train_size=self.train_size, random_state=1234)
                test_loc += test_loc
            else:
                train, val = train_test_split(loc, train_size=self.train_size, random_state=1234)
            train_loc += train
            val_loc += val

        return sorted(train_loc), sorted(val_loc), sorted(test_loc)

    @staticmethod
    def norm(mat):
        num_times, num_features, num_rows, num_cols = mat.shape
        mat_2d = np.moveaxis(mat, 1, -1)  # shape: (num_times, num_rows, num_cols, num_features)
        mat_2d = mat_2d.reshape(-1, num_features)  # shape: (num_samples, num_features)
        norm_mat = StandardScaler().fit_transform(mat_2d)
        norm_mat = norm_mat.reshape(num_times, num_rows, num_cols, num_features)
        norm_mat = np.moveaxis(norm_mat, -1, 1)  # shape: (num_times, num_features, num_rows, num_cols)
        return norm_mat


def load_data_from_db(args):
    pass


def load_data_from_file(data_path):
    """ load data from a file """
    
    if not os.path.isfile(data_path):
        raise FileNotFoundError

    data = np.load(data_path)
    dynamic_feature_names, static_feature_names = list(data['dynamic_features']), list(data['static_features'])
    data_obj = DataObj(label_mat=data['label_mat'],
                       dynamic_x=data['dynamic_mat'],
                       static_x=data['static_mat'],
                       dynamic_feature_names=dynamic_feature_names,
                       static_feature_names=static_feature_names,
                       mapping_mat=data['mapping_mat'])
    return data_obj




def data_logging(data_obj):
    logging.info('Number of features = {}.'.format(data_obj.n_features))
    logging.info('Number of dynamic features = {}.'.format(data_obj.n_dynamic_features))
    logging.info('Number of static features = {}.'.format(data_obj.n_static_features))
    logging.info('Number of time points = {}.'.format(data_obj.n_times))
    logging.info('Shape of the matrix = ({}, {}).'.format(data_obj.n_rows, data_obj.n_cols))
    logging.info('Number of given pm locations = {}.'.format(len(data_obj.train_loc + data_obj.val_loc + data_obj.test_loc)))
    logging.info('Number of training locations = {}.'.format(len(data_obj.train_loc)))
    logging.info('Number of validation locations = {}.'.format(len(data_obj.val_loc)))
    logging.info('Number of testing locations = {}.'.format(len(data_obj.test_loc)))

    