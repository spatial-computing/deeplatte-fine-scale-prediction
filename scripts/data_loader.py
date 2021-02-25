import os
import json
import logging
import numpy as np
import torch.utils.data as dat
import torch

from utils.metrics import normalize_mat

class DataObj:

    def __init__(self, label_mat, dynamic_x, static_x, dynamic_features, static_features, mapping_mat):

        self.label_mat = label_mat
        self.dynamic_x = dynamic_x
        self.static_x = static_x
    
        self.dynamic_feature_names = dynamic_features
        self.static_feature_names = static_features
        self.features_names = dynamic_features + static_features
        self.mapping_mat = mapping_mat

        self.n_features = len(self.dynamic_feature_names) + len(self.static_feature_names)
        self.n_dynamic_features = len(self.dynamic_feature_names)
        self.n_static_features = len(self.static_feature_names)
        self.n_times, _, self.n_rows, self.n_cols = self.dynamic_x.shape

        """ initialization """
        self.train_loc, self.val_loc, self.test_loc = [], [], []
        self.train_y, self.val_y, self.test_y = None, None, None

    def gen_train_val_test_label(self, label_mat, locations):
        new_label_mat = np.full(label_mat.shape, np.nan)
        for loc in locations:
            r, c = np.where(self.mapping_mat == loc)
            new_label_mat[..., r[0], c[0]] = label_mat[..., r[0], c[0]]
        return new_label_mat


def load_data(data_dir, args, **kwargs):
    """ load data of current months and previous month to for generating complete prediction """
    
    dynamic_mat, label_mat = [], []
    
    """ load data for previous month if exists """
    data_file = os.path.join(data_dir, f'{args.area}_{args.resolution}m_{args.last_year}/{args.area}_{args.resolution}m_{args.last_year}_{args.last_month}.npz')
    if os.path.isfile(data_file):
        data = np.load(data_file)
        label_mat.append(data['label_mat'][-args.seq_len:, ...])
        dynamic_mat.append(data['dynamic_mat'][-args.seq_len:, ...])
    
    """ load data for current months """    
    for m in sorted(args.months):
        data_file = os.path.join(data_dir, f'{args.area}_{args.resolution}m_{args.year}/{args.area}_{args.resolution}m_{args.year}_{m}.npz')
        data = np.load(data_file)
        label_mat.append(data['label_mat'])
        dynamic_mat.append(data['dynamic_mat'])

    mapping_mat = data['mapping_mat']
    static_mat = data['static_mat']
    dynamic_features, static_features = list(data['dynamic_features']), list(data['static_features'])
    label_mat = np.concatenate(label_mat)
    dynamic_mat = np.concatenate(dynamic_mat)
    
    """ normalize data """
    if_retain_last_dim = True if kwargs.get('if_retain_last_dim') is None else kwargs['if_retain_last_dim']
    dynamic_x = normalize_mat(dynamic_mat, if_retain_last_dim=if_retain_last_dim)
    static_x = normalize_mat(static_mat, if_retain_last_dim=if_retain_last_dim)
    
    data_obj = DataObj(label_mat, dynamic_x, static_x, dynamic_features, static_features, mapping_mat)
    return data_obj


def load_locations(train_val_test, args):
    """ build train, val, test labels """
    
    val_loc = train_val_test[f'{args.year}-{args.tar_month}']['val_loc']
    test_loc = train_val_test[f'{args.year}-{args.tar_month}']['test_loc']
    
    # remove the selected testing locations from other dates
    locations = []
    for m in args.months:
        ym = f'{args.year}-{m}'
        locations += train_val_test[ym]['train_loc'] + train_val_test[ym]['val_loc'] + train_val_test[ym]['test_loc']
    train_loc = [i for i in locations if i not in val_loc and i not in test_loc]
    return train_loc, val_loc, test_loc


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

    