import argparse
import sys
import time
import os


def parse_args():
    """ this class includes training options """

    # basic parameters
    parser = argparse.ArgumentParser(description='DeepLatte')

    parser.add_argument('--area', type=str, default='los_angeles', help='target area')
    parser.add_argument('--resolution', type=int, default=1000, help='target resolution of the grids')
    parser.add_argument('--min_time', type=str, default='2018-01-01', help='target start time')
    parser.add_argument('--max_time', type=str, default='2018-02-01', help='target end time')

    tmp = '/Users/yijunlin/Research/PRISMS2020/training_data/los_angeles_500m_2020/los_angeles_500m_2020_03.npz'
    parser.add_argument('--data_path', type=str, default=tmp,
                        help='data path, if not provided, load data from database')
    parser.add_argument('--result_path', type=str, default='./sample_data/results/',
                        help='result directory')

    parser.add_argument('--model_path', type=str, default='./sample_data/results/', help='model path')
    parser.add_argument('--model_name', type=str, default='los_angeles_1000', help='model name')

    # training parameters
    parser.add_argument('--device', type=str, default='3', help='GPU id')

    # model parameters
    parser.add_argument('--seq_len', type=int, default=6, help='sequence length for rnn')
    parser.add_argument('--en_features', type=str, default='64,16', help='encoder sizes')
    parser.add_argument('--de_features', type=str, default='16,64', help='decoder sizes')
    parser.add_argument('--kernel_sizes', type=str, default='1,3,5', help='kernel sizes for convolution operation')
    parser.add_argument('--h_channels', type=int, default=32, help='number of channels for convolution operation')
    parser.add_argument('--fc_h_features', type=int, default=32, help='hidden size for the fully connected layer')

    args = parser.parse_args()
    input_check(args)
    return args


def input_check(args):

    # check result path existence
    if not os.path.exists(args.result_path):
        print('Result path does not exist.')
        sys.exit(-1)

    # check model path existence
    model_file = os.path.join(args.model_path, args.model_name + '.pkl')
    if not os.path.exists(model_file):
        print('Model file does not exist.')
        sys.exit(-1)

