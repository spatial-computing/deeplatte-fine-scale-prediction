import argparse
import logging
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

    parser.add_argument('--data_path', type=str, default='./backup/data/los_angeles_500m_2020_03.npz',
                        help='data path, if not provided, load data from database')
    parser.add_argument('--result_dir', type=str, default='./backup/results/',
                        help='result directory')

    parser.add_argument('--model_path', type=str, default='./backup/sample/', help='model path')
    parser.add_argument('--model_name', type=str, default='los_angeles_1000', help='model name')

    parser.add_argument('--log_dir', type=str, help='')
    parser.add_argument('--use_log', action='store_true', help='default: False, visualize loss in tensor board')
    parser.add_argument('--verbose', action='store_true', help='default: False, print more debugging information')

    # training parameters
    parser.add_argument('--device', type=str, default='3', help='GPU id')
    parser.add_argument('--num_epochs', type=int, default=200, help='number of epochs for training')
    parser.add_argument('--batch_size', type=int, default=4, help='input batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='initial learning rate for adam')

    # model parameters
    parser.add_argument('--seq_len', type=int, default=6, help='sequence length for rnn')

    # parser.add_argument('--self.ae_en_h_dims', type=int, default=6, help='sequence length for rnn')
    # parser.add_argument('--self.ae_de_h_dims', type=int, default=6, help='sequence length for rnn')
    # parser.add_argument('--self.dapm_h_dim', type=int, default=6, help='sequence length for rnn')
    # parser.add_argument('--self.kernel_sizes', type=int, default=6, help='sequence length for rnn')
    # parser.add_argument('--self.fc_h_dims', type=int, default=6, help='sequence length for rnn')
    #
    # # hyper parameters """
    # parser.add_argument('--sp_neighbor', type=int, default=6, help='sequence length for rnn')
    # parser.add_argument('--l1_thr', type=float, default=0.0001, help='the threshold for L1 regularization')
    # parser.add_argument('--alpha', type=float, default=1, help='the weight for L1 regularization loss')
    # parser.add_argument('--beta', type=float, default=0.1, help='the weight for auto-encoder')
    # parser.add_argument('--gamma', type=float, default=5, help='')
    # parser.add_argument('--eta', type=float, default=0.01, help='')

    # others
    parser.add_argument('--use_tb', action='store_true', help='default: False')
    parser.add_argument('--tb_path', type=str, default='', help='tensor board path')
    parser.add_argument('--verbose', action='store_false', help='if logging the loss')

    args = parser.parse_args()
    input_check(args)
    return args


def verbose(args):

    if args.verbose:
        log_file = os.path.join(args.res_path, f'{args.model_name}_{time.time() // 1000}.log')
        logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s %(message)s',
                            datefmt='%m-%d %H:%M')
        logging.info('args.')
        logging.info('args.')
        logging.info('args.')
        logging.info('args.')


def input_check(args):

    # check result path existence
    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)

