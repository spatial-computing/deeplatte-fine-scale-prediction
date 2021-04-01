import argparse
import logging
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
    parser.add_argument('--model_types', type=str, default='l1,ae,sp', help='model types')

    # training parameters
    parser.add_argument('--device', type=str, default='3', help='GPU id')
    parser.add_argument('--num_epochs', type=int, default=200, help='number of epochs for training')
    parser.add_argument('--batch_size', type=int, default=4, help='training batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='initial learning rate for adam')
    parser.add_argument('--weight_decay', type=float, default=0.001, help='the scalar for l2 loss')
    parser.add_argument('--patience', type=int, default=5, help='the patience for early stop')

    # model parameters
    parser.add_argument('--seq_len', type=int, default=6, help='sequence length for rnn')
    parser.add_argument('--en_features', type=str, default='64,16', help='encoder sizes')
    parser.add_argument('--de_features', type=str, default='16,64', help='decoder sizes')
    parser.add_argument('--kernel_sizes', type=str, default='1,3,5', help='kernel sizes for convolution operation')
    parser.add_argument('--h_channels', type=int, default=32, help='number of channels for convolution operation')
    parser.add_argument('--fc_h_features', type=int, default=32, help='hidden size for the fully connected layer')

    # # hyper parameters """
    parser.add_argument('--sp_neighbor', type=int, default=1, help='sequence length for rnn')
    parser.add_argument('--tp_neighbor', type=int, default=1, help='sequence length for rnn')
    parser.add_argument('--alpha', type=float, default=1, help='the scalar for l1 regularization loss')
    parser.add_argument('--beta', type=float, default=0.1, help='the scalar for auto-encoder loss')
    parser.add_argument('--gamma', type=float, default=5, help='the scalar for spatial neighboring loss')
    parser.add_argument('--eta', type=float, default=0.01, help='the scalar for')

    # others
    parser.add_argument('--use_tb', action='store_true', help='default: False, visualize loss in tensor board')
    parser.add_argument('--tb_path', type=str, default='', help='tensor board path')
    parser.add_argument('--verbose', action='store_false', help='default: False, print more debugging information')

    args = parser.parse_args()
    input_check(args)
    return args


def verbose(args):

    if args.verbose:
        log_file = os.path.join(args.result_path, f'{args.model_name}.log')
        logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s - %(message)s',
                            datefmt='%y-%m-%d %H:%M')
        logging.info(f'Data | area - {args.area}')
        logging.info(f'Data | resolution - {args.resolution}')
        logging.info(f'Data | min_time/max_time - {args.min_time}/{args.max_time}')
        logging.info(f'Model | model_name - {args.model_name}')


def input_check(args):

    # check result path existence
    if not os.path.exists(args.result_path):
        os.makedirs(args.result_path)

    # check model path existence
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)
    verbose(args)
