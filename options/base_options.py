import argparse
import torch
import os


class BaseOptions:
    """ this class defines options used during bither training and testing time """

    def __init__(self):
        self.args = None

    def initialize(self, parser):
        """ define the common options that are used in both train and test set """

        # basic parameters
        parser.add_argument('--area', type=str, default='los_angeles', help='target area')
        parser.add_argument('--resolution', type=int, default=1000, help='target resolution of the grids')
        parser.add_argument('--min_time', type=str, default='2020-12-01-00', help='target start time')
        parser.add_argument('--max_time', type=str, default='2021-02-28-23', help='target end time')

        parser.add_argument('--model_dir', type=str, help='Model name')
        parser.add_argument('--log_dir', type=str, help='Model path')
        parser.add_argument('--run_dir', action='store_const', const=True, default=False, help='If logging the loss')
        parser.add_argument('--train_val_test', type=int, default=2, help='Number of Classes')

        # model parameters
        parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')
        parser.add_argument('--model_type', type=str, default='', help='')
        parser.add_argument('--batch_size', type=int, default=4, help='input batch size')

        """ hyper parameters """
        parser.add_argument('--sp_neighbor', type=int, default=6, help='sequence length for rnn')
        parser.add_argument('--seq_len', type=int, default=6, help='sequence length for rnn')
        parser.add_argument('--self.ae_en_h_dims', type=int, default=6, help='sequence length for rnn')
        parser.add_argument('--self.ae_de_h_dims', type=int, default=6, help='sequence length for rnn')
        parser.add_argument('--self.dapm_h_dim', type=int, default=6, help='sequence length for rnn')
        parser.add_argument('--self.kernel_sizes', type=int, default=6, help='sequence length for rnn')
        parser.add_argument('--self.fc_h_dims', type=int, default=6, help='sequence length for rnn')

        parser.add_argument('--device', type=str, default='3', help='GPU id')
        parser.add_argument('--verbose', action='store_true', help='if specified, print more debugging information')

        return parser

    def print_options(self, parser,args):
        """ print and save options

        It will print both current options and default values(if different).
        It will save options into a text file / [checkpoints_dir] / opt.txt
        """
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(args).items()):
            comment = ''
            default = parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)

        # save to the disk
        # expr_dir = os.path.join(args['checkpoints_dir'], args['name'])
        # util.mkdirs(expr_dir)
        # file_name = os.path.join(expr_dir, '{}_opt.txt'.format(args.phase))
        # with open(file_name, 'wt') as opt_file:
        #     opt_file.write(message)
        #     opt_file.write('\n')

    def parse(self, parser):
        args = parser.parse_args()
        args.device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')  # the gpu device
        self.print_options(parser,args)

        self.args = vars(args)
        return self.args
