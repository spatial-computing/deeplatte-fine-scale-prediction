from options.base_options import BaseOptions


class TrainOptions(BaseOptions):
    """ this class includes training options and the shared options in the BaseOptions """

    def initialize(self, parser):

        parser = BaseOptions.initialize(self, parser)
        parser.add_argument('--num_epochs', type=int, default=150, help='number of epochs for training')
        parser.add_argument('--lr', type=float, default=0.001, help='initial learning rate for adam')

        parser.add_argument('--l1_thr', type=float, default=0.0001, help='the threshold for L1 regularization')
        parser.add_argument('--alpha', type=float, default=1, help='the weight for L1 regularization loss')
        parser.add_argument('--beta', type=float, default=0.1, help='the weight for auto-encoder')
        parser.add_argument('--gamma', type=float, default=5, help='')
        parser.add_argument('--eta', type=float, default=0.01, help='')

        return parser
