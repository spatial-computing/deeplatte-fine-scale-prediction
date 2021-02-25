from options.base_options import BaseOptions


class TestOptions(BaseOptions):
    """ this class includes training options and the shared options in the BaseOptions """

    def initialize(self, parser):

        parser = BaseOptions.initialize(self, parser)  # define shared options

        parser.add_argument('--results_dir', type=str, default='./results/', help='saves results here.')
        parser.add_argument('--num_test', type=int, default=50, help='how many test images to run')

        parser.set_defaults(model='test')
        return parser