import datetime
import os
from importlib import reload

from models.deep_ap import DeepAP
from scripts.data_loader import *
from scripts.train_dap import train
from utils.metrics import normalize_mat


def main(args, **kwargs):

    """ extract information from target time period """
    data_file = os.path.join(kwargs['data_dir'],
                             '{}_{}m_{}_{}.npz'.format(args.area, args.resolution, args.year, args.tar_date))
    data = np.load(data_file)
    tar_label_mat = data['label_mat']
    tar_dynamic_mat = data['dynamic_mat']
    static_mat, tar_static_mat = data['static_mat'], data['static_mat']
    mapping_mat = data['mapping_mat']
    dynamic_features, static_features = list(data['dynamic_features']), list(data['static_features'])

    """ extract information from all time period """
    dynamic_mat, label_mat = [], []
    for date in args.dates:
        data_file = os.path.join(kwargs['data_dir'],
                                 '{}_{}m_{}_{}.npz'.format(args.area, args.resolution, args.year, date))
        data = np.load(data_file)
        dynamic_mat.append(data['dynamic_mat'])
        label_mat.append(data['label_mat'])
    dynamic_mat = np.concatenate(dynamic_mat)
    label_mat = np.concatenate(label_mat)

    data_obj = DataObj(label_mat, dynamic_mat, static_mat,
                       tar_label_mat, tar_dynamic_mat, tar_static_mat,
                       dynamic_features, static_features, mapping_mat)

    """ load train, val, test locations """
    data_obj.train_loc, data_obj.val_loc, data_obj.test_loc = load_train_val_test(kwargs['train_val_test_file'], args)

    data_obj.train_y = data_obj.gen_train_val_test_label(data_obj.label_mat, data_obj.train_loc)
    data_obj.val_y = data_obj.gen_train_val_test_label(data_obj.label_mat, data_obj.val_loc)
    data_obj.test_y = data_obj.gen_train_val_test_label(data_obj.tar_label_mat, data_obj.test_loc)

    logging.info('Number of features = {}.'.format(data_obj.n_features))
    logging.info('Number of dynamic features = {}.'.format(data_obj.n_dynamic_features))
    logging.info('Number of static features = {}.'.format(data_obj.n_static_features))
    logging.info('Number of time points = {}.'.format(data_obj.n_times))
    logging.info('Shape of the matrix = ({}, {}).'.format(data_obj.n_rows, data_obj.n_cols))

    """ normalize data """
    data_obj.dynamic_x = normalize_mat(data_obj.dynamic_mat, if_retain_last_dim=True)
    data_obj.static_x = normalize_mat(data_obj.static_mat, if_retain_last_dim=True)
    data_obj.tar_dynamic_x = normalize_mat(data_obj.tar_dynamic_mat, if_retain_last_dim=True)
    data_obj.tar_static_x = normalize_mat(data_obj.tar_static_mat, if_retain_last_dim=True)

    """ load auto-encoder model """
    ae = torch.load(os.path.join(kwargs['model_dir'], kwargs['ae_model_name'] + '.pkl'))

    """ define DeepAP model """
    dap = DeepAP(in_dim=data_obj.n_features,
                 ae_en_h_dims=[64, 32, 16],
                 ae_de_h_dims=[16, 32, 64],

                 conv_lstm_in_size=(data_obj.n_rows, data_obj.n_cols),
                 conv_lstm_in_dim=args.ae_h_dim,  # ae_h_dim
                 conv_lstm_h_dim=[args.dap_h_dim],  # dap_h_dim
                 conv_lstm_kernel_sizes=args.kernel_sizes,  # kernel_sizes
                 conv_lstm_n_layers=1,

                 fc_in_dim=args.dap_h_dim * len(args.kernel_sizes),
                 fc_h_dims=args.fc_h_dims,  # fc_h_dims
                 fc_out_dim=1,

                 ae_pretrain_weight=ae.state_dict(),
                 if_trainable=True,
                 fc_p_dropout=0.1,

                 mask_thre=args.mask_thr,
                 device=kwargs['device'])

    dap = dap.to(kwargs['device'])
    train(dap, data_obj, args, **kwargs)


class Param:
    def __init__(self):

        """ default configuration """
        self.area = 'los_angeles'
        self.resolution = 500
        self.year = 2018
        self.dates = ['01']
        self.tar_date = '01'

        """ training configuration """
        self.epochs = 150
        self.lr = 0.001
        self.batch_size = 4
        self.ae_h_dim = 16
        self.dap_h_dim = 32
        self.fc_h_dims = [64, 16]
        self.kernel_sizes = [1, 3]

        """ hyper parameters """
        self.sp_neighbor = 1
        self.tp_neighbor = 1
        self.seq_len = 6
        self.mask_thr = 0.0001
        self.alpha = 2
        self.beta = 0.1
        self.gamma = 5

    def if_dates_consistent(self):
        if self.tar_date not in self.dates:
            print('The target date is not in the provided dates')
        else:
            pass

    def generate_model_name(self):
        dap_config = ['dap', '_',
                      str(self.area),
                      str(self.resolution) + 'm',
                      str(self.year), '_',
                      '_'.join(self.dates), '_',
                      self.tar_date, '_',
                      str(self.seq_len),
                      str(self.sp_neighbor),
                      str(self.mask_thr).replace('.', ''), '_',
                      str(self.alpha).replace('.', ''),
                      str(self.beta).replace('.', ''),
                      str(self.gamma).replace('.', ''), '_',
                      str(self.ae_h_dim),
                      str(self.dap_h_dim),
                      ''.join([str(i) for i in self.fc_h_dims]),
                      ''.join([str(i) for i in self.kernel_sizes])]
        return '_'.join(dap_config)

    def generate_ae_model_name(self):
        ae_config = ['ae', '_',
                     str(self.area),
                     str(self.resolution) + 'm',
                     str(self.year), '_',
                     '_'.join(self.dates), '_',
                     str(self.ae_h_dim)]
        return '_'.join(ae_config)


if __name__ == '__main__':

    """ loop over hyper-parameters """
    param = Param()

    # define directory
    base_dir = 'data/'
    data_dir = '/home/yijun/notebooks/training_data/los_angeles_500m_2018/'
    train_val_test_file = 'data/data/los_angeles_500m_train_val_test.json'
    model_dir = os.path.join(base_dir, 'models/')
    log_dir = os.path.join(base_dir, 'logs/')
    run_dir = os.path.join(base_dir, 'runs/')
    result_dir = os.path.join(base_dir, 'results/')

    device = torch.device("cuda:3" if torch.cuda.is_available() else 'cpu')  # the gpu device

    kwargs = {
        'data_dir': data_dir,
        'model_dir': model_dir,
        'train_val_test_file': train_val_test_file,
        'device': device
    }

    d = ['10', '11', '12']
    t = '09'
    s = 6
    a = 1
    b = 0.01
    g = 1

    param.dates, param.tar_date, param.seq_len, param.alpha, param.beta, param.gamma = d, t, s, a, b, g
    param.if_dates_consistent()

    # define model name
    model_name = param.generate_model_name()
    ae_model_name = param.generate_ae_model_name()

    # define the file path
    model_file = os.path.join(model_dir, model_name + '.pkl')
    log_file = os.path.join(log_dir, model_name + '.log')
    run_file = os.path.join(run_dir, model_name + '_run_{}'.format(datetime.datetime.now().strftime('%d%H%m')))

    kwargs['model_name'] = model_name
    kwargs['ae_model_name'] = ae_model_name
    kwargs['model_file'] = model_file
    kwargs['log_file'] = log_file
    kwargs['run_file'] = run_file

    logger = logging.getLogger()
    logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s %(message)s', datefmt='%m-%d %H:%M')
    logging.info('{} STARTS.'.format(model_name))
    main(param, **kwargs)
    logging.info('{} ENDS.'.format(model_name))

    logging.shutdown()
    reload(logging)
