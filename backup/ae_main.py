import datetime
import os

from models.auto_encoder import AutoEncoder
from scripts.data_loader import *
from scripts.pretrain_ae import train
from utils.metrics import normalize_mat


def main(args, **kwargs):

    """ load data object """
    tar_date = args.dates[-1]
    data_file = os.path.join(data_dir, '{}_{}m_{}_{}.npz'.format(args.area, args.resolution, args.year, tar_date))
    data = np.load(data_file)
    label_mat = data['label_mat']
    mapping_mat = data['mapping_mat']
    static_mat = data['static_mat']
    dynamic_features, static_features = list(data['dynamic_features']), list(data['static_features'])

    dynamic_mat = []
    for date in args.dates:
        data_file = os.path.join(data_dir, '{}_{}m_{}_{}.npz'.format(args.area, args.resolution, args.year, date))
        data = np.load(data_file)
        dynamic_mat.append(data['dynamic_mat'])

    dynamic_mat = np.concatenate(dynamic_mat)
    data_obj = DataObj(label_mat, dynamic_mat, static_mat, None, None, None,
                       dynamic_features, static_features, mapping_mat)

    """ normalize data """
    data_obj.dynamic_x = normalize_mat(data_obj.dynamic_mat, if_retain_last_dim=False)
    data_obj.static_x = normalize_mat(data_obj.static_mat, if_retain_last_dim=False)

    """ define AutoEncoder model """
    ae = AutoEncoder(in_dim=data_obj.dynamic_feature_names + data_obj.static_feature_names,
                     en_h_dims=args.en_h_dims,
                     de_h_dims=args.de_h_dims)

    ae = ae.to(kwargs['device'])

    train(ae, data_obj, args, **kwargs)


class Param:
    def __init__(self):

        """ default configuration """
        self.area = 'los_angeles'
        self.resolution = 500
        self.year = 2018
        self.dates = ['01']

        """ training configuration """
        self.epochs = 100
        self.lr = 0.01
        self.batch_size = 4

        """ hyper parameters """
        self.en_h_dims = [64, 32, 16]
        self.de_h_dims = [16, 32, 64]

    def generate_model_name(self):
        dap_config = ['ae', '_',
                      str(self.area),
                      str(self.resolution) + 'm',
                      str(self.year), '_',
                      '_'.join(self.dates),
                      str(self.en_h_dims[-1])]
        return '_'.join(dap_config)


if __name__ == '__main__':

    """ loop over hyper-parameters """
    param = Param()

    # define directory
    base_dir = 'data/'
    data_dir = '/Users/yijunlin/Research/PRISMS/training_data/los_angeles_500m_2018/'
    model_dir = os.path.join(base_dir, 'models/')
    run_dir = os.path.join(base_dir, 'runs/')

    device = torch.device("cuda:3" if torch.cuda.is_available() else 'cpu')  # the gpu device

    kwargs = {
        'data_dir': data_dir,
        'model_dir': model_dir,
        'device': device
    }

    # give the dates

    for d in range(1, 13):
        d_str = str(d).rjust(2, '0')
        param.dates = ['01', '02', '03']

        # define model name
        model_name = param.generate_model_name()

        # define the file path
        model_file = os.path.join(model_dir, model_name + '.pkl')
        run_file = os.path.join(run_dir, model_name + '_run_{}'.format(datetime.datetime.now().strftime('%d%H%m')))
        kwargs['model_name'] = model_name
        kwargs['model_file'] = model_file
        kwargs['run_file'] = run_file

        main(param, **kwargs)
