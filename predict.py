import os
import numpy as np
import pymongo
import pytz
import torch
import torch.utils.data as dat

from data_preprocessing.db_conn.common_db import MONGO_CONN_URI
from scripts.data_loader import load_data_from_file, load_data_from_db
from options.test_options import parse_args
from utils.metrics import compute_error
from models.deeplatte import DeepLatte


def predict():
    """ train """

    """ construct index-based data loader """
    idx = np.array([i for i in range(args.seq_len + 1, data_obj.num_times)])
    idx_dat = dat.TensorDataset(torch.tensor(idx, dtype=torch.int32))
    test_idx_data_loader = dat.DataLoader(dataset=idx_dat, batch_size=1, shuffle=False)

    def construct_sequence_x(idx_list, dynamic_x, static_x):
        d_x = [dynamic_x[i - args.seq_len + 1: i + 1, ...] for i in idx_list]
        d_x = np.stack(d_x, axis=0)
        s_x = np.expand_dims(static_x, axis=0)
        s_x = np.repeat(s_x, args.seq_len, axis=1)  # shape: (t, c, h, w)
        s_x = np.repeat(s_x, len(idx_list), axis=0)  # shape: (b, t, c, h, w)
        x = np.concatenate([d_x, s_x], axis=2)
        return torch.tensor(x, dtype=torch.float).to(device)

    model.eval()
    prediction = []

    with torch.no_grad():
        for i, data in enumerate(test_idx_data_loader):
            batch_idx = data[0]
            batch_x = construct_sequence_x(batch_idx, data_obj.dynamic_x, data_obj.static_x)  # (b, t, c, h, w)
            out, _, _, _, _ = model(batch_x)
            prediction.append(out.cpu().data.numpy())

    prediction = np.concatenate(prediction)
    print((data_obj.test_y[args.seq_len + 1:, ...]>0).sum())
    #print(prediction)
    acc = compute_error(data_obj.test_y[args.seq_len + 1:, ...], prediction)

    def write_res(gids, prediction, time):
        """ write results to mongo db """

        output_list = [{'gid': gid, 'pm25': round(prediction[i], 5)} for i, gid in enumerate(gids)]
        document = {'timestamp': time, 'data': output_list}
        utc_time = time.astimezone(pytz.timezone('UTC'))
        collection_name = f'la_1000m_2021'

        mongo_client = pymongo.MongoClient(MONGO_CONN_URI)
        db = mongo_client['jonsnow']
        collection = db[collection_name]
        condition = {'timestamp': utc_time}
        output = collection.find_one(condition)

        if output is not None:
            collection.update_one(condition, {'$set': document})
            print(f'{collection_name}: Overwrite {utc_time}')
        else:
            collection.insert_one(document)
            print(f'{collection_name}: Insert {utc_time}')

        mongo_client.close()


if __name__ == '__main__':

    args = parse_args()
    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')  # the gpu device

    """ load data """
    if os.path.exists(args.data_path):
        data_obj = load_data_from_file(args.data_path)
    else:
        data_obj = load_data_from_db(args)

    """ load model """
    model = DeepLatte(in_features=data_obj.num_features,
                      en_features=[int(i) for i in args.en_features.split(',')],
                      de_features=[int(i) for i in args.de_features.split(',')],
                      in_size=(data_obj.num_rows, data_obj.num_cols),
                      h_channels=args.h_channels,
                      kernel_sizes=[int(i) for i in args.kernel_sizes.split(',')],
                      num_layers=1,
                      fc_h_features=args.fc_h_features,
                      out_features=1,  # fixed
                      p=0.5,
                      device=device).to(device)

    model_file = os.path.join(args.model_path, args.model_name + '_from_db.pkl')
    model.load_state_dict(torch.load(model_file))

    predict()



