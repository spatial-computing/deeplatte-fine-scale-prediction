import os
import numpy as np
import torch
import torch.utils.data as dat
import pickle
from paramiko import SSHClient
from scp import SCPClient
import paramiko
from dotenv import load_dotenv


from scripts.data_loader import load_data_from_file, load_data_from_db
from options.test_options import parse_args
from utils.metrics import compute_error
from models.deeplatte import DeepLatte


load_dotenv('.env')

def predict(mintime,maxtime):
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
    print((data_obj.label_mat[args.seq_len + 1:, ...]>0).sum())
    print("Prediction: number of nonnan values: ",((prediction>0).sum()))
    print("prediction shape",prediction.shape)
    transfer_to_jonsnow(prediction,mintime,maxtime)
    acc = compute_error(data_obj.label_mat[args.seq_len + 1:, ...], prediction)
    print("acc: ",acc)

def transfer_to_jonsnow(data,mintime,maxtime):
   with open("result_to_jonsnow/data.pkl", "wb") as f:
      pickle.dump(data, f)

   ssh = SSHClient()
   ssh.load_system_host_keys()
   ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
   ssh.connect(hostname=os.getenv("SSH_HOST"), username=os.getenv("SSH_USER"), password=os.getenv("SSH_PWD"))
   with SCPClient(ssh.get_transport()) as scp:
      scp.put('data_preprocessing/data/los_angeles_1000m_grid_mat.npz',
              'result_from_arya/los_angeles_1000m_grid_mat.npz')  # Copy my_file.txt to the server
      scp.put('result_to_jonsnow/data.pkl','result_from_arya/data.pkl')

   sin, out, err = ssh.exec_command(f"source activate test;python result_from_arya/saveToMongo.py {mintime} {maxtime}")
   
   err = err.readlines()
   if err:
      print("something wrong")
      for i in err:
        print(i)
   os.remove("result_to_jonsnow/data.pkl")


if __name__ == '__main__':

    #write_res()

    args = parse_args()
    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')  # the gpu device

    """ load data """
    if os.path.exists(args.data_path):
        data_obj = load_data_from_file(args.data_path)
    else:
        data_obj = load_data_from_db(args)

    #transfer_to_jonsnow(data_obj.label_mat,args.min_time,args.max_time)
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

    predict(args.min_time,args.max_time)

