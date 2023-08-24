import numpy as np
import pickle
import os
from utils import max_min_normalization,re_max_min_normalization
from dataloader import DataLoader

def load_dataset(data_dir, batch_size, valid_batch_size, test_batch_size, dataset_name):
    data_dict = {}
    # read data: train_x, train_y, val_x, val_y, test_x, test_y
    # the data has been processed and stored in datasets/{dataset}/{mode}.npz
    for mode in ['train', 'val', 'test']:
        _   = np.load(os.path.join('/content/drive/MyDrive/data/PEMS04/', mode + '.npz'))
        # length  = int(len(_['x']) * 0.1)
        # data_dict['x_' + mode]  = _['x'][:length, :, :, :]
        # data_dict['y_' + mode]  = _['y'][:length, :, :, :]
        data_dict['x_' + mode]  = _['x']
        data_dict['y_' + mode]  = _['y']
    _min = pickle.load(open("/content/drive/MyDrive/data/PEMS04/min.pkl", 'rb'))
    _max = pickle.load(open("/content/drive/MyDrive/data/PEMS04/max.pkl", 'rb'))

    # normalization
    y_train = np.squeeze(np.transpose(data_dict['y_train'], axes=[0, 2, 1, 3]), axis=-1)
    y_val = np.squeeze(np.transpose(data_dict['y_val'], axes=[0, 2, 1, 3]), axis=-1)
    y_test = np.squeeze(np.transpose(data_dict['y_test'], axes=[0, 2, 1, 3]), axis=-1)

    y_train_new = max_min_normalization(y_train, _max[:, :, 0, :], _min[:, :, 0, :])
    data_dict['y_train']    = np.transpose(y_train_new, axes=[0, 2, 1])
    y_val_new = max_min_normalization(y_val, _max[:, :, 0, :], _min[:, :, 0, :])
    data_dict['y_val']      = np.transpose(y_val_new, axes=[0, 2, 1])
    y_test_new = max_min_normalization(y_test, _max[:, :, 0, :], _min[:, :, 0, :])
    data_dict['y_test']     = np.transpose(y_test_new, axes=[0, 2, 1])

    data_dict['train_loader']   = DataLoader(data_dict['x_train'], data_dict['y_train'], batch_size, shuffle=True)
    data_dict['val_loader']     = DataLoader(data_dict['x_val'], data_dict['y_val'], valid_batch_size)
    data_dict['test_loader']    = DataLoader(data_dict['x_test'], data_dict['y_test'], test_batch_size)
    data_dict['scaler']         = re_max_min_normalization

    return data_dict