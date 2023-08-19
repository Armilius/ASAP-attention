import os
import pandas as pd
from torch.utils.data import Dataset
from utils.tools import StandardScaler
import math
import numpy as np
import warnings
warnings.filterwarnings('ignore')

class Dataset_ETT(Dataset):
    def __init__(self, root_path, flag='train', size=None, features='S', data_path='ETTh1.csv', target='OT', scale=True,
                 inverse=False, cols=None, padding="padding"):
        if size == None:
            self.seq_len = 24*4*4
            self.token_len = 24*4
            self.pred_len = 24*4
        else:
            self.seq_len = size[0]
            self.token_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train':0, 'val':1, 'test':2}
        self.set_type = type_map[flag]
        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.root_path = root_path
        self.data_path = data_path
        self.padding = padding
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))
        border1s = [0, 12*30*24 - self.seq_len, 12*30*24+4*30*24 - self.seq_len]
        border2s = [12*30*24, 12*30*24+4*30*24, 12*30*24+8*30*24]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]
        
        if self.features=='M' or self.features=='MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features=='S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        self.data_x = data[border1:border2]
        if self.inverse:
            self.data_y = df_data.values[border1:border2]
        else:
            self.data_y = data[border1:border2]
    
    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.token_len
        r_end = r_begin + self.token_len
        p_begin = r_end
        p_end = r_end + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_x[r_begin:r_end]
        seq_label = self.data_y[p_begin: p_end]

        if self.token_len < self.pred_len:
            if self.padding == "circular":
                num_iter = math.ceil(self.pred_len / self.token_len)
                dec_inp = np.tile(seq_y, (num_iter, 1))[:self.pred_len]
            else:
                dec_inp = np.zeros_like(seq_label, dtype=np.float)
                dec_inp[:self.token_len] = seq_y
        else:
            dec_inp = seq_y

        return seq_x, dec_inp, seq_label
    
    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_Custom(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, inverse=False, cols=None, padding="circular"):
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.token_len = size[1]
            self.pred_len = size[2]
        # init

        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]
        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.cols = cols
        self.root_path = root_path
        self.data_path = data_path
        self.padding = padding
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))
        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        if self.cols:
            cols = self.cols.copy()
            cols.remove(self.target)
        else:
            cols = list(df_raw.columns)
            cols.remove(self.target)
            cols.remove('date')
        df_raw = df_raw[['date'] + cols + [self.target]]

        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        self.data_x = data[border1:border2]
        if self.inverse:
            self.data_y = df_data.values[border1:border2]
        else:
            self.data_y = data[border1:border2]

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.token_len
        r_end = r_begin + self.token_len
        p_begin = r_end
        p_end = r_end + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_x[r_begin:r_end]
        seq_label = self.data_y[p_begin: p_end]

        if self.token_len < self.pred_len:
            if self.padding == "circular":
                num_iter = math.ceil(self.pred_len / self.token_len)
                dec_inp = np.tile(seq_y, (num_iter, 1))[:self.pred_len]
            else:
                dec_inp = np.zeros_like(seq_label, dtype=np.float)
                dec_inp[:self.token_len] = seq_y
        else:
            dec_inp = seq_y

        return seq_x, dec_inp, seq_label

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

