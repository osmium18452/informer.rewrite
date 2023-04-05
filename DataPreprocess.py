import numpy as np
import pickle

import torch


class DataPreprocessor:
    def __init__(self, file_name, input_length, pred_length, train_set_ratio=.8, encoding_dimension=6,
                 encoding_type='time_encoding'):
        data = pickle.load(open(file_name, 'rb'))
        self.time_encoding = self.generate_positional_encoding(data.shape[0], encoding_type=encoding_type,
                                                               dimension=encoding_dimension)
        frame_length = input_length + pred_length
        total_samples = data.shape[0] - frame_length + 1
        self.train_set_size = int(total_samples * train_set_ratio)
        self.test_set_size = total_samples - self.train_set_size
        window = np.arange(frame_length)
        dataset_index = np.tile(np.arange(total_samples), frame_length).reshape(frame_length, total_samples).T + window
        self.dataset = data[dataset_index]
        self.time_encoding = self.time_encoding[dataset_index]

    def load_train_set(self):
        return torch.Tensor(self.dataset[:self.train_set_size])

    def load_train_encoding_set(self):
        return torch.Tensor(self.time_encoding[:self.train_set_size])

    def load_test_set(self):
        return torch.Tensor(self.dataset[self.train_set_size:])

    def load_test_encoding_set(self):
        return torch.Tensor(self.time_encoding[self.train_set_size:])

    def generate_positional_encoding(self, len, encoding_type=None, **kwargs):
        if encoding_type is None:
            return None
        elif encoding_type == 'time_encoding':
            dimension = kwargs['dimension']
            if dimension > 6:
                print('dimension is too high')
                exit(1)
            time_stamp = np.zeros((len, 6))
            sec_of_min = (np.arange(60, dtype=float) - 30) / 60
            min_of_hour = (np.arange(60, dtype=float) - 30) / 60
            hour_of_day = (np.arange(24, dtype=float) - 12) / 24
            day_of_week = (np.arange(7, dtype=float) - 3) / 7
            day_of_month = (np.arange(31, dtype=float) - 15) / 31
            day_of_year = (np.arange(365, dtype=float) - 182) / 365
            for i in range(len):
                time_stamp[i, 0] = sec_of_min[i % 60]
                time_stamp[i, 1] = min_of_hour[i % 60]
                time_stamp[i, 2] = hour_of_day[i % 24]
                time_stamp[i, 3] = day_of_week[i % 7]
                time_stamp[i, 4] = day_of_month[i % 31]
                time_stamp[i, 5] = day_of_year[i % 365]
            return time_stamp[:, :dimension]


if __name__ == '__main__':
    file_name = 'data/ETTh1.pkl'
    data_preprocessor = DataPreprocessor(file_name, 20, 10)
    train_set = data_preprocessor.load_train_set()
    test_set = data_preprocessor.load_test_set()
    print('train/test set shape: ', train_set.shape, test_set.shape)
