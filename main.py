import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from torch.utils.data import DataLoader

from DataPreprocess import DataPreprocessor
from DataSet import DataSet


def position_encoding(length, dimension, kind='sinusodial', period=10000):
    if kind == 'sinusodial':
        position_encoding = np.zeros((length, dimension))
        for i in range(length):
            for j in range(dimension):
                if j % 2 == 0:
                    position_encoding[i, j] = np.sin(i / period ** (j / dimension))
                else:
                    position_encoding[i, j] = np.cos(i / period ** (j / dimension))
        return position_encoding
    else:
        return None


if __name__ == '__main__':
    input_length = 20
    induce_length = 5
    pred_length = 10
    encoding_dimension = 6
    encoding_type = 'time_encoding'
    batch_size = 30
    data_preprocessor = DataPreprocessor('data/ETTh1.pkl', input_length, pred_length, encoding_type=encoding_type,
                                         encoding_dimension=encoding_dimension)
    train_set = DataSet(data_preprocessor.load_train_set(), data_preprocessor.load_train_encoding_set(),
                        input_length, induce_length, pred_length)
    test_set = DataSet(data_preprocessor.load_test_set(), data_preprocessor.load_test_encoding_set(),
                       input_length, induce_length, pred_length)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    for i, (input_x, input_y, encoding_x, encoding_y, ground_truth) in enumerate(train_loader):
        print(i, input_x.shape, input_y.shape, encoding_x.shape, encoding_y.shape, ground_truth.shape)
