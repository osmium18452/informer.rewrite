import numpy as np
from torch.utils.data import Dataset

class DataSet(Dataset):
    def __init__(self,dataset,time_encoding, input_len,induce_len,pred_len):
        self.dataset=dataset
        self.time_encoding=time_encoding
        self.input_len=input_len
        self.induce_len=induce_len
        self.pred_len=pred_len

    def __getitem__(self, index):
        input_x=self.dataset[index,:self.input_len]
        ground_truth=self.dataset[index,self.input_len:]
        induce_slice=self.dataset[index,self.input_len-self.induce_len:self.input_len]
        input_y=np.concatenate((induce_slice,np.zeros(ground_truth.shape,dtype=np.float32)),axis=0)
        time_encoding_x=self.time_encoding[index,:self.input_len]
        time_encoding_y=self.time_encoding[index,-input_y.shape[0]:]
        return input_x,time_encoding_x,input_y,time_encoding_y,ground_truth

    def __len__(self):
        return self.dataset.shape[0]

