import torch
from torch.utils import data
import numpy as np

class Dataset(data.Dataset):
  'Characterizes a dataset for PyTorch'

  def __init__(self, data, labels, train=False):
  #def __init__(self, list_IDs, labels):
        self.data = torch.from_numpy(data)
        self.labels = torch.from_numpy(labels)
        length = len(self.data)
        self.list_IDs = range(0, length)
        self.train = train


  def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

  def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample

        ID = index

        X = self.data[ID]
        y = self.labels[ID]

        if self.train == True:
            if np.random.choice([True, False]):
                X = torch.flip(X, [1])
                y[1] = -y[1]  # Y
                y[3] = -y[3]  # Relative YAW

        return X, y