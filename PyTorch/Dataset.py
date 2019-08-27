import torch
from torch.utils import data
import numpy as np
import cv2
import sys
sys.path.append("../DataProcessing/")
from ImageTransformer import ImageTransformer


class Dataset(data.Dataset):
  'Characterizes a dataset for PyTorch'

  def __init__(self, data, labels, train=False):
        self.data = torch.from_numpy(data)
        self.labels = torch.from_numpy(labels)
        length = len(self.data)
        self.list_IDs = range(0, length)
        self.train = train
        self.it = ImageTransformer()


  def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

  def __getitem__(self, index):
        'Generates one sample of data'
        ID = index

        X = self.data[ID]
        y = self.labels[ID]

        if self.train == True:
            if np.random.choice([True, False]):
                X = torch.flip(X, [1])
                y[1] = -y[1]  # Y
                y[3] = -y[3]  # Relative YAW

            X = X.cpu().numpy()
            h, w = X.shape[1:3]
            X = np.reshape(X, (h, w)).astype("uint8")
            #if np.random.choice([True, False]):
            #    gamma = np.random.uniform(0.6, 1.4)
            #    table = self.it.adjust_gamma(gamma)
            #    X = cv2.LUT(X, table)
            if np.random.choice([True, False]):
                dr = np.random.uniform(0.4, 0.8)  # dynamic range
                lo = np.random.uniform(0, 0.3)
                hi = min(1.0, lo + dr)
                X = np.interp(X/255.0, [0, lo, hi, 1], [0, 0, 1, 1])
                X = 255 * X
            X = np.reshape(X, (1, h, w))
            X = torch.from_numpy(X).float()

        return X, y
