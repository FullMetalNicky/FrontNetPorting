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


  def toNumpy(self, X):
      X = X.cpu().numpy()
      h, w = X.shape[1:3]
      X = np.reshape(X, (h, w)).astype("uint8")
      return X

  def toTensor(self, X):
      h, w = X.shape
      X = np.reshape(X, (1, h, w))
      X = torch.from_numpy(X).float()
      return X

  def augmentGamma(self, X):

      # gamma correction augmentation
      gamma = np.random.uniform(0.6, 1.4)
      table = self.it.adjust_gamma(gamma)
      X = cv2.LUT(X, table)

      return X

  def augmentDR(self, X):

      # # dynamic range augmentation
      dr = np.random.uniform(0.4, 0.8)  # dynamic range
      lo = np.random.uniform(0, 0.3)
      hi = min(1.0, lo + dr)
      X = np.interp(X/255.0, [0, lo, hi, 1], [0, 0, 1, 1])
      X = 255 * X

      return X

  def augmentBlur(self, X):
      kernel = np.random.choice([3, 5])  # dynamic range
      X = cv2.blur(X, (kernel, kernel))

      return X

  def augmentNoise(self, X):
      h, w = X.shape[:2]
      noise = np.zeros((h,w), np.uint8)
      cv2.randn(noise, 0, 1)
      X = X + noise

      return X


  def __getitem__(self, index):
        'Generates one sample of data'
        ID = index

        X = self.data[ID]
        y = self.labels[ID]

        if self.train == True:
            if np.random.choice([True, False]):
                X = torch.flip(X, [2])
                y[1] = -y[1]  # Y
                y[3] = -y[3]  # Relative YAW

            if X.shape[0] == 1:
                X = self.toNumpy(X)
                if np.random.choice([True, False]):
                    X = self.augmentGamma(X)
                if np.random.choice([True, False]):
                    X = self.augmentDR(X)
                if np.random.choice([True, False]):
                    X = self.augmentBlur(X)
                if np.random.choice([True, False]):
                    X = self.augmentNoise(X)

                X = self.toTensor(X)

        return X, y
