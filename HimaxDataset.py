import torch
from torch.utils import data

class HimaxDataset(data.Dataset):

    def __init__(self, data, labels):
        self.data = torch.from_numpy(data)
        self.labels = torch.from_numpy(labels)
        self.length = len(self.data)

    def __len__(self):
        return self.length

    def __getitem__(self, index):

        X = self.data[index]
        y = self.labels[index]

        return X, y