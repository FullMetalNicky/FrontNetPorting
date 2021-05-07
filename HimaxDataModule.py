import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
import pandas as pd
import numpy as np
from HimaxDataset import HimaxDataset

class HimaxDataModule(pl.LightningDataModule):

    def __init__(self, trainset_path: str = "path/to/set", testset_path: str = "path/to/set", batch_size: int = 64):
        super().__init__()
        self.trainset_path = trainset_path
        self.testset_path = testset_path
        self.batch_size = batch_size

    def GetSizeDataFromDataFrame(self, dataset):
        h = int(dataset['h'].values[0])
        w = int(dataset['w'].values[0])
        c = int(dataset['c'].values[0])

        return h, w, c

    def unpack_data(self, dataset):

        h, w, c = self.GetSizeDataFromDataFrame(dataset)
        x = dataset['x'].values
        x = np.vstack(x[:]).astype(np.float32)
        x = np.reshape(x, (-1, h, w, c))

        x = np.swapaxes(x, 1, 3)
        x = np.swapaxes(x, 2, 3)

        y = dataset['y'].values
        y = np.vstack(y[:]).astype(np.float32)

        return x, y

    def setup(self, stage=None):

        train_set = pd.read_pickle(self.trainset_path)
        x_train, y_train = self.unpack_data(train_set)

        size = train_set.shape[0]
        val_size = int(float(size) * 0.2)
        trainset = HimaxDataset(x_train, y_train)
        self.trainset, self.valset = random_split(trainset, [size - val_size, val_size])

        test_set = pd.read_pickle(self.testset_path)
        x_test, y_test = self.unpack_data(test_set)
        self.testset = HimaxDataset(x_test, y_test)

    def train_dataloader(self):
        trainset = DataLoader(self.trainset, batch_size=64)
        return trainset

    def val_dataloader(self):
        valset = DataLoader(self.valset, batch_size=64)
        return valset

    def test_dataloader(self):
        testset = DataLoader(self.testset, batch_size=64)
        return testset