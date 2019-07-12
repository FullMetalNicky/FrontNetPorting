from __future__ import print_function
from FrontNet import BasicBlock
from FrontNet import FrontNet
from DataProcessor import DataProcessor
from ModelTrainer import ModelTrainer
from Dataset import Dataset
from torch.utils import data
from ModelManager import ModelManager
import torch

model = FrontNet(BasicBlock, [1, 1, 1])
#manager = ModelManager()
#manager.Read('Models/FrontNet-099-28-6.pkl', model, torch.optim.Adam(model.parameters(), lr=0.001))


DATA_PATH = "/Users/usi/Downloads/"
loader = DataProcessor()
[train_mean, train_std, x_train, x_validation, x_test, y_train, y_validation, y_test] = loader.ProcessData(DATA_PATH + "train.pickle", DATA_PATH + "test.pickle")
training_set = Dataset(x_train, y_train, True)
validation_set = Dataset(x_validation, y_validation)
test_set = Dataset(x_test, y_test)


# Parameters
params = {'batch_size': 64,
          'shuffle': True,
          'num_workers': 6}
training_generator = data.DataLoader(training_set, **params)
validation_generator = data.DataLoader(validation_set, **params)
params = {'batch_size': 64,
          'shuffle': False,
          'num_workers': 6}
test_generator = data.DataLoader(test_set, **params)



trainer = ModelTrainer(model, 2)
#model.load_state_dict(torch.load('Models/FrontNet-099-28-6.pkl'))

#trainer.Train(training_generator, validation_generator)
#trainer.PerdictSingleSample(test_generator)
trainer.Predict(test_generator)