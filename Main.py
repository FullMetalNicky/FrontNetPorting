from __future__ import print_function
from FrontNet import ResidualBlock
from FrontNet import FrontNet
from DataProcessor import DataProcessor
from ModelTrainer import ModelTrainer
from Dataset import Dataset
from torch.utils import data


model = FrontNet(ResidualBlock, [1, 1, 1])

DATA_PATH = "/Users/usi/Downloads/"
loader = DataProcessor()
[sel_idx, x_train, x_validation, x_test, y_train, y_validation, y_test] = loader.ProcessData(DATA_PATH + "train.pickle", DATA_PATH + "test.pickle")
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
trainer.Train(training_generator, validation_generator)
trainer.PerdictSingleSample(test_generator)
#trainer.Predict(test_generator)
