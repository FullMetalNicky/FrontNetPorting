from __future__ import print_function
from FrontNet import ResidualBlock
from FrontNet import FrontNet
from DataProcessor import DataProcessor
from ModelTrainer import ModelTrainer
from Dataset import Dataset
from torch.utils import data


model = FrontNet(ResidualBlock, [1, 1, 1])
print(model)

DATA_PATH = "/Users/usi/Downloads/"
loader = DataProcessor()
[sel_idx, x_train, x_validation, x_test, y_train, y_validation, y_test] = loader.ProcessData(DATA_PATH + "train.pickle", DATA_PATH + "test.pickle")
training_set = Dataset(x_train, y_train)
validation_set = Dataset(x_validation, y_validation)


# Parameters
params = {'batch_size': 64,
          'shuffle': True,
          'num_workers': 6}
training_generator = data.DataLoader(training_set, **params)
validation_generator = data.DataLoader(validation_set, **params)



trainer = ModelTrainer(model, 2)
trainer.Train(sel_idx, x_train, x_validation, x_test, y_train, y_validation)
trainer.Predict(x_test, y_test)
