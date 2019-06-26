from __future__ import print_function
from FrontNet import ResidualBlock
from FrontNet import FrontNet
from DataLoader import DataLoader
from ModelTrainer import ModelTrainer


model = FrontNet(ResidualBlock, [1, 1, 1])
print(model)

DATA_PATH = "/Users/usi/Downloads/"
loader = DataLoader(DATA_PATH + "train.pickle", DATA_PATH + "test.pickle")
[sel_idx, x_train, x_validation, x_test, y_train, y_validation, y_test] = loader.ProcessData()

trainer = ModelTrainer(model, 2)
trainer.Train(sel_idx, x_train, x_validation, x_test, y_train, y_validation)
trainer.Predict(x_test, y_test)
