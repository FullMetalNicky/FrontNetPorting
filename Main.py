from __future__ import print_function
from FrontNet import PreActBlock
from FrontNet import FrontNet
from DataProcessor import DataProcessor
from ModelTrainer import ModelTrainer
from Dataset import Dataset
from torch.utils import data
from ModelManager import ModelManager
from DataVisualization import DataVisualization
import torch
import logging

logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s - %(levelname)s - %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S",
                        filename="log.txt",
                        filemode='w')


console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter('%(message)s')
console.setFormatter(formatter)
logging.getLogger('').addHandler(console)



model = FrontNet(PreActBlock, [1, 1, 1])
ModelManager.Read('Models/FrontNet-097.pkl', model)


DATA_PATH = "/Users/usi/PycharmProjects/data/"

#[train_mean, train_std, x_train, x_validation, y_train, y_validation] = DataProcessor.ProcessTrainData(DATA_PATH + "train.pickle", 60, 108)
[x_test, y_test] = DataProcessor.ProcessTestData(DATA_PATH + "test.pickle", 60, 108)

# training_set = Dataset(x_train, y_train, True)
# validation_set = Dataset(x_validation, y_validation)
#
#
# # Parameters
# params = {'batch_size': 64,
#           'shuffle': True,
#           'num_workers': 0}
# training_generator = data.DataLoader(training_set, **params)
# validation_generator = data.DataLoader(validation_set, **params)

test_set = Dataset(x_test, y_test)
params = {'batch_size': 1,
          'shuffle': False,
          'num_workers': 0}
test_generator = data.DataLoader(test_set, **params)

trainer = ModelTrainer(model)
x_test_res , y_test_res, outputs = trainer.PerdictSingleSample(test_generator)
DataVisualization.DisplayFrameAndPose(x_test_res , y_test_res, outputs)

