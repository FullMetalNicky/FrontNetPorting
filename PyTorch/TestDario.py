from DataProcessor import DataProcessor
from ModelTrainer import ModelTrainer
from Dataset import Dataset
from torch.utils import data
from ModelManager import ModelManager
import torch

import logging
from DarioNet import DarioNet
from PreActBlockOld import PreActBlock
from ConvBlock import ConvBlock
from FrontNet import FrontNet
import cv2
import numpy as np
from torchsummary import summary
import torch

def Show(x_test):

    for i in range(len(x_test)):
        img = x_test[i].astype("uint8")
        cv2.putText(img, str(i), (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, 255)
        cv2.imshow("show", img)
        cv2.waitKey()


def Train():
    model = FrontNet(PreActBlock, [1, 1, 1], False)
    summary(model, (3, 60, 108))


    size = "108x60"
    DATA_PATH = "/Users/usi/PycharmProjects/data/" + size + "/old/"
    picklename = "108x60BebopRGBTrain.pickle"

    [x_train, x_validation, y_train, y_validation] = DataProcessor.ProcessTrainData(DATA_PATH + picklename)

    training_set = Dataset(x_train, y_train, False)
    validation_set = Dataset(x_validation, y_validation)

    # Parameters
    # num_workers - 0 for debug in Mac+PyCharm, 6 for everything else
    num_workers = 0
    params = {'batch_size':64,
              'shuffle': True,
              'num_workers': 0}
    train_loader = data.DataLoader(training_set, **params)
    validation_loader = data.DataLoader(validation_set, **params)

    trainer = ModelTrainer(model)
    trainer.Train(train_loader, validation_loader)


def Test():
    size = "108x60"
    # DATA_PATH = "/Users/usi/PycharmProjects/data/" + size + "/old/"
    # picklename = "108x60BebopRGBTest.pickle"
    # [x_test, y_test] = DataProcessor.ProcessTestData(DATA_PATH + picklename)

    DATA_PATH = "/Users/usi/PycharmProjects/data/" + size + "/"
    picklename = size + "PaperTestsetPrune2.pickle"
    #[x_test, y_test] = DataProcessor.ProcessTestDataAsRGB(DATA_PATH + picklename)
    [x_test, y_test] = DataProcessor.ProcessTestData(DATA_PATH + picklename)

    # h = x_test.shape[2]
    # w = x_test.shape[3]
    # x_test = np.reshape(x_test, (-1, h, w))
    # Show(x_test)

    model = FrontNet(PreActBlock, [1, 1, 1], True)
    ModelManager.Read('../PyTorch/Models/FrontNet108x60Others.pt', model)
    test_set = Dataset(x_test, y_test)
    params = {'batch_size': 1,
              'shuffle': False,
              'num_workers': 1}
    test_generator = data.DataLoader(test_set, **params)
    trainer = ModelTrainer(model)
    MSE, MAE, r2_score, outputs, gt_labels = trainer.Test(test_generator)



def main():

    # [NeMO] Setup of console logging.
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

    #Train()
    Test()





if __name__ == '__main__':
    main()