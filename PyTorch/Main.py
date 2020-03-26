from __future__ import print_function
from PreActBlock import PreActBlock
from FrontNet import FrontNet
from Dronet import Dronet

from DataProcessor import DataProcessor
from ModelTrainer import ModelTrainer
from Dataset import Dataset
from torch.utils import data
from ModelManager import ModelManager
from DataManipulator import DataManipulator
import logging
import numpy as np
import cv2
import pandas as pd
from torchsummary import summary
import torch

def Filter():
    [x_test, y_test] = DataProcessor.ProcessTestData("/Users/usi/PycharmProjects/data/test_vignette4.pickle")
    x_test2 = []
    y_test2 = []
    for i in range(len(x_test)):
        gt = y_test[i]
        if ((gt[0] > 1.0) and (gt[0] < 2.0)):
            x_test2.append(x_test[i])
            y_test2.append(y_test[i])

    x_test2 = np.asarray(x_test2)
    y_test2 = np.asarray(y_test2)
    test_set = Dataset(x_test2, y_test2)

    params = {'batch_size': 64,
              'shuffle': False,
              'num_workers': 0}
    test_loader = data.DataLoader(test_set, **params)
    model = Dronet(PreActBlock, [1, 1, 1], True)
    ModelManager.Read("Models/DronetGray.pt", model)
    trainer = ModelTrainer(model)
    trainer.Predict(test_loader)



def Test():
    model = Dronet(PreActBlock, [1, 1, 1], True)
    ModelManager.Read("Models/DronetHimax160x90.pt", model)

    trainer = ModelTrainer(model)

    #ModelManager.Read("Models/FrontNetGray.pt", model)
    [x_test, y_test] = DataProcessor.ProcessTestData("/Users/usi/PycharmProjects/data/160x90HimaxStatic_12_03_20.pickle")
    test_set = Dataset(x_test, y_test)

    params = {'batch_size': 64,
              'shuffle': False,
              'num_workers': 1}
    test_loader = data.DataLoader(test_set, **params)
    trainer.Predict(test_loader)

def TestInference():

    frame = cv2.imread("8.jpg", 0)
    #frame = frame[92:152, 108:216]
    frame = np.reshape(frame, (60, 108, 1))
    #print(frame.flatten()[:10])
    #cv2.imshow("", frame)
    #cv2.waitKey()
    model = Dronet(PreActBlock, [1, 1, 1], True)
    ModelManager.Read("Models/DronetGrayAug120.pt", model)
    # weight = model.conv.weight.data
    # weight = np.reshape(weight, (-1))
    # weight = weight[:10]
    # print(weight)
    trainer = ModelTrainer(model)
    v1_pred = trainer.InferSingleSample(frame)
    print("output")
    print(v1_pred)


def MergeDatasets():
    DATA_PATH = "/Users/usi/PycharmProjects/data/"

    dataset = pd.read_pickle(DATA_PATH + "train.pickle").values
    dataset2 = pd.read_pickle(DATA_PATH + "TrainNicky.pickle").values
    x_dataset = dataset[:, 0]
    y_dataset = dataset[:, 1]
    x_dataset2 = dataset2[:, 0]
    y_dataset2 = dataset2[:, 1]
    x_dataset = np.append(x_dataset, x_dataset2)
    y_dataset = np.append(y_dataset, y_dataset2)

    print("dataset ready x:{} y:{}".format(len(x_dataset), len(y_dataset)))
    df = pd.DataFrame(data={'x': x_dataset, 'y': y_dataset})
    print("dataframe ready")
    df.to_pickle("TrainNickyFull.pickle")

def Train():
    model = FrontNet(PreActBlock, [1, 1, 1], False)
    trainer = ModelTrainer(model)

    DATA_PATH = "/Users/usi/PycharmProjects/data/"

    [x_train, x_validation, y_train, y_validation] = DataProcessor.ProcessTrainData(
        DATA_PATH + "HandHead.pickle", 60, 108)

    training_set = Dataset(x_train, y_train, True)
    params = {'batch_size': 64,
              'shuffle': True,
              'num_workers': 0}
    training_generator = data.DataLoader(training_set, **params)

    validation_set = Dataset(x_validation, y_validation)
    validation_generator = data.DataLoader(validation_set, **params)

    trainer.Train(training_generator, validation_generator)



def TrainGray():
    model = Dronet(PreActBlock, [1, 1, 1], True)
    summary(model, (1, 60, 108))
    trainer = ModelTrainer(model)

    DATA_PATH = "/Users/usi/PycharmProjects/data/"
    [x_train, x_validation, y_train, y_validation] = DataProcessor.ProcessTrainData(
        DATA_PATH + "160x160HimaxDynamic_12_03_20.pickle")

    training_set = Dataset(x_train, y_train, True)
    params = {'batch_size': 64,
              'shuffle': True,
              'num_workers': 0}
    training_generator = data.DataLoader(training_set, **params)

    validation_set = Dataset(x_validation, y_validation)
    validation_generator = data.DataLoader(validation_set, **params)

    trainer.Train(training_generator, validation_generator)

def DumpImages():
    DATA_PATH = "/Users/usi/PycharmProjects/data/"
    [x_test, y_test] = DataProcessor.ExtractValidationLabels(DATA_PATH + "test_grey.pickle")


def ConvertToGray():
    DATA_PATH = "/Users/usi/PycharmProjects/data/"
    DataManipulator.CreateGreyPickle(DATA_PATH + "BebopFlightSim_06_03_20.pickle", 60, 108, "GreyBebopFlightSim_06_03_20.pickle")
    DataManipulator.CreateGreyPickle(DATA_PATH + "BebopPatterns_06_03_20.pickle", 60, 108, "GreyBebopPatterns_06_03_20.pickle")

def CropDataset():
    DATA_PATH = "/Users/usi/PycharmProjects/data/"
    DataManipulator.CropCenteredDataset(DATA_PATH + "160x160HimaxStatic_12_03_20.pickle", [90, 160], DATA_PATH + "160x90HimaxStatic_12_03_20.pickle")

def Shift():
    DATA_PATH = "/Users/usi/PycharmProjects/data/"
    DataManipulator.ShiftVideoDataset(DATA_PATH + "160x90HimaxStatic_12_03_20.pickle", DATA_PATH + "160x90HimaxStatic_12_03_20.pickle")

def MixAndMatch():
    DATA_PATH = "/Users/usi/PycharmProjects/data/160x160/"
    path1 = DATA_PATH + "160x160HimaxStatic_12_03_20.pickle"
    path2 = DATA_PATH +"160x160HimaxDynamic_12_03_20.pickle"
    train = DATA_PATH +"160x160HimaxMixedTrain_12_03_20.pickle"
    test = DATA_PATH + "160x160HimaxMixedTest_12_03_20.pickle"
    DataManipulator.MixAndMatch(path1, path2, train, test)

def AddColumnsToDataSet(picklename, height, width, channels):

    DATA_PATH = "/Users/usi/PycharmProjects/data/"
    dataset = pd.read_pickle(DATA_PATH + picklename)
    df = pd.DataFrame({
    'c': channels,
    'w' : width,
    'h' : height
    }, index=[0])

    new = pd.concat([dataset, df], axis=1)
    print(new.head)
    print("dataframe ready")
    new.to_pickle(DATA_PATH + "train_grey2.pickle")


def Augment():
    DATA_PATH = "/Users/usi/PycharmProjects/data/160x90/"
    train = DATA_PATH + "160x90HimaxMixedTrain_12_03_20.pickle"
    DataManipulator.Augment(train, DATA_PATH +"160x90HimaxMixedTrain_12_03_20Augmented.pickle", 10)


def CropRandomTest():
    DATA_PATH = "/Users/usi/PycharmProjects/data/160x160/"
    train = DATA_PATH + "160x160HimaxMixedTest_12_03_20.pickle"
    DataManipulator.CropDataset(train, "/Users/usi/PycharmProjects/data/160x90/" + "160x90HimaxMixedTest_12_03_20Cropped.pickle", [90, 160], 20)



def main():
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

    #Test()
    #TrainGray()
    #ConvertToGray()
    #MergeDatasets()
    #Train()
    #TestInference()
    #DumpImages()
    #Filter()
    #CropDataset()
    #Shift()
    #AddColumnsToDataSet("train_grey.pickle", 60, 108, 1)
    #MixAndMatch()
    CropRandomTest()


if __name__ == '__main__':
    main()