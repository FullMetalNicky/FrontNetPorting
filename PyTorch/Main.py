from __future__ import print_function
from PreActBlock import PreActBlock
from ConvBlock import ConvBlock
from FrontNet import FrontNet
from Dronet import Dronet
from PenguiNet import PenguiNet
import Utils as utils
import nemo


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
    model = PenguiNet(ConvBlock, [1, 1, 1], True)
    ModelManager.Read("Models/PenguiNet160x96.pt", model)

    trainer = ModelTrainer(model)

    DATA_PATH = "/Users/usi/PycharmProjects/data/160x96/"
    test = "160x96HimaxTrain16_4_2020AugCrop.pickle"
    #ModelManager.Read("Models/FrontNetGray.pt", model)
    [x_test, y_test] = DataProcessor.ProcessTestData(DATA_PATH+test)
    t_test = DataProcessor.GetTimeStampsFromTestData(DATA_PATH+test)
    test_set = Dataset(x_test, y_test)

    params = {'batch_size': 64,
              'shuffle': False,
              'num_workers': 1}
    test_loader = data.DataLoader(test_set, **params)
    MSE, MAE, r2_score, outputs, labels = trainer.Test(test_loader)
    utils.SaveResultsToCSV(labels, outputs, t_test, "results.csv")

def TestInference():

    #logging.disable(logging.INFO)
    frame = cv2.imread("../Deployment/dataset/87.pgm", 0)
    zeros = np.zeros((96, 162))
    zeros[:, 1:161] = frame
    frame = np.reshape(frame, (96, 160, 1))
    zeros = np.reshape(zeros, (96, 162, 1))

    model = PenguiNet(ConvBlock, [1, 1, 1], True)
    ModelManager.Read("Models/PenguiNet160x96_32c.pt", model)
    trainer = ModelTrainer(model)
    v1_pred = trainer.InferSingleSample(frame)
    v2_pred = trainer.InferSingleSample(zeros)
    print("output")
    print(v1_pred)
    print(v2_pred)


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
    summary(model, (1, 48, 80))
    trainer = ModelTrainer(model)

    DATA_PATH = "/Users/usi/PycharmProjects/data/80x48/"
    [x_train, x_validation, y_train, y_validation] = DataProcessor.ProcessTrainData(
        DATA_PATH + "80x48HimaxTrain.pickle")

    training_set = Dataset(x_train, y_train, True)
    params = {'batch_size': 64,
              'shuffle': True,
              'num_workers': 6}
    training_generator = data.DataLoader(training_set, **params)

    validation_set = Dataset(x_validation, y_validation)
    validation_generator = data.DataLoader(validation_set, **params)

    trainer.Train(training_generator, validation_generator)

def DumpImages():
    DATA_PATH = "/Users/usi/PycharmProjects/data/"
    [x_test, y_test] = DataProcessor.ExtractValidationLabels(DATA_PATH + "test_grey.pickle")


def ConvertToGray():
    DATA_PATH = "/Users/usi/PycharmProjects/data/108x60/old/"
    DataManipulator.CreateGreyPickle(DATA_PATH + "trainRGB.pickle", 60, 108, "bebop_train_grey.pickle")
    DataManipulator.CreateGreyPickle(DATA_PATH + "testRGB.pickle", 60, 108, "bebop_test_grey.pickle")

def CropDataset():
    DATA_PATH = "/Users/usi/PycharmProjects/data/160x160/"
    DATA_PATH2 = "/Users/usi/PycharmProjects/data/160x96/"

    DataManipulator.CropCenteredDataset(DATA_PATH + "160x160PaperTestset.pickle", [96, 160], DATA_PATH2 + "160x96PaperTestset.pickle")

def Shift():
    DATA_PATH = "/Users/usi/PycharmProjects/data/160x160/"
    DataManipulator.ShiftVideoDataset(DATA_PATH + "160x160Himax.pickle", DATA_PATH + "160x160Pitch.pickle")

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
    DATA_PATH = "/Users/usi/PycharmProjects/data/160x96/"
    train = DATA_PATH + "160x96HimaxTest16_4_2020.pickle"
    DataManipulator.Augment(train, DATA_PATH +"160x96HimaxTest16_4_2020Aug.pickle", 10)


def CropRandomTest():
    DATA_PATH = "/Users/usi/PycharmProjects/data/160x160/"
    train = DATA_PATH + "160x160HimaxMixedTest_12_03_20.pickle"
    DataManipulator.CropDataset(train, "/Users/usi/PycharmProjects/data/160x96/" + "160x96HimaxMixedTest_12_03_20Crop64.pickle", [96, 160], 64)

def AugmentAndCrop():
    DATA_PATH = "/Users/usi/PycharmProjects/data/160x160/"
    train = DATA_PATH + "160x96HimaxTest16_4_2020.pickle"
    DATA_PATH2 = "/Users/usi/PycharmProjects/data/160x90/"
    DataManipulator.AugmentAndCrop(train, DATA_PATH2 + "160x90HimaxTrain16_4_2020AugCrop.pickle", [90, 160], 10)

def Rotate():
    DATA_PATH = "/Users/usi/PycharmProjects/data/160x160/"
    test = DATA_PATH + "160x160HimaxTest16_4_2020.pickle"
    DATA_PATH2 = "/Users/usi/PycharmProjects/data/160x96/"

    DataManipulator.Rotate(test, DATA_PATH + "160x160HimaxTest16_4_2020Rot.pickle", 29)
    DataManipulator.CropCenteredDataset(DATA_PATH + "160x160HimaxTest16_4_2020Rot.pickle", [96, 160], DATA_PATH2 + "160x96HimaxTest16_4_2020Rot.pickle")

def Divide():
    DATA_PATH = "/Users/usi/PycharmProjects/data/Hand/"
    DATA_PATH2 = "/Users/usi/PycharmProjects/data/ScarletWitch/"

    old_pickle = "160x160HimaxHand_12_03_20.pickle"
    test = "160x160HimaxHandTest_12_03_20.pickle"
    train = "160x160HimaxHandTrain_12_03_20.pickle"

    DataManipulator.DivideDataset(DATA_PATH+old_pickle, DATA_PATH2+train, DATA_PATH2+test, 700)

def JoinDatasets():
    DATA_PATH = "/Users/usi/PycharmProjects/data/160x90/"
    DATA_PATH2 = "/Users/usi/PycharmProjects/data/ScarletWitch/"
    train2 = "160x90HimaxHandTrain_12_03_20AugCrop.pickle"
    train1 = "160x90HimaxMixedTrain_12_03_20AugCrop.pickle"
    train = "160x90HimaxScarletTrain.pickle"
    DataManipulator.JoinDataset(DATA_PATH+train1, DATA_PATH2+train2, DATA_PATH2+train)

def ExrtactImages():
    DATA_PATH = "/Users/usi/PycharmProjects/data/160x96/"
    test = "160x96HimaxMixedTest_12_03_20.pickle"
    DataProcessor.ExtractValidationLabels(DATA_PATH+test)


def Downsample():
    DATA_PATH = "/Users/usi/PycharmProjects/data/160x96/"
    DATA_PATH2 = "/Users/usi/PycharmProjects/data/108x60/"
    pickle = DATA_PATH + "160x96PaperTestsetPrune2.pickle"
    new = DATA_PATH2 + "108x60PaperTestsetPrune2.pickle"

    # cv2.INTER_NEAREST , cv2.INTER_LINEAR (bilinear)
    DataManipulator.DownsampleDataset(pickle, [60, 108], cv2.INTER_NEAREST, new)


def Foveate():
    DATA_PATH = "/Users/usi/PycharmProjects/data/160x96/"
    DATA_PATH2 = "/Users/usi/PycharmProjects/data/80x48/"
    pickle = DATA_PATH + "160x96HimaxTrain16_4_2020AugCrop.pickle"
    new = DATA_PATH2 + "80x48HimaxTraintFoveate.pickle"

    DataManipulator.FoveateDataset80x48(pickle, new)

def SamplingBlur():
    DATA_PATH = "/Users/usi/PycharmProjects/data/"
    orig = "160x96/160x96HimaxMixedTest_12_03_20.pickle"
    DataManipulator.BlurBySamplingDataset(DATA_PATH+orig, (48, 80), DATA_PATH+"Sizes/80x48TestNicky.pickle")
    DataManipulator.BlurBySamplingDataset(DATA_PATH+orig, (24, 40), DATA_PATH+"Sizes/40x24TestNicky.pickle")
    DataManipulator.BlurBySamplingDataset(DATA_PATH+orig, (12, 20), DATA_PATH+"Sizes/20x12TestNicky.pickle")


def Summary():
    model = Dronet(PreActBlock, [1, 1, 1], True)
    summary(model, (1, 96, 160))

def GetStats():
    DATA_PATH = "/Users/usi/PycharmProjects/data/108x60/"
    #file = "108x60HimaxTrainLinear.pickle"
    #file = "old/bebop_train_grey.pickle"
    DataProcessor.GetDatasetStatisics(DATA_PATH+file)

def TestQ():
    frame = cv2.imread("../Deployment/dataset/87.pgm", 0)
    model = PenguiNet(ConvBlock, [1, 1, 1], True, c=16, fc_nodes=960)
    # frame = cv2.resize(frame, dsize=(80, 48), interpolation=cv2.INTER_LINEAR)
    # frame = np.reshape(frame, (48, 80, 1))
    # model = nemo.transform.quantize_pact(model, dummy_input=torch.ones((1, 1, 48, 80)).to("cpu"))
   # epoch, prec_dict = ModelManager.ReadQ("Models/Q/PenguiNetfinal80x48x32.pth", model)
    frame = np.reshape(frame, (96, 160, 1))
    model = nemo.transform.quantize_pact(model, dummy_input=torch.ones((1, 1, 96, 160)).to("cpu"))
    epoch, prec_dict = ModelManager.ReadQ("Models/Q/PenguiNetfinal160x96x16.pth", model)
    trainer = ModelTrainer(model)
    v1_pred = trainer.InferSingleSample(frame)
    print("output")
    print(v1_pred)


def ToInt():
    DATA_PATH = "/Users/usi/PycharmProjects/data/160x96/"
    pickle = "160x96HimaxTrain16_4_2020AugCrop.pickle"
    DataManipulator.ConvertToInt(DATA_PATH + pickle, DATA_PATH + "160x96HimaxTrain16_4_2020AugCrop2.pickle")

def Prune():
    DATA_PATH = "/Users/usi/PycharmProjects/data/160x96/"
    pickle = "160x96PaperTestset.pickle"
    DataManipulator.PruneBadFrames(DATA_PATH + pickle, DATA_PATH + "160x96PaperTestsetPrune.pickle")
    #DataManipulator.TrimDataset(DATA_PATH + "KrstoPrune.pickle", DATA_PATH + "KrstoFinal.pickle", start=330, end=2190)

def Trim():
    DATA_PATH = "/Users/usi/PycharmProjects/data/160x96/"
    pickle = "160x96PaperTestsetPrune.pickle"
    DataManipulator.TrimDataset(DATA_PATH + pickle, DATA_PATH + "160x96PaperTestsetPrune2.pickle")



def JoinSeveralDataets():
    DATA_PATH = "/Users/usi/PycharmProjects/data/160x160/"
    pathlist = [DATA_PATH + "KrstoFinal.pickle", DATA_PATH + "GabryFinal.pickle",DATA_PATH + "JamalFinal.pickle",DATA_PATH + "BorisFinal.pickle"]
    new_path = DATA_PATH + "160x160PaperTestset.pickle"
    DataManipulator.JoinDatasetFromList(pathlist, new_path)


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

    #JoinSeveralDataets()
    #Prune()
    #TestQ()
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
    #Augment()
    #AddColumnsToDataSet("train_grey.pickle", 60, 108, 1)
    #MixAndMatch()
    #CropRandomTest()
    #AugmentAndCrop()
    #Rotate()
    #Divide()
    #JoinDatasets()
    #ExrtactImages()
    Downsample()
    #Foveate()
    #ExrtactImages()
    #SamplingBlur()
    #Summary()
    #GetStats()
    #Rotate()
    #ToInt()
    #Trim()


if __name__ == '__main__':
    main()