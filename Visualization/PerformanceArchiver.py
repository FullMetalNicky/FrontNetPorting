from __future__ import print_function
import logging
import numpy as np
import matplotlib.pyplot as plt
from torch.utils import data
import sys
import sklearn.metrics
import pandas as pd


sys.path.append("../PyTorch/")

from PreActBlock import PreActBlock
from Dronet import Dronet
from DataProcessor import DataProcessor
from ModelTrainer import ModelTrainer
from Dataset import Dataset
from ModelManager import ModelManager



def LoadPerformanceResults(pickle_path):
    dataset = pd.read_pickle(pickle_path)
    logging.info('[DataManipulator] dataset shape: ' + str(dataset.shape))

    outputs = dataset['outputs'].values
    gt_labels = dataset['gt_labels'].values

    # outputs = np.reshape(outputs, (-1, 4))
    # gt_labels = np.reshape(gt_labels, (-1, 4))

    outputs = np.vstack(outputs[:]).astype(np.float32)
    gt_labels = np.vstack(gt_labels[:]).astype(np.float32)


    return outputs, gt_labels



def DumpPerformanceResults(datset_path, model_path, name):

    [x_test2, y_test2] = DataProcessor.ProcessTestData(datset_path)
    test_set2 = Dataset(x_test2, y_test2)
    params = {'batch_size': 1, 'shuffle': False, 'num_workers': 0}
    test_generator = data.DataLoader(test_set2, **params)

    model = Dronet(PreActBlock, [1, 1, 1], True)
    ModelManager.Read(model_path, model)
    trainer = ModelTrainer(model)
    MSE, MAE, r2_score, outputs, gt_labels = trainer.Test(test_generator)


    outputs = list(outputs)
    df = pd.DataFrame({'outputs': outputs, 'gt_labels': gt_labels})
    df.to_pickle(name)


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


    DATA_PATH = "/Users/usi/PycharmProjects/data/160x90/"
    picklename = "160x90HimaxMixedTest_12_03_20Cropped70.pickle"
    datset_path = DATA_PATH + picklename

    # model_path1 = '../PyTorch/Models/DronetHimax160x90AugCrop.pt'
    # name1 = "DronetHimax160x90AugCropResults.pickle"
    # DumpPerformanceResults(datset_path, model_path1, name1)

    name2 = "DronetHimax160x90AugmentedResults.pickle"
    model_path2 = '../PyTorch/Models/DronetHimax160x90Augmented.pt'
    DumpPerformanceResults(datset_path, model_path2, name2)



if __name__ == '__main__':
    main()
