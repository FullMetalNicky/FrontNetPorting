
import logging
import numpy as np
import matplotlib.pyplot as plt
from torch.utils import data
import sys
import sklearn.metrics
import pandas as pd
import random

sys.path.append("../PyTorch/")

from PreActBlock import PreActBlock
from Dronet import Dronet
from DataProcessor import DataProcessor
from ModelTrainer import ModelTrainer
from Dataset import Dataset
from ModelManager import ModelManager
from PerformanceArchiver import LoadPerformanceResults






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

    size = "160x96"
    DATA_PATH = "/Users/usi/PycharmProjects/data/" + size + "/"
    picklename = size + "HimaxTest16_4_2020Cropped64.pickle"
    p_test = DataProcessor.GetPitchFromTestData(DATA_PATH + picklename)

    name2 = "pickles/160x96DronetNickyTestedWithOthers.pickle"
    outputs, gt_labels = LoadPerformanceResults(name2)

    outputs = np.reshape(outputs, (-1, 64, 4))
    gt_labels = np.reshape(gt_labels, (-1, 64, 4))


    x_outputs = outputs[:,:, 0]

    np.save('x_pred_NickyOnOthers.npy', x_outputs)

    gt = gt_labels[:, 0, 0]
    avg_gt = np.mean(gt)

    pitch_avg_pred = []
    pitch_avg_pred_random = []
    rnd = random.sample(range(0, 803), 10)

    for i in range(64):
        pred = outputs[:, i, 0]
        avg = np.mean(pred)
        pitch_avg_pred.append(avg)
        #print("pitch {} avg pred x is {}, avg gt is {}".format(i, avg, avg_gt))


    samples = []
    for j in rnd:
        pred = outputs[j, :, 0]
        samples.append(pred)


    tmp = samples[0]
    x = np.arange(64)  # the label locations
    fig, ax = plt.subplots(figsize=(20, 8))
    labels = range(64)
    x = np.arange(64)  # the label locations
    plt.plot(x, pitch_avg_pred, color='black', label="All")
    plt.plot(x, samples[0], color='blue', label="sample0")
    plt.plot(x, samples[1], color='red', label="sample1")
    plt.plot(x, samples[2], color='green', label="sample2")
    plt.plot(x, samples[3], color='purple', label="sample3")
    plt.plot(x, samples[4], color='pink', label="sample4")
    plt.plot(x, samples[5], color='orange', label="sample5")
    plt.plot(x, samples[6], color='yellow', label="sample6")
    plt.plot(x, samples[7], color='grey', label="sample7")

    plt.title('x')
    plt.ylabel('Avg Prediction')
    plt.xticks(x)
    plt.legend()
    #plt.xticklabels(labels)

    plt.savefig("barplotspitchavvgx.png")
    plt.show()


    # # radians to degrees
    p_test = 180.0 * p_test / np.pi
    # p_test = p_test.astype(int)




if __name__ == '__main__':
    main()