



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
from PerformanceArchiver import LoadPerformanceResults


vertical_range = 70
half_vertical_range = vertical_range/2
val_range = [-14.5, 14.5]


def OffsetToPitch(p_test, h, vfov):
    pitch = -(half_vertical_range - p_test) * vfov / h

    return pitch

def DivideToBins(p_test, h, vfov):

    bin_num = 30
    # Negative pitch means looking up, positive pitch means looking down
    pitch = p_test
    pitch = -(half_vertical_range - pitch) * vfov / h
    interval = np.linspace(val_range[0], val_range[1], bin_num)
    assignment = np.digitize(pitch, interval)
    ind = []

    for i in range(1, bin_num+1):
        res = np.where(assignment == i)
        ind.append(res)

    return ind, interval




def CalculateR2ForPitch(outputs, gt_labels, range_p):

    tot_x_r2 = []
    tot_y_r2 = []
    tot_z_r2 = []
    tot_phi_r2 = []

    for i in range(range_p):
        output = outputs[:, i]
        label = gt_labels[:, i]

        x = output[:, 0]
        x_gt = label[:, 0]
        y = output[:, 1]
        y_gt = label[:, 1]
        z = output[:, 2]
        z_gt = label[:, 2]
        phi = output[:, 3]
        phi_gt = label[:, 3]

        x_r2 = sklearn.metrics.r2_score(x_gt, x)
        y_r2 = sklearn.metrics.r2_score(y_gt, y)
        z_r2 = sklearn.metrics.r2_score(z_gt, z)
        phi_r2 = sklearn.metrics.r2_score(phi_gt, phi)
        tot_x_r2.append(x_r2)
        tot_y_r2.append(y_r2)
        tot_z_r2.append(z_r2)
        tot_phi_r2.append(phi_r2)


    return tot_x_r2, tot_y_r2, tot_z_r2, tot_phi_r2


def PlotModelPred(ax, y_values, y_gt, x_values,  x_labels, len, skip, color, title, model_label, lim):
    ax.scatter(x_values, y_values, color=color, label=model_label)
    ax.scatter(x_values, y_gt, color='k', label='GT')
    ax.set_title(title)
    #ax.set_xticks(np.arange(0, len, skip))
    #ax.set_xticklabels(x_labels, rotation=30, fontsize=8)
    ax.set_xlabel('pitch')
    ax.set_ylabel('Prediction')
    ax.set_ylim([lim[0], lim[1]])
    ax.legend(fontsize=15)



def VizPitchvsPredSubPlots(ax, outputs, gt_labels, p_test, color, model_label):
    min_p = np.min(p_test)
    max_p = np.max(p_test)

    range_p = int(max_p - min_p + 1)
    print(max_p, min_p)
   # outputs = np.reshape(outputs, (-1, range_p, 4))
    #gt_labels = np.reshape(gt_labels, (-1, range_p, 4))
    tot_pitch = list(range(range_p))
    skip = 1
    pitch_labels = np.linspace(-20, 26, 47, endpoint=True)
    len_labels = len(tot_pitch) + 1
    print(len_labels)


    PlotModelPred(ax[0], outputs[:, 0], gt_labels[:, 0], p_test, pitch_labels, len_labels, skip, color, "x", model_label, [1, 2])
    PlotModelPred(ax[1], outputs[:, 1], gt_labels[:, 1], p_test, pitch_labels, len_labels, skip, color, "y", model_label, [-0.5, 0.5])
    PlotModelPred(ax[2], outputs[:, 2], gt_labels[:, 2], p_test, pitch_labels, len_labels, skip, color, "z", model_label, [-1, 1])
    PlotModelPred(ax[3], outputs[:, 3], gt_labels[:, 3], p_test, pitch_labels, len_labels, skip, color, "phi",
                  model_label, [-0.6, 0.6])


    return range_p


def Plot2Models(p_test, name, base_r2_score):


    fig, ax = plt.subplots(4, 1, figsize=(16, 12))
    fig.suptitle("Prediction as a function of Pitch")

    name1 = "pickles/DronetHimax160x90AugCropResults.pickle"
    outputs, gt_labels = LoadPerformanceResults(name1)
    range_p = VizPitchvsPredSubPlots(ax, outputs, gt_labels, p_test, 'b', 'pitch-augmented')

    name2 = "pickles/DronetHimax160x90AugmentedResults.pickle"
    outputs, gt_labels = LoadPerformanceResults(name2)
    range_p = VizPitchvsPredSubPlots(ax, outputs, gt_labels, p_test, 'g', 'non-augmented')

    if name.find(".pickle"):
        name = name.replace(".pickle", '')
    plt.savefig(name + '_predpitch.png')
    plt.show()


def Plot1Model(p_test, name):

    fig, ax = plt.subplots(4, 1, figsize=(16, 12))
    fig.suptitle("Prediction as a function of Pitch")

    name1 = "pickles/160x96Himax16_4_2020StaticPitch.pickle"
    outputs, gt_labels = LoadPerformanceResults(name1)

    outputs= outputs[70:120]
    gt_labels = gt_labels[70:120]
    p_test = p_test[70:120]

    range_p = VizPitchvsPredSubPlots(ax, outputs, gt_labels, p_test, 'b', 'Prediction')

    plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.35,
                        wspace=0.35)


    if name.find(".pickle"):
        name = name.replace(".pickle", '')
    plt.savefig(name + '_predpitch.png')
    plt.show()




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


    DATA_PATH = "/Users/usi/PycharmProjects/data/160x96/"
    #DATA_PATH = "/Users/usi/PycharmProjects/data/160x90/"

    # Get baseline results

    picklename = "160x96HimaxPitch16_4_2020.pickle"
    # picklename = "160x90HimaxMixedTest_12_03_20.pickle"
    # p_test = DataProcessor.GetPitchFromTestData(DATA_PATH + picklename)
    # [x_test, y_test] = DataProcessor.ProcessTestData(DATA_PATH + picklename)
    # test_set = Dataset(x_test, y_test)
    # params = {'batch_size': 1, 'shuffle': False, 'num_workers': 1}
    # test_generator = data.DataLoader(test_set, **params)
    # model = Dronet(PreActBlock, [1, 1, 1], True)
    # ModelManager.Read('../PyTorch/Models/DronetHimax160x90AugCrop.pt', model)
    # trainer = ModelTrainer(model)
    # MSE2, MAE2, r2_score2, outputs, gt_labels = trainer.Test(test_generator)

    # Get pitch values

    #picklename = "160x90HimaxMixedTest_12_03_20Cropped70.pickle"
    p_test = DataProcessor.GetPitchFromTestData(DATA_PATH + picklename)

    # radians to degrees
    p_test = 180.0 * p_test / np.pi
    p_test = p_test.astype(int)



    if picklename.find(".pickle"):
        picklename = picklename.replace(".pickle", '')


    #Plot2Models(p_test, picklename, r2_score2)


    Plot1Model(p_test, picklename)


if __name__ == '__main__':
    main()