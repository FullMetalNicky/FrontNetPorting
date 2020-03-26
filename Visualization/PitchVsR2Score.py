



from __future__ import print_function
import logging
import numpy as np
import matplotlib.pyplot as plt
from torch.utils import data
import sys
import sklearn.metrics


sys.path.append("../PyTorch/")

from PreActBlock import PreActBlock
from Dronet import Dronet
from DataProcessor import DataProcessor
from ModelTrainer import ModelTrainer
from Dataset import Dataset
from ModelManager import ModelManager


vertical_range = 70
half_vertical_range = vertical_range/2
val_range = [-14.5, 14.5]


def DivideToBins(p_test, h, vfov):

    bin_num = 30
    # Negative pitch means looking up, positive pitch means looking down
    pitch = p_test
    pitch = -(half_vertical_range - pitch) * vfov / h
    interval = np.linspace(val_range[0], val_range[1], bin_num)
    assignment = np.digitize(pitch, interval)
    ind = []

    for i in range(1, bin_num):
        res = np.where(assignment == i)
        ind.append(res)

    return ind, interval


def PitchvsR2Score(outputs, gt_labels, p_test, name, h, vfov, base_r2_score=None):

    ind, interval = DivideToBins(p_test, h, vfov)

    x = outputs[:, 0]
    x_gt = gt_labels[:, 0]

    y = outputs[:, 1]
    y_gt = gt_labels[:, 1]

    z = outputs[:, 2]
    z_gt = gt_labels[:, 2]

    phi = outputs[:, 3]
    phi_gt = gt_labels[:, 3]

    tot_x_r2=[]
    tot_y_r2 = []
    tot_z_r2 = []
    tot_phi_r2 = []
    pitch_labels = []
    tot_pitch = []

    for i in range(len(ind)):

        p_ind = ind[i]
        x_r2 = sklearn.metrics.r2_score(x_gt[p_ind], x[p_ind])
        y_r2 = sklearn.metrics.r2_score(y_gt[p_ind], y[p_ind])
        z_r2 = sklearn.metrics.r2_score(z_gt[p_ind], z[p_ind])
        phi_r2 = sklearn.metrics.r2_score(phi_gt[p_ind], phi[p_ind])
        tot_x_r2.append(x_r2)
        tot_y_r2.append(y_r2)
        tot_z_r2.append(z_r2)
        tot_phi_r2.append(phi_r2)
        tot_pitch.append(i)
        pitch_labels.append("{}".format(int((interval[i] +interval[i + 1]) / 2)))


    fig, ax = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("R2 Score as a function of Pitch")

    ax[0][0].plot(tot_pitch, tot_x_r2)
    ax[0][0].set_title("x")
    ax[0][0].set_xticks(np.arange(len(pitch_labels)))
    ax[0][0].set_xticklabels(pitch_labels, rotation=30, fontsize=8)
    ax[0][0].set_xlabel('pitch')
    ax[0][0].set_ylabel('R2')

    ax[0][1].plot(tot_pitch, tot_y_r2)
    ax[0][1].set_title("y")
    ax[0][1].set_xticks(np.arange(len(pitch_labels)))
    ax[0][1].set_xticklabels(pitch_labels, rotation=30, fontsize=8)
    ax[0][1].set_xlabel('pitch')
    ax[0][1].set_ylabel('R2')

    ax[1][0].plot(tot_pitch, tot_z_r2)
    ax[1][0].set_title("z")
    ax[1][0].set_xticks(np.arange(len(pitch_labels)))
    ax[1][0].set_xticklabels(pitch_labels, rotation=30, fontsize=8)
    ax[1][0].set_xlabel('pitch')
    ax[1][0].set_ylabel('R2')

    ax[1][1].plot(tot_pitch, tot_phi_r2)
    ax[1][1].set_title("phi")
    ax[1][1].set_xticks(np.arange(len(pitch_labels)))
    ax[1][1].set_xticklabels(pitch_labels, rotation=30, fontsize=8)
    ax[1][1].set_xlabel('pitch')
    ax[1][1].set_ylabel('R2')

    if base_r2_score is not None:
        ax[0][0].hlines(base_r2_score[0], 0, len(pitch_labels), colors='r', label='Base')
        ax[0][0].legend()
        ax[0][1].hlines(base_r2_score[1], 0, len(pitch_labels), colors='r', label='Base')
        ax[0][1].legend()
        ax[1][0].hlines(base_r2_score[2], 0, len(pitch_labels), colors='r', label='Base')
        ax[1][0].legend()
        ax[1][1].hlines(base_r2_score[3], 0, len(pitch_labels), colors='r', label='Base')
        ax[1][1].legend()


    if name.find(".pickle"):
        name = name.replace(".pickle", '')
    plt.savefig(name + '_pitch.png')
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

    model = Dronet(PreActBlock, [1, 1, 1], True)
    ModelManager.Read('../PyTorch/Models/DronetHimax160x90Augmented.pt', model)

    DATA_PATH = "/Users/usi/PycharmProjects/data/160x90/"
    picklename = "160x90HimaxMixedTest_12_03_20Cropped.pickle"
    [x_test, y_test] = DataProcessor.ProcessTestData(DATA_PATH + picklename)
    p_test = DataProcessor.GetPitchFromTestData(DATA_PATH + picklename)

    test_set = Dataset(x_test, y_test)
    params = {'batch_size': 1,
              'shuffle': False,
              'num_workers': 1}
    test_generator = data.DataLoader(test_set, **params)
    trainer = ModelTrainer(model)

    MSE, MAE, r2_score, outputs, gt_labels = trainer.Test(test_generator)
    gt_labels = np.reshape(gt_labels, (-1, 4))

    if picklename.find(".pickle"):
        picklename = picklename.replace(".pickle", '')

    picklename2 = "160x90HimaxMixedTest_12_03_20.pickle"
    [x_test2, y_test2] = DataProcessor.ProcessTestData(DATA_PATH + picklename2)
    test_set2 = Dataset(x_test2, y_test2)
    params = {'batch_size': 1,
              'shuffle': False,
              'num_workers': 1}
    test_generator = data.DataLoader(test_set2, **params)
    trainer = ModelTrainer(model)
    MSE2, MAE2, r2_score2, outputs2, gt_labels2 = trainer.Test(test_generator)
    base = r2_score2

    PitchvsR2Score(outputs, gt_labels, p_test, picklename, 160, 65.65, base)


if __name__ == '__main__':
    main()