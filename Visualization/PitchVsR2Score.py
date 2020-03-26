



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



def DivideToBins(p_test, h, vfov):

    bin_num = 12
    pitch = p_test
    pitch = (35 - pitch) * vfov / h
    interval = np.linspace(np.min(pitch), np.max(pitch), bin_num)
    assignment = np.digitize(pitch, interval)
    ind = []

    for i in range(1, bin_num):
        res = np.where(assignment == i)
        ind.append(res)

    #ind = np.histogram(pitch, bins=bin_num)

    return ind, interval


def PitchvsR2Score(outputs, gt_labels, p_test, name, h, vfov):

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
    tot_pitch = []
    pitch_labels = []


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
        pitch_labels.append("{} - {}".format(np.around(interval[i], 2), np.around(interval[i+1], 2)))


    fig, ax = plt.subplots(2, 2, figsize=(16, 9))
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
    [x_test, y_test, z_test] = DataProcessor.ProcessTestData(DATA_PATH + picklename, True)
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


    PitchvsR2Score(outputs, gt_labels, p_test, picklename, 160, 65.65)


if __name__ == '__main__':
    main()