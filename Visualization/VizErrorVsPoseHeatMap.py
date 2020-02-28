from __future__ import print_function

import logging
import numpy as np
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patches as patches

import sys

import pandas as pd


sys.path.append("../PyTorch/")
from DataProcessor import DataProcessor
from Dataset import Dataset
from PreActBlock import PreActBlock
from Dronet import Dronet
from DataProcessor import DataProcessor
from ModelTrainer import ModelTrainer
from ModelManager import ModelManager
from torch.utils import data

mincord = -1
maxcord = 1


def VizHeatMap(valid_loss, a, samples, xlen, ylen, w, min, max):

    wallnames = ["curtains", "TV", "nice wall 1", "nice wall 2"]

    # [(1,1), (1, 2)... , (3,2), (3,3)]
    valid_loss = np.asarray(valid_loss)
    samples = np.asarray(samples)
    # [[(1,1), (1, 2), (1,3)], ... , [(3,1), (3, 2), (3,3)]]
    valid_loss = np.reshape(valid_loss, (xlen, ylen))
    samples = np.reshape(samples, (xlen, ylen))
    # [[(1,3), (1, 2), (1,3)], ... , [(3,3), (3, 2), (3,1)]]
    # valid_loss = np.flip(valid_loss, 1)
    # samples = np.flip(samples, 1)

    idx = w % 2
    idy = int((w / 2) % 2)
    im = a[idy][idx].imshow(valid_loss)

    a[idy][idx].set_xticklabels(np.around([max[0], (max[0] + mincord) / 2, maxcord, 0, mincord, (min[0] + maxcord) / 2, min[0]], 2))
    a[idy][idx].set_yticklabels(np.around([max[1], (max[1] + maxcord) / 2, maxcord, 0, mincord, (min[0] + mincord) / 2, min[1]], 2))

    # Loop over data dimensions and create text annotations.
    for i in range(xlen):
        for j in range(ylen):
            loss = float("{0:.2f}".format(valid_loss[i, j]))
            sample = samples[i, j]
            line = "{}\n {}".format(loss, sample)
            text = a[idy][idx].text(j, i, line, ha="center", va="center", color="k")

    title = ("Facing {}".format(wallnames[w]))
    a[idy][idx].set_title(title)

def SortByAngle(df):

    low = (df['phi'] > -np.pi / 4)
    high = (df['phi'] < np.pi / 4)
    wall0 = df.index[low & high].tolist()

    low = (df['phi'] > np.pi / 4)
    high = (df['phi'] < 3 * np.pi / 4)
    wall1 = df.index[low & high].tolist()

    low = (df['phi'] < -np.pi / 4)
    high = (df['phi'] > -3 * np.pi / 4)
    wall3 = df.index[low & high].tolist()

    low = (df['phi'] < -3 * np.pi / 4)
    high = (df['phi'] > 3 * np.pi / 4)
    wall2 = df.index[low | high].tolist()

    return wall0, wall1, wall2, wall3


def SortByCoordinates(df, cord_id):

    high = (df[cord_id] > maxcord)
    cell1 = df.index[high].tolist()

    low = (df[cord_id] > mincord)
    high = (df[cord_id] < maxcord)
    cell2 = df.index[low & high].tolist()

    low = (df[cord_id] < mincord)
    cell3 = df.index[low].tolist()

    return cell1, cell2, cell3


def intersection(lst1, lst2):
    lst3 = [value for value in lst1 if value in lst2]
    return lst3


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

    DATA_PATH = "/Users/usi/PycharmProjects/data/"
    [x_test, y_test, z_test] = DataProcessor.ProcessTestData(DATA_PATH + "himaxposetest.pickle", 60, 108, True, True)
    model = Dronet(PreActBlock, [1, 1, 1], True)
    ModelManager.Read('../PyTorch/Models/DronetGray.pt', model)
    trainer = ModelTrainer(model)

    df = pd.DataFrame({'x': z_test[:, 0], 'y': z_test[:, 1], 'z': z_test[:,2], 'phi': z_test[:, 3]})
    walls = SortByAngle(df)

    fig, a = plt.subplots(2, 2, figsize=(9,9))

    for w in range(len(walls)):
        zw_set = z_test[walls[w]]
        xw_set = x_test[walls[w]]
        yw_set = y_test[walls[w]]
        df_w = pd.DataFrame({'x': zw_set[:, 0], 'y': zw_set[:, 1], 'z': zw_set[:, 2], 'phi': zw_set[:, 3]})

        min = np.amin(zw_set, axis=0)
        max = np.amax(zw_set, axis=0)

        xcells = SortByCoordinates(df_w, 'x')
        ycells = SortByCoordinates(df_w, 'y')

        valid_loss = []
        samples = []

        for i in range(len(xcells)):
            for j in range(len(ycells)):
                cell = intersection(xcells[i], ycells[j])
                if len(cell) > 0:
                    x = xw_set[cell]
                    y = yw_set[cell]
                    test_set = Dataset(x, y)
                    params = {'batch_size': 1, 'shuffle': False, 'num_workers': 0}
                    test_generator = data.DataLoader(test_set, **params)

                    MSE, MAE, r2_score, y_pred, gt_labels = trainer.Test(test_generator)
                    loss = MAE[1]
                else:
                    loss = 0
                valid_loss.append(loss)
                samples.append(len(cell))

        VizHeatMap(valid_loss, a, samples, len(xcells), len(ycells), w, min, max)

    fig.tight_layout()
    fig.suptitle('Y axis Error vs Pose')
    plt.savefig("yheatmap.png")
    plt.show()



if __name__ == '__main__':
    main()