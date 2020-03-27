

import logging
import numpy as np
import pandas as pd
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import sys
sys.path.append("../PyTorch/")
from DataProcessor import DataProcessor
from Dataset import Dataset
from PreActBlock import PreActBlock
from Dronet import Dronet
from DataProcessor import DataProcessor
from ModelTrainer import ModelTrainer
from ModelManager import ModelManager
from torch.utils import data



def CalcError(trainer, x, y):

    test_set = Dataset(x, y)
    params = {'batch_size': 1, 'shuffle': False, 'num_workers': 0}
    test_generator = data.DataLoader(test_set, **params)

    MSE, MAE, r2_score, y_pred, gt_labels = trainer.Test(test_generator)

    return MAE

def DivideData(x, y, bin_num, xedges, yedges):


    assignment_x = np.digitize(x, xedges, right=True)
    assignment_y = np.digitize(y, yedges, right=True)

    indx = []
    indy = []

    for i in range(bin_num):
        res = np.where(assignment_x == i)
        indx.append(res)
        res = np.where(assignment_y == i)
        indy.append(res)

    return indx,indy


def CalcHeatmap(dataset):

    x = dataset[:, 0]
    y = dataset[:, 1]

    hist_bins = [-2.5, -1.5, -0.5, 0.5, 1.5, 2.5]
    bin_num = len(hist_bins) - 1

    heatmap, xedges, yedges = np.histogram2d(x, y, bins=hist_bins)

    indx,indy = DivideData(x, y, bin_num, xedges[1:-1], yedges[1:-1])

    return heatmap, indx, indy


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



def GetSubplotIndex(w):
    idx = w % 2
    idy = int((w / 2) % 2)

    return idx, idy


def VizHeatMapsByAngle(pose, frames, labels, trainer):

    df = pd.DataFrame({'x': pose[:, 0], 'y': pose[:, 1], 'z': pose[:, 2], 'phi': pose[:, 3]})
    walls = SortByAngle(df)

    fig, ax = plt.subplots(2, 2, figsize=(9, 9))
    fig.suptitle("Error HeatMap For Different Orientations")

    wallnames = ["curtains", "TV", "nice wall 1", "nice wall 2"]
    bins_num = 5


    for w in range(len(walls)):
        w_pose = pose[walls[w]]
        heatmap, indx, indy = CalcHeatmap(w_pose)
        w_frames = frames[walls[w]]
        w_labels = labels[walls[w]]
        idx, idy = GetSubplotIndex(w)
        ax[idy][idx].imshow(heatmap)

        ax[idy][idx].set_xlabel('Y')
        ax[idy][idx].set_ylabel('X')

        for i in range(bins_num):
            for j in range(bins_num):
                cell_ind = np.intersect1d(indx[i], indy[j])
                cell_val = int(heatmap[i][j])

                if len(cell_ind) > 0:
                    error = CalcError(trainer, w_frames[cell_ind], w_labels[cell_ind])
                    loss = float("{0:.2f}".format(error[0]))
                    line = "{}\n {}".format(loss, cell_val)
                else:
                    line = "{}".format(cell_val)

                text = ax[idy][idx].text(j, i, line, ha="center", va="center", color="w")


        ax[idy][idx].invert_yaxis()
        ax[idy][idx].invert_xaxis()
        title = ("Facing {}".format(wallnames[w]))
        ax[idy][idx].set_title(title)

    plt.savefig("ErrorHeatMap.png")
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

    DATA_PATH = "/Users/usi/PycharmProjects/data/160x90/"
    [x_test, y_test, z_test] = DataProcessor.ProcessTestData(DATA_PATH + "160x90HimaxMixedTest_12_03_20.pickle", True)

    model = Dronet(PreActBlock, [1, 1, 1], True)
    ModelManager.Read('../PyTorch/Models/DronetHimax160x90Augmented.pt', model)
    trainer = ModelTrainer(model)


    VizHeatMapsByAngle(z_test, x_test, y_test, trainer)



if __name__ == '__main__':
    main()