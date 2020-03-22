

import logging
import numpy as np
import pandas as pd
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import sys
sys.path.append("../PyTorch/")
from DataProcessor import DataProcessor



def VizHeatMap(dataset, ax):

    x = dataset[:, 1]
    y = dataset[:, 0]

    bins_num = 5

    heatmap, xedges, yedges = np.histogram2d(y, x, bins=bins_num)
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

    ax.set_xlabel('Y')
    ax.set_ylabel('X')
    ax.imshow(heatmap)

    for i in range(bins_num):
        for j in range(bins_num):
            line = "{}".format(heatmap[i][j])
            text = ax.text(j, i, line, ha="center", va="center", color="w")

    x_tick = np.linspace(xedges[0], xedges[-1], bins_num)
    x_tick = np.insert(x_tick, 0, -np.inf)
    y_tick = np.linspace(yedges[0], yedges[-1], bins_num)
    y_tick = np.insert(y_tick, 0, -np.inf)
    ax.set_xticklabels(np.around(x_tick, 2))
    ax.set_yticklabels(np.around(y_tick, 2))

    ax.invert_yaxis()
    ax.invert_xaxis()


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



def VizHeatMapsByAngle(dataset):
    df = pd.DataFrame({'x': dataset[:, 0], 'y': dataset[:, 1], 'z': dataset[:, 2], 'phi': dataset[:, 3]})
    walls = SortByAngle(df)

    fig, ax = plt.subplots(2, 2, figsize=(9, 9))
    plt.title("Heat Map by Walls")

    wallnames = ["curtains", "TV", "nice wall 1", "nice wall 2"]

    for w in range(len(walls)):
        w_set = dataset[walls[w]]
        idx = w % 2
        idy = int((w / 2) % 2)
        VizHeatMap(w_set, ax[idy][idx])
        title = ("Facing {}".format(wallnames[w]))
        ax[idy][idx].set_title(title)

    plt.savefig("wallsheatmap.png")
    plt.show()


def VizGeneralHeatMap(dataset):

    fig, ax = plt.subplots(1,1 , figsize=(9,9))
    ax.set_title("General Heat Map")
    VizHeatMap(dataset, ax)

    plt.savefig("generalheatmap.png")
    plt.show()

def ScatterPlot(dataset):

    fig, ax = plt.subplots()
    x = dataset[:, 1]
    y = dataset[:, 0]

    ax.scatter(x, y, alpha=0.003)
    ax.invert_xaxis()
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
    [x_test, y_test, z_test] = DataProcessor.ProcessTestData(DATA_PATH + "160x90HimaxMixedTrain_12_03_20.pickle", True)

    #VizGeneralHeatMap(z_test)
    #VizHeatMapsByAngle(z_test)
    #ScatterPlot(z_test)




if __name__ == '__main__':
    main()