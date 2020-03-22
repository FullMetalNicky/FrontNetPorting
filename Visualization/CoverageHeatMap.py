

import logging
import numpy as np
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import sys
sys.path.append("../PyTorch/")
from DataProcessor import DataProcessor



def VizGeneralHeatMap(dataset):
    x = dataset[:, 0]
    y = dataset[:, 1]

    bins_num = 5

    heatmap, xedges, yedges = np.histogram2d(y, x, bins=bins_num)
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]


    fig, ax = plt.subplots(1,1 , figsize=(9,9))
    ax.set_title("General Heat Map")
    ax.set_xlabel('Y')
    ax.set_ylabel('X')
    ax.imshow(heatmap.T)

    for i in range(bins_num):
        for j in range(bins_num):
            line = "{}".format(heatmap[i][j])
            text = ax.text(i, j, line, ha="center", va="center", color="w")


    x_tick = np.linspace(xedges[0], xedges[-1], bins_num)
    x_tick = np.insert(x_tick, 0, -np.inf)
    y_tick = np.linspace(yedges[0], yedges[-1], bins_num)
    y_tick = np.insert(y_tick, 0, -np.inf)
    ax.set_xticklabels(np.around(x_tick, 2))
    ax.set_yticklabels(np.around(y_tick, 2))

    plt.savefig("generalheatmap.png")
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

    VizGeneralHeatMap(z_test)




if __name__ == '__main__':
    main()