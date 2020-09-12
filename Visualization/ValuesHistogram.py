from __future__ import print_function
import logging
import numpy as np
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from torch.utils import data
import sys
import matplotlib.patches as patches

sys.path.append("../PyTorch/")


from DataProcessor import DataProcessor

def VizHistogram(y_test, name):
    fig, ax = plt.subplots(2, 2, figsize=(9, 9))
    fig.suptitle("Pose Range Histogram")
    bins_num = 10

    x = y_test[:, 0]
    y = y_test[:, 1]
    z = y_test[:, 2]
    phi = y_test[:, 3]

    ax[0][0].hist(x, bins=bins_num)
    ax[0][0].set_title("x")

    ax[0][1].hist(y, bins=bins_num)
    ax[0][1].set_title("y")

    ax[1][0].hist(z, bins=bins_num)
    ax[1][0].set_title("z")

    ax[1][1].hist(phi, bins=bins_num)
    ax[1][1].set_title("phi")


    if name.find(".pickle"):
        name = name.replace(".pickle", '')
    plt.savefig(name + '_histogram.png')
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


    DATA_PATH = "/Users/usi/PycharmProjects/data/108x60/old/"
    name = "bebop_train_grey.pickle"

    [x_test, y_test] = DataProcessor.ProcessTestData(DATA_PATH + name)

    VizHistogram(y_test, name)




if __name__ == '__main__':
    main()