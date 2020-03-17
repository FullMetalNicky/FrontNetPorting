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
    name = "160x90HimaxStatic_12_03_20.pickle"
    #name =

    [x_test, y_test, z_test] = DataProcessor.ProcessTestData(DATA_PATH + name, True)
    h = x_test.shape[2]
    w = x_test.shape[3]
    x_test = np.reshape(x_test, (-1, h, w))


    phi= y_test[:, 3]

    plt.hist(phi, bins=10)
    if name.find(".pickle"):
        name = name.replace(".pickle", '')
    plt.savefig(name + '.png')
    plt.show()



if __name__ == '__main__':
    main()