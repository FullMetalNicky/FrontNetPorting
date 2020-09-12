import pandas as pd
import numpy as np
import sklearn.metrics
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import logging
import sys
from torch.utils import data

sys.path.append("../PyTorch/")

from PreActBlock import PreActBlock
from FrontNet import FrontNet
from DataProcessor import DataProcessor
from ModelTrainer import ModelTrainer
from Dataset import Dataset
from ModelManager import ModelManager


def PlotGTVsEstimation(x_c, y_c, z_c, phi_c, x_py, y_py, z_py, phi_py):
    fig = plt.figure(666, figsize=(16, 10))



    gs = gridspec.GridSpec(2, 2)
    ax = plt.subplot(gs[0, 0])
    ax.set_title('output variable: x')
    ax.set_xlabel('Ground Truth')
    ax.set_ylabel('PyTorch model prediction')
    #ax.set_xmargin(0.2)

    plt.scatter(x_py, x_c, color='green', marker='o')
    plt.plot(x_py, x_py, color='black')


    ax = plt.subplot(gs[0, 1])
    ax.set_title('output variable: y')
    ax.set_xlabel('Ground Truth')
    ax.set_ylabel('PyTorch model prediction')
    #ax.set_xmargin(0.2)

    plt.scatter(y_py, y_c, color='blue', marker='*')
    plt.plot(y_py, y_py, color='black')

    ax = plt.subplot(gs[1,0])
    ax.set_title('output variable: z')
    ax.set_xlabel('Ground Truth')
    ax.set_ylabel('PyTorch model prediction')
    # ax.set_xmargin(0.2)

    plt.scatter(z_py, z_c, color='purple', marker='^')
    plt.plot(z_py, z_py, color='black')

    ax = plt.subplot(gs[1,1])
    ax.set_title('output variable: phi')
    ax.set_xlabel('Ground Truth')
    ax.set_ylabel('PyTorch model prediction')
    #ax.set_xmargin(0.2)

    plt.scatter(phi_py, phi_c, color='orangered', marker='D')
    plt.plot(phi_py, phi_py, color='black')


    #plt.subplots_adjust(hspace=0.3)
    plt.suptitle('Ground Truth vs PyTorch Prediction')
    gs.tight_layout(fig, rect=[0, 0.03, 1, 0.95])
    plt.savefig('GTvsPython.png')


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

    size = "108x60"
    DATA_PATH = "/Users/usi/PycharmProjects/data/" + size + "/"
    picklename = size + "PaperTestsetPrune2.pickle"

    # [x_test, y_test] = DataProcessor.ProcessTestData(DATA_PATH + picklename)
    # model = FrontNet(PreActBlock, [1, 1, 1], True)
    # ModelManager.Read('../PyTorch/Models/FrontNet108x60Others.pt', model)

    [x_test, y_test] = DataProcessor.ProcessTestDataAsRGB(DATA_PATH + picklename)
    model = FrontNet(PreActBlock, [1, 1, 1], False)
    ModelManager.Read('../PyTorch/Models/FrontNet108x60Dario.pt', model)


    test_set = Dataset(x_test, y_test)
    params = {'batch_size': 1,
              'shuffle': False,
              'num_workers': 1}
    test_generator = data.DataLoader(test_set, **params)
    trainer = ModelTrainer(model)
    MSE, MAE, r2_score, outputs, gt_labels = trainer.Test(test_generator)

    PlotGTVsEstimation(outputs[:,0], outputs[:,1], outputs[:,2], outputs[:,3], y_test[:, 0], y_test[:, 1], y_test[:, 2], y_test[:, 3])






if __name__ == '__main__':
    main()