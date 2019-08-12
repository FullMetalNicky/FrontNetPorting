from __future__ import print_function
from FrontNet import PreActBlock
from FrontNet import FrontNet
from DataProcessor import DataProcessor
from ModelTrainer import ModelTrainer
from Dataset import Dataset
from torch.utils import data
from ModelManager import ModelManager
import logging
import numpy as np
import sys
sys.path.append("../pulp/")
from ImageIO import ImageIO
import sklearn.metrics





def TestCamerasAgainstEachPther():
    model = FrontNet(PreActBlock, [1, 1, 1])
    ModelManager.Read('Models/FrontNetGray-096.pt', model)
    trainer = ModelTrainer(model)


    images = ImageIO.ReadImagesFromFolder("../data/himax_processed/", '.jpg', 0)
    [x_live, y_live] = DataProcessor.ProcessInferenceData(images, 60, 108)
    live_set = Dataset(x_live, y_live)
    params = {'batch_size': 1,
              'shuffle': False,
              'num_workers': 0}
    live_generator = data.DataLoader(live_set, **params)

    y_pred_himax = trainer.Infer(live_generator)
    y_pred_himax = np.reshape(y_pred_himax, (-1, 4))

    images = ImageIO.ReadImagesFromFolder("../data/bebop_processed/", '.jpg', 0)
    [x_live, y_live] = DataProcessor.ProcessInferenceData(images, 60, 108)
    live_set = Dataset(x_live, y_live)
    params = {'batch_size': 1,
              'shuffle': False,
              'num_workers': 0}
    live_generator = data.DataLoader(live_set, **params)

    y_pred_bebop = trainer.Infer(live_generator)
    y_pred_bebop = np.reshape(y_pred_bebop, (-1, 4))

    for i in range(len(y_pred_bebop)):
        logging.info("sample {}:".format(i))
        himax_output = y_pred_himax[i]
        logging.info("himax prediction is {}, {}, {}, {}".format(himax_output[0], himax_output[1], himax_output[2],
                                                          himax_output[3]))
        bebop_output = y_pred_bebop[i]
        logging.info("bebop prediction is {}, {}, {}, {}".format(bebop_output[0], bebop_output[1], bebop_output[2],
                                                          bebop_output[3]))

    MSE = np.mean(np.power(y_pred_bebop - y_pred_himax, 2), 0)
    MAE = np.mean(np.fabs(y_pred_bebop - y_pred_himax), 0)
    logging.info("MSE is {}, {}, {}, {}".format(MSE[0], MSE[1], MSE[2], MSE[3]))
    logging.info("MAE is {}, {}, {}, {}".format(MAE[0], MAE[1], MAE[2], MAE[3]))


    x = y_pred_bebop[:, 0]
    x_gt = y_pred_himax[:, 0]

    y = y_pred_bebop[:, 1]
    y_gt = y_pred_himax[:, 1]

    z = y_pred_bebop[:, 2]
    z_gt = y_pred_himax[:, 2]

    phi = y_pred_bebop[:, 3]
    phi_gt = y_pred_himax[:, 3]

    x_r2 = sklearn.metrics.r2_score(x_gt, x)
    y_r2 = sklearn.metrics.r2_score(y_gt, y)
    z_r2 = sklearn.metrics.r2_score(z_gt, z)
    phi_r2 = sklearn.metrics.r2_score(phi_gt, phi)

    logging.info("r^2 score is {}, {}, {}, {}".format(x_r2, y_r2, z_r2, phi_r2))


def TrainGray():
    model = FrontNet(PreActBlock, [1, 1, 1])
    trainer = ModelTrainer(model)

    DATA_PATH = "/Users/usi/PycharmProjects/data/"
    [train_mean, train_std, x_train, x_validation, y_train, y_validation] = DataProcessor.ProcessTrainDataGray(
        DATA_PATH + "train_gray.pickle", 60, 108)

    training_set = Dataset(x_train, y_train, True)
    params = {'batch_size': 64,
              'shuffle': True,
              'num_workers': 0}
    training_generator = data.DataLoader(training_set, **params)

    validation_set = Dataset(x_validation, y_validation)
    validation_generator = data.DataLoader(validation_set, **params)

    trainer.Train(training_generator, validation_generator)

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

    TestCamerasAgainstEachPther()



if __name__ == '__main__':
    main()