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





def TestCamerasAgainstEachPther():
    images = ImageIO.ReadImagesFromFolder("../data/himax_processed/", '.jpg')
    [x_live, y_live] = DataProcessor.ProcessInferenceData(images)
    live_set = Dataset(x_live, y_live)
    params = {'batch_size': 1,
              'shuffle': False,
              'num_workers': 0}
    live_generator = data.DataLoader(live_set, **params)

    y_pred_himax = trainer.Infer(live_generator)
    y_pred_himax = np.reshape(y_pred_himax, (-1, 4))

    images = ImageIO.ReadImagesFromFolder("../data/bebop_processed/", '.jpg')
    [x_live, y_live] = DataProcessor.ProcessInferenceData(images)
    live_set = Dataset(x_live, y_live)
    params = {'batch_size': 1,
              'shuffle': False,
              'num_workers': 0}
    live_generator = data.DataLoader(live_set, **params)

    y_pred_bebop = trainer.Infer(live_generator)
    y_pred_bebop = np.reshape(y_pred_bebop, (-1, 4))

    for i in range(len(y_pred_bebop)):
        print("sample {}:".format(i))
        himax_output = y_pred_himax[i]
        print("himax prediction is {}, {}, {}, {}".format(himax_output[0], himax_output[1], himax_output[2],
                                                          himax_output[3]))
        bebop_output = y_pred_bebop[i]
        print("himax prediction is {}, {}, {}, {}".format(bebop_output[0], bebop_output[1], bebop_output[2],
                                                          bebop_output[3]))

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



    model = FrontNet(PreActBlock, [1, 1, 1])
    trainer = ModelTrainer(model)

    DATA_PATH = "/Users/usi/PycharmProjects/data/"

    #DataProcessor.CreateGreyPickle(DATA_PATH + "test.pickle", 60, 108)
    [train_mean, train_std, x_train, x_validation, y_train, y_validation] = DataProcessor.ProcessTrainDataGray(DATA_PATH + "train_gray.pickle", 60, 108)


#    [train_mean, train_std, x_train, x_validation, y_train, y_validation] = DataProcessor.ProcessTrainData(DATA_PATH + "train.pickle", 60, 108)
    training_set = Dataset(x_train, y_train, True)
    params = {'batch_size': 64,
               'shuffle': True,
               'num_workers': 0}
    training_generator = data.DataLoader(training_set, **params)

    validation_set = Dataset(x_validation, y_validation)
    validation_generator = data.DataLoader(validation_set, **params)

    # [x_test, y_test] = DataProcessor.ProcessTestDataGray(DATA_PATH + "test_gray.pickle", 60, 108)
    # test_set = Dataset(x_test, y_test)
    # params = {'batch_size': 1,
    #          'shuffle': False,
    #          'num_workers': 0}
    # test_generator = data.DataLoader(test_set, **params)

    trainer.Train(training_generator, validation_generator)
    #trainer.Predict(test_generator)


if __name__ == '__main__':
    main()