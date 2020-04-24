from DataManipulator import DataManipulator
import logging
import numpy as np
import cv2
import pandas as pd
import os




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

    pickle_folder = "/Users/usi/PycharmProjects/data/16_4_2020/Pitch/"
    files = os.listdir(pickle_folder)

    # train_dir = pickle_folder + "Train/"
    # test_dir = pickle_folder + "Test/"
    # if not os.path.exists(train_dir):
    #     os.makedirs(train_dir)
    # if not os.path.exists(test_dir):
    #     os.makedirs(test_dir)
    #
    #
    # train_pickle_list=[]
    # test_pickle_list = []
    #
    # for f in files:
    #     if ".pickle" in f:
    #         DataManipulator.DivideDataset(pickle_folder+f, train_dir+f, test_dir+f, 0.7)
    #         train_pickle_list.append(train_dir+f)
    #         test_pickle_list.append(test_dir + f)
    #
    # DataManipulator.JoinDatasetFromList(train_pickle_list, train_dir + "160x160HimaxHeightTrain16_4_2020.pickle")
    # DataManipulator.JoinDatasetFromList(test_pickle_list, test_dir + "160x160HimaxHeightTest16_4_2020.pickle")

    pickle_list = []
    for f in files:
        if ".pickle" in f:
            pickle_list.append(pickle_folder + f)

    DataManipulator.JoinDatasetFromList(pickle_list, pickle_folder + "160x160HimaxPitch16_4_2020.pickle")


if __name__ == '__main__':
    main()