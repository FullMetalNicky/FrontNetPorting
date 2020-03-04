from Dataset import Dataset
from DataProcessor import DataProcessor
import logging
import numpy as np
import cv2
import torch

def TestDataset():

    path = "/Users/usi/PycharmProjects/data/Nicewall2.pickle"
    [x_test, y_test, z_test] = DataProcessor.ProcessTestData(path, 60, 108, True, True)
    test_set = Dataset(x_test, y_test)
    t_test = DataProcessor.GetTimeStampsFromTestData(path)





def main():
    print("here")
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

    TestDataset()


if __name__ == '__main__':
    main()
