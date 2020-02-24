from Dataset import Dataset
from DataProcessor import DataProcessor
import logging
import numpy as np
import cv2
import torch

def TestDataset():

    [x_test, y_test] = DataProcessor.ProcessTestData("/Users/usi/PycharmProjects/data/beboptest.pickle", 60, 108, True)
    test_set = Dataset(x_test, y_test)

    frame = cv2.imread("13.jpg", 0)
    frame = np.reshape(frame, (60, 108, 1))
    frame = np.swapaxes(frame, 0, 2)
    frame = np.swapaxes(frame, 1, 2)

    for i in range(20):
        newframe = torch.from_numpy(frame).float()
        newframe = test_set.augmentNoise(newframe)
        newframe = test_set.toNumpy(newframe)
        cv2.imshow("frame", newframe)
        cv2.waitKey()




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

    print("here")
    TestDataset()


if __name__ == '__main__':
    main()
