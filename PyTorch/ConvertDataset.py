import pandas as pd
import numpy as np
import random
import logging
import cv2
import sys



def main():
    size = "108x60"
    DATA_PATH = "/Users/usi/PycharmProjects/data/" + size + "/old/"
    picklename = "testRGB.pickle"
    train_set = pd.read_pickle(DATA_PATH + picklename).values

    logging.info('[DataProcessor] train shape: ' + str(train_set.shape))

    x_train = train_set[:, 0]

    # viz = DataVisualization()
    # viz.DisplayDatasetVideo(x_train)

    # (63726, 60, 108, 3)
    # x_train = np.reshape(x_train, (-1, 3, image_height, image_width))
    # (63726, 3, 60, 108)

    y_train = train_set[:, 1]


    sizes = pd.DataFrame({
        'c': 3,
        'w': 108,
        'h': 60
    }, index=[0])

    data = pd.DataFrame(data={'x': x_train, 'y': y_train})
    df = pd.concat([data, sizes], axis=1)
    df.to_pickle(DATA_PATH + "108x60BebopRGBTest.pickle")







if __name__ == '__main__':
    main()