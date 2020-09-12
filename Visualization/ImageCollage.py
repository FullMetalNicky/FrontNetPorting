

import logging
import numpy as np
import pandas as pd
import cv2

import sys
sys.path.append("../PyTorch/")
from DataProcessor import DataProcessor


def CollageMakerRawVsAugmented(raw, aug, name):

    l = len(raw)
    n = 25
    ind = np.random.randint(0,l, n)
    selected_raw = raw[ind]
    selected_aug = aug[ind*10]
    c ,h, w = selected_raw[0].shape
    collage = np.zeros((5*h, 5*w))
    collage2 = np.zeros((5 * h, 5 * w))

    for i in range(5):
        for j in range(5):
            tmp = np.reshape(selected_raw[i*5+j], (h, w))
            collage[h*i:h*(i+1), w*j:w*(j+1)] = tmp
            tmp = np.reshape(selected_aug[i * 5 + j], (h, w))
            collage2[h * i:h * (i + 1), w * j:w * (j + 1)] = tmp

    cv2.imwrite(name + "_raw.png", collage)
    cv2.imwrite(name + "_aug.png", collage2)


def CollageMaker(frames, name):

    l = len(frames)
    n = 25
    ind = np.random.randint(0,l, n)
    selected = frames[ind]
    c ,h, w = selected[0].shape

    collage = np.zeros((5*h, 5*w))

    for i in range(5):
        for j in range(5):
            tmp = np.reshape(selected[i*5+j], (h, w))
            collage[h*i:h*(i+1), w*j:w*(j+1)] = tmp

    cv2.imwrite(name + ".png", collage)


def PitchCollage(orig, name):

    desiresSize = (96, 160)
    c, h, w = orig.shape
    col_Size = 3
    hor_offset = int((w - desiresSize[1]) / 2)
    vertical_range = h - desiresSize[0]
    collage = np.zeros((col_Size * desiresSize[0], col_Size * desiresSize[1]))

    for i in range(col_Size):
        for j in range(col_Size):
            crop_offset = np.random.randint(0, vertical_range)
            img = np.reshape(orig, (h, w)).astype("uint8")
            img = img[crop_offset:(crop_offset + desiresSize[0]), hor_offset: (hor_offset + desiresSize[1])]
            collage[desiresSize[0] * i: desiresSize[0] * (i + 1), desiresSize[1] * j:desiresSize[1] * (j + 1)] = img

    cv2.imwrite(name + "_pitch.png", collage)


def RollCollage(frames, name):

    c, h, w = frames[0].shape
    col_Size = 3
    collage = np.zeros((col_Size * h, col_Size * w))


    for i in range(col_Size):
        for j in range(col_Size):
            id = np.random.randint(0, 29)
            img = np.reshape(frames[id], (h, w)).astype("uint8")
            collage[h * i:h * (i + 1), w * j:w * (j + 1)] = img


    cv2.imwrite(name + "_roll.png", collage)


def ResizeCollage(orig, name):
    c, h, w = orig.shape
    col_Size = 2
    collage = np.zeros((col_Size * h, col_Size * w))
    sizes = [[160,96], [80, 48], [40,24], [20, 12]]

    for i in range(col_Size):
        for j in range(col_Size):
            ds_size = sizes[i*2+j]
            img = np.reshape(orig, (h, w)).astype("uint8")
            img = cv2.resize(img, (ds_size[1], ds_size[0]), interpolation=cv2.INTER_LINEAR)
            img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)
            img = img.astype(np.uint8)
            collage[h * i:h * (i + 1), w * j:w * (j + 1)] = img

    cv2.imwrite(name + "_pixinfo.png", collage)







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

    DATA_PATH = "/Users/usi/PycharmProjects/data/160x96/"
    # [x_test, y_test] = DataProcessor.ProcessTestData(DATA_PATH + "160x96HimaxTrain16_4_2020AugCrop.pickle")
    [x_test, y_test2] = DataProcessor.ProcessTestData(DATA_PATH + "160x96HimaxTrain16_4_2020Aug.pickle")
    #DATA_PATH = "/Users/usi/PycharmProjects/data/160x160/"
    #[x_test3, y_test3] = DataProcessor.ProcessTestData(DATA_PATH + "160x160HimaxTrain16_4_2020.pickle")
    # PitchCollage(x_test3[1], "others")

    #ResizeCollage(x_test2[300], "others")

    # DATA_PATH = "/Users/usi/PycharmProjects/data/160x90/"
    # [x_test, y_test3] = DataProcessor.ProcessTestData(DATA_PATH + "160x90HimaxMixedTest_12_03_20Rot.pickle")
    # RollCollage(x_test, "nicky")
    for i in range(len(x_test)):
        cv2.imshow("aug", np.reshape(x_test[i], (96, 160)).astype("uint8"))
        #cv2.imshow("raw", np.reshape(x_test2[i], (96, 160)).astype("uint8"))
        cv2.waitKey()


    #CollageMaker(x_test, "nicky_raw_frames")
    #CollageMakerRawVsAugmented(x_test2, x_test, "others")


if __name__ == '__main__':
    main()