
from __future__ import print_function
import logging
import cv2
import numpy as np
import sys


#sys.path.append("../PyTorch/")


from DataProcessor import DataProcessor


def Foveate(img, h, w):

    img = np.reshape(img, (h, w)).astype("uint8")
    fov_img = np.zeros((48, 80), dtype="uint8")

    # in  corners res is half on both axes

    fov_img[0:8, 0:13] = cv2.resize(img[0:32, 0:53], (13, 8), cv2.INTER_LINEAR)
    fov_img[-8:, 0:13] = cv2.resize(img[-32:, 0:53], (13, 8), cv2.INTER_LINEAR)
    fov_img[0:8, -13:] = cv2.resize(img[0:32, -53:], (13, 8), cv2.INTER_LINEAR)
    fov_img[-8:, -13:] = cv2.resize(img[-32:, -53:], (13, 8), cv2.INTER_LINEAR)

    # top/bottom center - every second row
    fov_img[0:8, 13:67] = cv2.resize(img[0:32, 53:107], (54, 8), cv2.INTER_LINEAR)
    fov_img[-8:, 13:67] = cv2.resize(img[-32:, 53:107], (54, 8), cv2.INTER_LINEAR)

    # left/right center - everey second column
    fov_img[8:40, 0:13] = cv2.resize(img[32:64, 0:53], (13, 32), cv2.INTER_LINEAR)
    fov_img[8:40, -13:] = cv2.resize(img[32:64, -53:], (13, 32), cv2.INTER_LINEAR)

    # center is full res
    fov_img[8:40, 13:67] = img[32:64, 53:107]
    cv2.imwrite("Ale_foveate.png", fov_img)




def Show(x_test):

    for i in range(len(x_test)):
        img = x_test[i].astype("uint8")
        cv2.putText(img, str(i), (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, 255)
        cv2.imshow("show", img)
        cv2.waitKey()


def DownScale():


    rnd = np.random.randint(0, 256, size=(16, 16)).astype("uint8")
    nn = cv2.resize(rnd, (4, 4), cv2.INTER_NEAREST)
    bl = cv2.resize(rnd, (4, 4), cv2.INTER_LINEAR)
    crop = rnd[4:8, 4:8]
    # cv2.imshow("show", rnd)
    # cv2.imshow("show1", nn)
    # cv2.imshow("show2", bl)
    # cv2.waitKey()

    cv2.imwrite("crop.png", crop)
    cv2.imwrite("nn.png", nn)
    cv2.imwrite("bl.png", bl)
    cv2.imwrite("orig.png", rnd)


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
    picklename = "160x96HimaxTrain16_4_2020.pickle"
    [x_test, y_test, z_test] = DataProcessor.ProcessTestData(DATA_PATH + picklename, True)
    h = x_test.shape[2]
    w = x_test.shape[3]
    x_test = np.reshape(x_test, (-1, h, w))


    # Foveate(x_test[255], h, w)
    # cv2.imwrite("Ale_full.png", x_test[255])
    # img = cv2.resize(x_test[255], (80, 48), cv2.INTER_LINEAR)
    # cv2.imwrite("Ale_lin.png", img)
    #
    # img = cv2.resize(x_test[255], (80, 48), cv2.INTER_NEAREST)
    # cv2.imwrite("Ale_NN.png", img)
    #
    # img = x_test[255][24:72, 40:120]
    # cv2.imwrite("Ale_crop.png", img)

    DownScale()





    #cv2.imwrite("full.png", x_test[0])


if __name__ == '__main__':
    main()