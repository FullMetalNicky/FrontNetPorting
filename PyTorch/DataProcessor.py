import pandas as pd
import numpy as np
import random
import logging
import cv2
import matplotlib.pyplot as plt
import sys
sys.path.append("../pulp/")
from ImageTransformer import ImageTransformer



class DataProcessor:

    @staticmethod
    def ProcessTrainData(trainPath, image_height, image_width):
        train_set = pd.read_pickle(trainPath).values
        logging.info('[DataProcessor] train shape: ' + str(train_set.shape))

        n_val = 13000
        np.random.seed()

        # split between train and test sets:
        x_train = train_set[:, 0]
        x_train = np.vstack(x_train[:]).astype(np.float32)
        x_train = np.reshape(x_train, (-1, image_height, image_width, 3))

        x_train= np.swapaxes(x_train, 1, 3)
        x_train = np.swapaxes(x_train, 2, 3)

        y_train = train_set[:, 1]
        y_train = np.vstack(y_train[:]).astype(np.float32)

        ix_val, ix_tr = np.split(np.random.permutation(x_train.shape[0]), [n_val])
        x_validation = x_train[ix_val, :]
        x_train = x_train[ix_tr, :]
        y_validation = y_train[ix_val, :]
        y_train = y_train[ix_tr, :]
        train_mean  = np.mean(y_train, 0)
        train_std = np.std(y_train, 0)

        shape_ = x_train.shape[0]
        sel_idx = random.sample(range(0, shape_), k=50000)
        x_train = x_train[sel_idx, :]
        y_train = y_train[sel_idx, :]

        return [train_mean, train_std, x_train, x_validation, y_train, y_validation]

    @staticmethod
    def ProcessTestData(testPath, image_height, image_width):
        test_set = pd.read_pickle(testPath).values
        logging.info('[DataProcessor] test shape: ' + str(test_set.shape))

        x_test = test_set[:, 0]
        x_test = np.vstack(x_test[:]).astype(np.float32)
        x_test = np.reshape(x_test, (-1, image_height, image_width, 3))
        x_test = np.swapaxes(x_test, 1, 3)
        x_test = np.swapaxes(x_test, 2, 3)
        y_test = test_set[:, 1]
        y_test = np.vstack(y_test[:]).astype(np.float32)

        return [x_test, y_test]

    @staticmethod
    def ProcessInferenceData(images, image_height, image_width):

        x_test = np.stack(images, axis=0).astype(np.float32)
        x_test = np.reshape(x_test, (-1, image_height, image_width, 1))
        x_test = np.swapaxes(x_test, 1, 3)
        x_test = np.swapaxes(x_test, 2, 3)
        y_test = [0, 0, 0, 0] * len(x_test)
        y_test = np.vstack(y_test[:]).astype(np.float32)
        y_test = np.reshape(y_test, (-1, 4))


        return [x_test, y_test]

    @staticmethod
    def CreateGreyPickle(trainPath, image_height, image_width, file_name="train.pickle"):
        train_set = pd.read_pickle(trainPath).values
        logging.info('[DataProcessor] train shape: ' + str(train_set.shape))

        # split between train and test sets:
        x_train = train_set[:, 0]
        x_train = np.vstack(x_train[:])
        x_train = np.reshape(x_train, (-1, image_height, image_width, 3))

        it = ImageTransformer()

        x_train_grey = []
        sigma = 50
        mask = it.ApplyVignette(image_width, image_width, sigma)

        for i in range(len(x_train)):
            gray_image = cv2.cvtColor(x_train[i], cv2.COLOR_RGB2GRAY)
            gray_image = gray_image * mask[24:84, 0:108]
            gray_image = gray_image.astype(np.uint8)
            x_train_grey.append(gray_image)

        x_train_grey = np.array(x_train_grey)
        x_train_grey = x_train_grey.flatten().reshape(len(x_train), -1)
        y_train = train_set[:, 1]
        t = (x_train_grey, y_train)

        df = pd.DataFrame(t)
        df.to_pickle(file_name)

    @staticmethod
    def ProcessTrainDataGray(trainPath, image_height, image_width):
        train_set = pd.read_pickle(trainPath).values

        n_val = 13000
        np.random.seed()

        # split between train and test sets:
        x_train = train_set[0]
        x_train = np.vstack(x_train[:]).astype(np.float32)
        x_train = np.reshape(x_train, (-1, image_height, image_width,1))

        logging.info('[DataProcessor] train shape: ' + str(x_train.shape))

        x_train = np.swapaxes(x_train, 1, 3)
        x_train = np.swapaxes(x_train, 2, 3)

        y_train = train_set[1][0]
        y_train = np.vstack(y_train[:]).astype(np.float32)

        ix_val, ix_tr = np.split(np.random.permutation(x_train.shape[0]), [n_val])
        x_validation = x_train[ix_val, :]
        x_train = x_train[ix_tr, :]
        y_validation = y_train[ix_val, :]
        y_train = y_train[ix_tr, :]
        train_mean = np.mean(y_train, 0)
        train_std = np.std(y_train, 0)

        shape_ = x_train.shape[0]
        sel_idx = random.sample(range(0, shape_), k=50000)
        x_train = x_train[sel_idx, :]
        y_train = y_train[sel_idx, :]

        return [train_mean, train_std, x_train, x_validation, y_train, y_validation]

    @staticmethod
    def ProcessTestDataGray(testPath, image_height, image_width):
        test_set = pd.read_pickle(testPath).values

        x_test = test_set[0]
        x_test = np.vstack(x_test[:]).astype(np.float32)
        x_test = np.reshape(x_test, (-1, image_height, image_width, 1))

        logging.info('[DataProcessor] test shape: ' + str(x_test.shape))

        x_test = np.swapaxes(x_test, 1, 3)
        x_test = np.swapaxes(x_test, 2, 3)
        y_test = test_set[1][0]
        y_test = np.vstack(y_test[:]).astype(np.float32)

        return [x_test, y_test]