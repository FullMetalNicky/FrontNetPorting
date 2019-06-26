import pandas as pd
import numpy as np
import random

class DataLoader:
    def __init__(self, trainPath, testPath):
        self.train_set = pd.read_pickle(trainPath).values
        print('train shape: ' + str(self.train_set.shape))
        self.test_set = pd.read_pickle(testPath).values
        print('test shape: ' + str(self.test_set.shape))

    def ProcessData(self):
        n_val = 13000
        np.random.seed(100)

        image_height = 60
        image_width = 108

        # split between train and test sets:
        x_train = 255 - self.train_set[:, 0]  # otherwise is inverted
        x_train = np.vstack(x_train[:]).astype(np.float32)
       # x_train = np.reshape(x_train, (-1, image_height, image_width, 3))
       #(63726, 60, 108, 3)
        x_train = np.reshape(x_train, (-1, 3, image_height, image_width))
        #(63726, 3, 60, 108)
        y_train = self.train_set[:, 1]
        y_train = np.vstack(y_train[:]).astype(np.float32)

        ix_val, ix_tr = np.split(np.random.permutation(x_train.shape[0]), [n_val])
        x_validation = x_train[ix_val, :]
        x_train = x_train[ix_tr, :]
        y_validation = y_train[ix_val, :]
        y_train = y_train[ix_tr, :]

        x_test = 255 - self.test_set[:, 0]
        x_test = np.vstack(x_test[:]).astype(np.float32)
        #x_test = np.reshape(x_test, (-1, image_height, image_width, 3))
        x_test = np.reshape(x_test, (-1, 3, image_height, image_width))
        y_test = self.test_set[:, 1]
        y_test = np.vstack(y_test[:]).astype(np.float32)
        visual_odom = self.test_set[:, 2]
        visual_odom = np.vstack(visual_odom[:]).astype(np.float32)

        model_name = "Model_v1"
        shape_ = x_train.shape[0]
        sel_idx = random.sample(range(0, shape_), k=50000)
        x_train = x_train[sel_idx, :]
        y_train = y_train[sel_idx, :]
        return [sel_idx, x_train, x_validation, x_test, y_train, y_validation, y_test]