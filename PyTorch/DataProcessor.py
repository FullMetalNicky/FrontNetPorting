import pandas as pd
import numpy as np
import random
import logging
import cv2

class DataProcessor:

    @staticmethod
    def GetSizeDataFromDataFrame(dataset):

        h = int(dataset['h'].values[0])
        w = int(dataset['w'].values[0])
        c = int(dataset['c'].values[0])

        return h, w, c

    @staticmethod
    def ProcessTrainData(trainPath, isExtended=False):
        """Reads the .pickle file and converts it into a format suitable fot training

            Parameters
            ----------
            trainPath : str
                The file location of the .pickl
            isExtended : bool, optional
                True if the dataset contains both head and hand pose and you wish to retrieve both


            Returns
            -------
            list
                list of video frames and list of labels (poses)
            """
        train_set = pd.read_pickle(trainPath)

        logging.info('[DataProcessor] train shape: ' + str(train_set.shape))
        size = train_set.shape[0]
        n_val = int(float(size) * 0.2)

        h, w, c = DataProcessor.GetSizeDataFromDataFrame(train_set)

        np.random.seed(1749)
        random.seed(1749)
        # split between train and test sets:
        x_train = train_set['x'].values
        x_train = np.vstack(x_train[:]).astype(np.float32)
        x_train = np.reshape(x_train, (-1, h, w, c))

        x_train= np.swapaxes(x_train, 1, 3)
        x_train = np.swapaxes(x_train, 2, 3)

        y_train = train_set['y'].values
        y_train = np.vstack(y_train[:]).astype(np.float32)

        ix_val, ix_tr = np.split(np.random.permutation(train_set.shape[0]), [n_val])
        x_validation = x_train[ix_val, :]
        x_train = x_train[ix_tr, :]
        y_validation = y_train[ix_val, :]
        y_train = y_train[ix_tr, :]

        shape_ = len(x_train)

        sel_idx = random.sample(range(0, shape_), k=(size-n_val))
        x_train = x_train[sel_idx, :]
        y_train = y_train[sel_idx, :]



        if isExtended == True:
            z_train = train_set['z'].values
            z_train = np.vstack(z_train[:]).astype(np.float32)
            z_validation = z_train[ix_val, :]
            z_train = z_train[ix_tr, :]
            z_train = z_train[sel_idx, :]
            return [x_train, x_validation, y_train, y_validation, z_train, z_validation]

        return [x_train, x_validation, y_train, y_validation]

    @staticmethod
    def ProcessTestData(testPath, isExtended=False):
        """Reads the .pickle file and converts it into a format suitable fot testing

            Parameters
            ----------
            testPath : str
                The file location of the .pickle
            isExtended : bool, optional
                True if the dataset contains both head and hand pose and you wish to retrieve both


            Returns
            -------
            list
                list of video frames and list of labels (poses)
            """

        test_set = pd.read_pickle(testPath)
        logging.info('[DataProcessor] test shape: ' + str(test_set.shape))
        h, w, c = DataProcessor.GetSizeDataFromDataFrame(test_set)

        x_test = test_set['x'].values
        x_test = np.vstack(x_test[:]).astype(np.float32)
        x_test = np.reshape(x_test, (-1, h, w, c))


        x_test = np.swapaxes(x_test, 1, 3)
        x_test = np.swapaxes(x_test, 2, 3)
        y_test = test_set['y'].values
        y_test = np.vstack(y_test[:]).astype(np.float32)

        if isExtended ==True:
            z_test = test_set['z'].values
            z_test = np.vstack(z_test[:]).astype(np.float32)
            return [x_test, y_test, z_test]


        return [x_test, y_test]

    @staticmethod
    def ProcessTestDataAsRGB(testPath, isExtended=False):
        """Reads the .pickle file and converts it into a format suitable fot testing

            Parameters
            ----------
            testPath : str
                The file location of the .pickle
            isExtended : bool, optional
                True if the dataset contains both head and hand pose and you wish to retrieve both


            Returns
            -------
            list
                list of video frames and list of labels (poses)
            """

        test_set = pd.read_pickle(testPath)
        logging.info('[DataProcessor] test shape: ' + str(test_set.shape))
        h, w, c = DataProcessor.GetSizeDataFromDataFrame(test_set)

        x_test = test_set['x'].values
        x_test = np.vstack(x_test[:]).astype(np.float32)
        x_test = np.reshape(x_test, (-1, h, w, c))
        x_testRGB = []

        for i in range(len(x_test)):
            img = cv2.cvtColor(x_test[i],cv2.COLOR_GRAY2RGB)
            x_testRGB.append(img)

        x_test = np.swapaxes(x_testRGB, 1, 3)
        x_test = np.swapaxes(x_test, 2, 3)
        y_test = test_set['y'].values
        y_test = np.vstack(y_test[:]).astype(np.float32)

        if isExtended == True:
            z_test = test_set['z'].values
            z_test = np.vstack(z_test[:]).astype(np.float32)
            return [x_test, y_test, z_test]

        return [x_test, y_test]

    @staticmethod
    def GetTimeStampsFromTestData(testPath):

        """Reads the .pickle file and extrects the frames' timestamps

                   Parameters
                   ----------
                   testPath : str
                       The file location of the .pickle

                   Returns
                   -------
                   list
                       list of timestamps
                   """

        t_test = None
        test_set = pd.read_pickle(testPath)
        logging.info('[DataProcessor] test shape: ' + str(test_set.shape))
        if 't' in test_set.columns:
            t_test = test_set['t'].values


        return t_test

    @staticmethod
    def GetOutputsFromTestData(testPath):

        """Reads the .pickle file and extrects the recorded NNs outputs

                   Parameters
                   ----------
                   testPath : str
                       The file location of the .pickle

                   Returns
                   -------
                   list
                       list of outputs/predictions
                   """

        o_test = None
        test_set = pd.read_pickle(testPath)
        logging.info('[DataProcessor] test shape: ' + str(test_set.shape))
        if 'o' in test_set.columns:
            o_test = test_set['o'].values
            o_test = np.vstack(o_test[:]).astype(np.float32)

        return o_test

    @staticmethod
    def GetPitchFromTestData(testPath):

        """Reads the .pickle file and extracts the pitch values

                   Parameters
                   ----------
                   testPath : str
                       The file location of the .pickle

                   Returns
                   -------
                   list
                       list of pitch  values
                   """

        p_test = None
        test_set = pd.read_pickle(testPath)
        logging.info('[DataProcessor] test shape: ' + str(test_set.shape))
        if 'p' in test_set.columns:
            p_test = test_set['p'].values

        return p_test

    @staticmethod
    def GetRollFromTestData(testPath):

        """Reads the .pickle file and extracts the roll values

                   Parameters
                   ----------
                   testPath : str
                       The file location of the .pickle

                   Returns
                   -------
                   list
                       list of roll  values
                   """

        r_test = None
        test_set = pd.read_pickle(testPath)
        logging.info('[DataProcessor] test shape: ' + str(test_set.shape))
        if 'r' in test_set.columns:
            r_test = test_set['r'].values

        return r_test

    @staticmethod
    def GetDatasetStatisics(testPath):
        """Reads the .pickle file and provides statistic data about the pose values

                    Parameters
                    ----------
                    testPath : str
                        The file location of the .pickle

                   """

        test_set = pd.read_pickle(testPath)
        logging.info('[DataProcessor] test shape: ' + str(test_set.shape))

        y_test = test_set['y'].values
        y_test = np.vstack(y_test[:]).astype(np.float32)
        mean = np.mean(y_test, 0)
        std = np.std(y_test, 0)
        min = np.min(y_test, 0)
        max = np.max(y_test, 0)

        print("mean: {}".format(mean))
        print("std: {}".format(std))
        print("min: {}".format(min))
        print("max: {}".format(max))




