import pandas as pd
import numpy as np
import random
import logging
import cv2
import sys
sys.path.append("../DataProcessing/")
from ImageTransformer import ImageTransformer

class DataProcessor:

    @staticmethod
    def ProcessTrainData(trainPath, image_height, image_width, isGray = False, isExtended=False):
        """Reads the .pickle file and converts it into a format suitable fot training

            Parameters
            ----------
            trainPath : str
                The file location of the .pickle
            image_height : int
                Please...
            image_width : int
                Please...
            isGray : bool, optional
                True is the dataset is of 1-channel (gray) images, False if RGB
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
        #n_val = 13000

        np.random.seed()
        # split between train and test sets:
        x_train = train_set['x'].values
        x_train = np.vstack(x_train[:]).astype(np.float32)
        if isGray == True:
            x_train = np.reshape(x_train, (-1, image_height, image_width, 1))
        else:
            x_train = np.reshape(x_train, (-1, image_height, image_width, 3))

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
        #sel_idx = random.sample(range(0, shape_), k=50000)
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
    def ProcessTestData(testPath, image_height, image_width, isGray = False, isExtended=False):
        """Reads the .pickle file and converts it into a format suitable fot testing

            Parameters
            ----------
            testPath : str
                The file location of the .pickle
            image_height : int
                Please...
            image_width : int
                Please...
            isGray : bool, optional
                True is the dataset is of 1-channel (gray) images, False if RGB
            isExtended : bool, optional
                True if the dataset contains both head and hand pose and you wish to retrieve both


            Returns
            -------
            list
                list of video frames and list of labels (poses)
            """

        test_set = pd.read_pickle(testPath)
        logging.info('[DataProcessor] test shape: ' + str(test_set.shape))

        x_test = test_set['x'].values
        x_test = np.vstack(x_test[:]).astype(np.float32)
        if isGray == True:
            x_test = np.reshape(x_test, (-1, image_height, image_width, 1))
        else:
            x_test = np.reshape(x_test, (-1, image_height, image_width, 3))

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
    def ExtractValidationLabels(testPath, image_height, image_width, isGray = False):
        """Reads the .pickle file and converts it into a format suitable for testing on pulp
            You need to create a folder called test though

            Parameters
            ----------
            testPath : str
                The file location of the .pickle
            image_height : int
                Please...
            image_width : int
                Please...
            isGray : bool, optional
                True is the dataset is of 1-channel (gray) images, False if RGB

           """

        test_set = pd.read_pickle(testPath)
        logging.info('[DataProcessor] test shape: ' + str(test_set.shape))

        x_test = test_set['x'].values
        x_test = np.vstack(x_test[:]).astype(np.float32)
        if isGray == True:
            x_test = np.reshape(x_test, (-1, image_height, image_width, 1))
        else:
            x_test = np.reshape(x_test, (-1, image_height, image_width, 3))

        x_test = np.swapaxes(x_test, 1, 3)
        x_test = np.swapaxes(x_test, 2, 3)
        y_test = test_set['y'].values
        y_test = np.vstack(y_test[:]).astype(np.float32)

        f = open("test/labels.txt", "w")

        for i in range(0, len(x_test)):
            data = x_test[i]
            data = np.swapaxes(data, 0, 2)
            data = np.swapaxes(data, 0, 1)
            img = np.reshape(data, (60, 108))
            #img = np.zeros((244, 324), np.uint8)
            #img[92:152, 108:216] = data
            cv2.imwrite("test/{}.pgm".format(i), img)
            label = y_test[i]
            #f.write("{},{},{},{}\n".format(label[0], label[1],label[2],label[3]))
        f.close()


    @staticmethod
    def CropCenteredDataset(path, imgSize, desiredSize, file_name, isGray = False):
        """Crop dataset in a centered way into a desired size

              Parameters
            ----------
            path : str
                The file location of the .pickle
            imgSize : (int height, int width)
                Please...
            desiredSize : (int height, int width)
                Please...
            isGray : bool, optional
                True is the dataset is of 1-channel (gray) images, False if RGB
            file_name: string
                name of the .pickle with the cropped images

           """
        dataset = pd.read_pickle(path)
        logging.info('[DataProcessor] train shape: ' + str(dataset.shape))

        x_set = dataset['x'].values
        x_set = np.vstack(x_set[:]).astype(np.float32)
        if isGray == True:
            x_set = np.reshape(x_set, (-1, imgSize[0], imgSize[1], 1))
        else:
            x_set = np.reshape(x_set, (-1, imgSize[0], imgSize[1], 3))

        x_cropped = []

        for i in range(len(x_set)):
            img = x_set[i]
            w = int((imgSize[1] - desiredSize[1]) / 2)
            h = int((imgSize[0] - desiredSize[0]) / 2)
            img = img[h:(h+desiredSize[0]), w:(w+desiredSize[1])]
            x_cropped.append(img)

        y_set= dataset['y'].values
        z_set= dataset['z'].values
        t_set = dataset['t'].values
        #o_train = train_set['o'].values

        # df = pd.DataFrame(data={'x': x_train_grey, 'y': y_train})
        #df = pd.DataFrame(data={'x': x_train_grey, 'y': y_train, 'z': z_train, 'o': o_train, 't': t_train})
        df = pd.DataFrame(data={'x': x_cropped, 'y': y_set, 'z': z_set, 't': t_set})
        df.to_pickle(file_name)

    @staticmethod
    def ProcessInferenceData(images, image_height, image_width, isGray=False):
        """Converts a list of images into a format suitable fot inference

            Parameters
            ----------
            images : list
                list of images
            image_height : int
                Please...
            image_width : int
                Please...
            isGray : bool, optional
                True is the dataset is of 1-channel (gray) images, False if RGB

            Returns
            -------
            list
                list of video frames and list of labels (poses, which are garbage)
            """

        x_test = np.stack(images, axis=0).astype(np.float32)
        if isGray == True:
            x_test = np.reshape(x_test, (-1, image_height, image_width, 1))
        else:
            x_test = np.reshape(x_test, (-1, image_height, image_width, 3))
        x_test = np.swapaxes(x_test, 1, 3)
        x_test = np.swapaxes(x_test, 2, 3)
        y_test = [0, 0, 0, 0] * len(x_test)
        y_test = np.vstack(y_test[:]).astype(np.float32)
        y_test = np.reshape(y_test, (-1, 4))


        return [x_test, y_test]

    @staticmethod
    def CreateGreyPickle(trainPath, image_height, image_width, file_name):
        """Converts Dario's RGB dataset to a gray + vignette dataset

            Parameters
            ----------
            images : list
                list of images
            image_height : int
                Please...
            image_width : int
                Please...
            file_name : str
                name of the new .pickle

            """
        train_set = pd.read_pickle(trainPath)
        logging.info('[DataProcessor] train shape: ' + str(train_set.shape))

        # split between train and test sets:
        x_train = train_set['x'].values
        x_train = np.vstack(x_train[:])
        x_train = np.reshape(x_train, (-1, image_height, image_width, 3))

        it = ImageTransformer()

        x_train_grey = []
        sigma = 50
        mask = it.GetVignette(image_width, image_width, sigma)

        for i in range(len(x_train)):
            gray_image = cv2.cvtColor(x_train[i], cv2.COLOR_RGB2GRAY)
            #gray_image = gray_image * mask[24:84, 0:108]
            gray_image = gray_image.astype(np.uint8)
            x_train_grey.append(gray_image)

        y_train = train_set['y'].values
        z_train = train_set['z'].values
        t_train = train_set['t'].values
        o_train = train_set['o'].values

        #df = pd.DataFrame(data={'x': x_train_grey, 'y': y_train})
        df = pd.DataFrame(data={'x': x_train_grey, 'y': y_train, 'z': z_train, 'o': o_train, 't': t_train})
        df.to_pickle(file_name)







