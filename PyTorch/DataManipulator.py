import pandas as pd
import numpy as np
import random
import logging
import cv2
import sys
sys.path.append("../DataProcessing/")
from ImageTransformer import ImageTransformer

class DataManipulator:


    @staticmethod
    def CropCenteredDataset(path,desiredSize, file_name):
        """Crop dataset in a centered way into a desired size

              Parameters
            ----------
            path : str
                The file location of the .pickle
            desiredSize : (int height, int width)
                Please...
            file_name: string
                name of the .pickle with the cropped images

           """
        dataset = pd.read_pickle(path)
        logging.info('[DataManipulator] dataset shape: ' + str(dataset.shape))

        x_set = dataset['x'].values
        x_set = np.vstack(x_set[:]).astype(np.float32)
        h, w, c = DataManipulator.GetSizeDataFromDataFrame(dataset)
        sizes = DataManipulator.CreateSizeDataFrame(desiredSize[0], desiredSize[1], c)

        x_set = np.reshape(x_set, (-1, h, w, c))


        x_cropped = []
        dw = int((w - desiredSize[1]) / 2)
        dh = int((h - desiredSize[0]) / 2)

        for i in range(len(x_set)):
            img = x_set[i]
            img = img[dh:(dh+desiredSize[0]), dw:(dw+desiredSize[1])]
            x_cropped.append(img)

        y_set = dataset['y'].values
        z_set = dataset['z'].values
        t_set = dataset['t'].values
        #r_set =  dataset['r'].values
        #o_train = train_set['o'].values

        data = pd.DataFrame(data={'x': x_cropped, 'y': y_set, 'z': z_set, 't': t_set})
        #data = pd.DataFrame(data={'x': x_cropped, 'y': y_set, 'z': z_set, 't': t_set, 'r': r_set})
        df = pd.concat([data, sizes], axis=1)
        df.to_pickle(file_name)

    @staticmethod
    def ShiftVideoDataset(path, file_name):
        """Shifts video frames by 1 to compensate for camera delay

              Parameters
            ----------
            path : str
                The file location of the .pickle
            desiredSize : (int height, int width)
                Please...
            file_name: string
                name of the .pickle with the cropped images

           """
        dataset = pd.read_pickle(path)
        logging.info('[DataManipulator] dataset shape: ' + str(dataset.shape))

        h, w, c = DataManipulator.GetSizeDataFromDataFrame(dataset)
        sizes = DataManipulator.CreateSizeDataFrame(h, w, c)

        x_set = dataset['x'].values
        y_set = dataset['y'].values
        z_set = dataset['z'].values
        t_set = dataset['t'].values
        x_set = x_set[1:]
        y_set = y_set[:-1]
        z_set = z_set[:-1]
        t_set = t_set[:-1]
        # o_train = train_set['o'].values

        data = pd.DataFrame(data={'x': x_set, 'y': y_set, 'z': z_set, 't': t_set})
        df = pd.concat([data, sizes], axis=1)
        df.to_pickle(file_name)


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
        logging.info('[DataManipulator] dataset shape: ' + str(train_set.shape))

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
            # gray_image = gray_image * mask[24:84, 0:108]
            gray_image = gray_image.astype(np.uint8)
            x_train_grey.append(gray_image)

        y_train = train_set['y'].values
        z_train = train_set['z'].values
        t_train = train_set['t'].values
        o_train = train_set['o'].values

        # df = pd.DataFrame(data={'x': x_train_grey, 'y': y_train})
        df = pd.DataFrame(data={'x': x_train_grey, 'y': y_train, 'z': z_train, 'o': o_train, 't': t_train})
        df.to_pickle(file_name)



    @staticmethod
    def MixAndMatch(path1, path2, train_name, test_name):
        """Mixeds 2 datasets to create train and test sets

              Parameters
            ----------
            path1 : str
                The file location of the first .pickle
            path1 : str
                The file location of the second .pickle
            train_name: string
                name of the mixed .pickle of the trian data

            test_name: string
                name of the mixed .pickle of the test data

           """
        dataset1 = pd.read_pickle(path1)
        logging.info('[DataManipulator] dataset shape 1: ' + str(dataset1.shape))

        dataset2 = pd.read_pickle(path2)
        logging.info('[DataManipulator] dataset shape2 : ' + str(dataset2.shape))

        np.random.seed()

        # size info

        h, w, c = DataManipulator.GetSizeDataFromDataFrame(dataset1)
        sizes = DataManipulator.CreateSizeDataFrame(h, w, c)

        # Split first dataset

        size1 = dataset1.shape[0]
        n_val1 = int(float(size1) * 0.25)


        x_set1 = dataset1['x'].values
        y_set1 = dataset1['y'].values
        z_set1 = dataset1['z'].values
        t_set1 = dataset1['t'].values


        #ind_test1, ind_train1 = np.split(np.random.permutation(size1), [n_val1])
        ind_test1, ind_train1 = np.split(np.arange(size1), [n_val1])
        x_test1 = x_set1[ind_test1]
        y_test1 = y_set1[ind_test1]
        z_test1 = z_set1[ind_test1]
        t_test1 = t_set1[ind_test1]

        x_train1 = x_set1[ind_train1]
        y_train1 = y_set1[ind_train1]
        z_train1 = z_set1[ind_train1]
        t_train1 = t_set1[ind_train1]

        # Split second dataset ?


        x_set2 = dataset2['x'].values
        y_set2 = dataset2['y'].values
        z_set2 = dataset2['z'].values
        t_set2 = dataset2['t'].values


        # merge train

        x_train1 = np.concatenate((x_train1, x_set2), axis=0)
        y_train1 = np.concatenate((y_train1, y_set2), axis=0)
        z_train1 = np.concatenate((z_train1, z_set2), axis=0)
        t_train1 = np.concatenate((t_train1, t_set2), axis=0)


        data1 = pd.DataFrame(data={'x': x_train1, 'y': y_train1, 'z': z_train1, 't': t_train1})
        df1 = pd.concat([data1, sizes], axis=1)
        df1.to_pickle(train_name)

        data2 = pd.DataFrame(data={'x': x_test1, 'y': y_test1, 'z': z_test1, 't': t_test1})
        df2 = pd.concat([data2, sizes], axis=1)
        df2.to_pickle(test_name)


    @staticmethod
    def Augment(path, pickle_name, factor=1):
        """Augment dataset

              Parameters
            ----------
            path : str
                The file location of the .pickle
            factor: int
                How many augmented variations will be created for each frame

           """
        dataset = pd.read_pickle(path)
        logging.info('[DataManipulator] dataset shape: ' + str(dataset.shape))

        h, w, c = DataManipulator.GetSizeDataFromDataFrame(dataset)
        sizes = DataManipulator.CreateSizeDataFrame(h, w, c)

        x_set = dataset['x'].values
        y_set = dataset['y'].values
        z_set = dataset['z'].values
        t_set = dataset['t'].values

        it = ImageTransformer()
        x_set = np.vstack(x_set[:])
        x_set = np.reshape(x_set, (-1, h, w, c))
        np.random.seed()

        x_augset = []
        y_augset = []
        z_augset = []
        t_augset = []

        for i in range(len(x_set)):

            y = y_set[i]
            z = z_set[i]
            t = t_set[i]
            x = x_set[i]
            x = np.reshape(x, (h, w)).astype("uint8")

            for p in range(factor):

                img = it.ApplyVignette(x, np.random.randint(25, 50))
                if np.random.choice([True, False]):
                    img = it.ApplyBlur(img, 3)
                # if np.random.choice([True, False]):
                #     X = self.it.ApplyNoise(X, 0, 1)
                if np.random.choice([True, False]):
                    img = it.ApplyExposure(img, np.random.uniform(0.7, 2.0))
                if np.random.choice([True, False]):
                     img = it.ApplyGamma(img, 0.4, 2.0)
                elif np.random.choice([True, False]):
                    img = it.ApplyDynamicRange(img, np.random.uniform(0.7, 0.9), np.random.uniform(0.0, 0.2))

                x_augset.append(img)
                y_augset.append(y)
                z_augset.append(z)
                t_augset.append(t)

        #     # imv = X.astype("uint8")
        #     # cv2.imshow("frame", imv)
        #     # cv2.waitKey()

        data = pd.DataFrame(data={'x': x_augset, 'y': y_augset, 'z': z_augset, 't': t_augset})
        df2 = pd.concat([data, sizes], axis=1)
        df2.to_pickle(pickle_name)




    @staticmethod
    def CropDataset(path, pickle_name, desiresSize, factor=1):

        """Augment dataset

                     Parameters
                   ----------
                   path : str
                       The file location of the .pickle
                   desiresSize: (int height, int width)
                        The desired size after cropping
                   factor: int
                       How many augmented variations will be created for each frame

                  """
        dataset = pd.read_pickle(path)
        logging.info('[DataManipulator] dataset shape: ' + str(dataset.shape))

        h, w, c = DataManipulator.GetSizeDataFromDataFrame(dataset)

        sizes = DataManipulator.CreateSizeDataFrame(desiresSize[0], desiresSize[1], c)

        x_set = dataset['x'].values
        y_set = dataset['y'].values
        z_set = dataset['z'].values
        t_set = dataset['t'].values

        it = ImageTransformer()
        x_set = np.vstack(x_set[:])
        x_set = np.reshape(x_set, (-1, h, w, c))
        np.random.seed()

        x_augset = []
        y_augset = []
        z_augset = []
        t_augset = []
        p_augset = []

        vertical_range = h - desiresSize[0]

        for i in range(len(x_set)):

            y = y_set[i]
            z = z_set[i]
            t = t_set[i]
            x = x_set[i]
            x = np.reshape(x, (h, w)).astype("uint8")

            for p in range(factor):
                #crop_offset = np.random.randint(0, vertical_range)
                crop_offset = p
                img = x[crop_offset:(crop_offset + desiresSize[0]), 0:w]

                p_augset.append(crop_offset)
                x_augset.append(img)
                y_augset.append(y)
                z_augset.append(z)
                t_augset.append(t)

        data = pd.DataFrame(data={'x': x_augset, 'y': y_augset, 'z': z_augset, 't': t_augset, 'p': p_augset})
        df2 = pd.concat([data, sizes], axis=1)
        df2.to_pickle(pickle_name)


    @staticmethod
    def GetSizeDataFromDataFrame(dataset):

        h = int(dataset['h'].values[0])
        w = int(dataset['w'].values[0])
        c = int(dataset['c'].values[0])

        return h, w, c

    @staticmethod
    def CreateSizeDataFrame(h, w, c):

        sizes_df = pd.DataFrame({'c': c, 'w': w, 'h': h}, index=[0])

        return sizes_df



    @staticmethod
    def AugmentAndCrop(path, pickle_name, desiresSize, factor=1):
        """Augment dataset

              Parameters
            ----------
            path : str
                The file location of the .pickle
            desiresSize: (int height, int width)
                The desired size after cropping
            factor: int
                How many augmented variations will be created for each frame

           """
        dataset = pd.read_pickle(path)
        logging.info('[DataManipulator] dataset shape: ' + str(dataset.shape))

        h, w, c = DataManipulator.GetSizeDataFromDataFrame(dataset)
        sizes = DataManipulator.CreateSizeDataFrame(desiresSize[0], desiresSize[1], c)

        x_set = dataset['x'].values
        y_set = dataset['y'].values
        z_set = dataset['z'].values
        t_set = dataset['t'].values

        it = ImageTransformer()
        x_set = np.vstack(x_set[:])
        x_set = np.reshape(x_set, (-1, h, w, c))
        np.random.seed()

        x_augset = []
        y_augset = []
        z_augset = []
        t_augset = []
        p_augset = []

        vertical_range = h - desiresSize[0]

        for i in range(len(x_set)):

            y = y_set[i]
            z = z_set[i]
            t = t_set[i]
            x = x_set[i]
            x = np.reshape(x, (h, w)).astype("uint8")

            for p in range(factor):

                crop_offset = np.random.randint(0, vertical_range)
                img = x[crop_offset:(crop_offset + desiresSize[0]), 0:w]
                img = it.ApplyVignette(img, np.random.randint(25, 50))

                if np.random.choice([True, False]):
                    img = it.ApplyBlur(img, 3)
                # if np.random.choice([True, False]):
                #     X = self.it.ApplyNoise(X, 0, 1)
                if np.random.choice([True, False]):
                    img = it.ApplyExposure(img, np.random.uniform(0.7, 2.0))
                if np.random.choice([True, False]):
                     img = it.ApplyGamma(img, 0.4, 2.0)
                elif np.random.choice([True, False]):
                    img = it.ApplyDynamicRange(img, np.random.uniform(0.7, 0.9), np.random.uniform(0.0, 0.2))

                p_augset.append(crop_offset)
                x_augset.append(img)
                y_augset.append(y)
                z_augset.append(z)
                t_augset.append(t)

        #     # imv = X.astype("uint8")
        #     # cv2.imshow("frame", imv)
        #     # cv2.waitKey()

        data = pd.DataFrame(data={'x': x_augset, 'y': y_augset, 'z': z_augset, 't': t_augset, 'p': p_augset})
        df = pd.concat([data, sizes], axis=1)
        df.to_pickle(pickle_name)


    @staticmethod
    def Rotate(path, pickle_name, factor=1):
        """Augment dataset

              Parameters
            ----------
            path : str
                The file location of the .pickle
            factor: int
                How many augmented variations will be created for each frame

           """
        dataset = pd.read_pickle(path)
        logging.info('[DataManipulator] dataset shape: ' + str(dataset.shape))

        h, w, c = DataManipulator.GetSizeDataFromDataFrame(dataset)
        sizes = DataManipulator.CreateSizeDataFrame(h, w, c)


        x_set = dataset['x'].values
        y_set = dataset['y'].values
        z_set = dataset['z'].values
        t_set = dataset['t'].values

        x_set = np.vstack(x_set[:])
        x_set = np.reshape(x_set, (-1, h, w, c))
        np.random.seed()

        x_augset = []
        y_augset = []
        z_augset = []
        t_augset = []
        r_augset = []

        max_angle = int(factor / 2)
        center = (w / 2, h / 2)
        scale = 1.0


        for i in range(len(x_set)):

            y = y_set[i]
            z = z_set[i]
            t = t_set[i]
            x = x_set[i]
            x = np.reshape(x, (h, w)).astype("uint8")

            for r in range(factor):

                #rot_angle = np.random.randint(-max_angle, max_angle)
                rot_angle = r - max_angle

                M = cv2.getRotationMatrix2D(center, rot_angle, scale)
                img = cv2.warpAffine(x, M, (h, w))

                x_augset.append(img)
                y_augset.append(y)
                z_augset.append(z)
                t_augset.append(t)
                r_augset.append(rot_angle)


        data = pd.DataFrame(data={'x': x_augset, 'y': y_augset, 'z': z_augset, 't': t_augset, 'r': r_augset})
        df2 = pd.concat([data, sizes], axis=1)
        df2.to_pickle(pickle_name)


    @staticmethod
    def DivideDataset(pickle_path, new_path1, new_path2, num_split):

        """Divide dataset

                      Parameters
                    ----------
                    path : str
                        The file location of the .pickle
                    new_path1 : str
                        The file location of the first new .pickle
                    new_path2 : str
                        The file location of the second new .pickle
                    num_split: int
                        At which frame to split the dataset

                   """
        dataset = pd.read_pickle(pickle_path)
        logging.info('[DataManipulator] dataset shape: ' + str(dataset.shape))

        h, w, c = DataManipulator.GetSizeDataFromDataFrame(dataset)
        sizes = DataManipulator.CreateSizeDataFrame(h, w, c)

        x_set = dataset['x'].values
        y_set = dataset['y'].values
        z_set = dataset['z'].values
        t_set = dataset['t'].values

        data1 = pd.DataFrame(data={'x': x_set[:num_split], 'y': y_set[:num_split],
                                   'z': z_set[:num_split], 't': t_set[:num_split]})
        data2 = pd.DataFrame(data={'x': x_set[num_split:-1], 'y': y_set[num_split:-1],
                                   'z': z_set[num_split:-1], 't': t_set[num_split:-1]})
        df1 = pd.concat([data1, sizes], axis=1)
        df1.to_pickle(new_path1)
        df2 = pd.concat([data2, sizes], axis=1)
        df2.to_pickle(new_path2)

    @staticmethod
    def JoinDataset(path1, path2, new_path):
        """Divide dataset

              Parameters
            ----------
            path1 : str
                The file location of the first .pickle
            path2 : str
                The file location of the second .pickle
            new_path : str
                The file location of the new .pickle

           """

        dataset1 = pd.read_pickle(path1)
        logging.info('[DataManipulator] dataset1 shape: ' + str(dataset1.shape))

        dataset2 = pd.read_pickle(path2)
        logging.info('[DataManipulator] dataset2 shape: ' + str(dataset2.shape))

        h, w, c = DataManipulator.GetSizeDataFromDataFrame(dataset1)
        sizes = DataManipulator.CreateSizeDataFrame(h, w, c)

        x_set = dataset1['x'].values
        y_set = dataset1['y'].values
        z_set = dataset1['z'].values
        t_set = dataset1['t'].values
        p_set = dataset1['p'].values

        x_set = np.concatenate((x_set, dataset2['x'].values), axis=0)
        y_set = np.concatenate((y_set, dataset2['y'].values), axis=0)
        z_set = np.concatenate((z_set, dataset2['z'].values), axis=0)
        t_set = np.concatenate((t_set, dataset2['t'].values), axis=0)
        p_set = np.concatenate((p_set, dataset2['p'].values), axis=0)

        data = pd.DataFrame(data={'x': x_set, 'y': y_set, 'z': z_set, 't': t_set, 'p': p_set})
        df2 = pd.concat([data, sizes], axis=1)
        df2.to_pickle(new_path)