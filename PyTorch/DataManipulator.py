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
            img = img.astype("uint8")
            x_cropped.append(img)

        y_set = dataset['y'].values
        z_set = dataset['z'].values
        t_set = dataset['t'].values
        #p_set = dataset['p'].values
       # r_set =  dataset['r'].values
        #o_train = train_set['o'].values

        data = pd.DataFrame(data={'x': x_cropped, 'y': y_set, 'z': z_set, 't': t_set})
       # data = pd.DataFrame(data={'x': x_cropped, 'y': y_set, 'z': z_set, 't': t_set, 'r': r_set})
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
        p_set = dataset['p'].values
        x_set = x_set[1:]
        y_set = y_set[:-1]
        z_set = z_set[:-1]
        t_set = t_set[:-1]
        p_set = p_set[:-1]
        # o_train = train_set['o'].values

        data = pd.DataFrame(data={'x': x_set, 'y': y_set, 'z': z_set, 't': t_set, 'p': p_set})
        df = pd.concat([data, sizes], axis=1)
        df.to_pickle(file_name)


    @staticmethod
    def CreateGreyPickle(trainPath, image_height, image_width, pickle_name):
        """Converts Dario's RGB dataset to a gray + vignette dataset

            Parameters
            ----------
            images : list
                list of images
            image_height : int
                Please...
            image_width : int
                Please...
            pickle_name : str
                name of the new .pickle

            """
        train_set = pd.read_pickle(trainPath)
        logging.info('[DataManipulator] dataset shape: ' + str(train_set.shape))

        # split between train and test sets:
        x_train = train_set['x'].values
        x_train = np.vstack(x_train[:])
        x_train = np.reshape(x_train, (-1, image_height, image_width, 3))

        sizes = DataManipulator.CreateSizeDataFrame(image_height, image_width, 1)

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
        # z_train = train_set['z'].values
        # t_train = train_set['t'].values
        # o_train = train_set['o'].values

        data = pd.DataFrame(data={'x': x_train_grey, 'y': y_train})
        #df = pd.DataFrame(data={'x': x_train_grey, 'y': y_train, 'z': z_train, 'o': o_train, 't': t_train})
        df = pd.concat([data, sizes], axis=1)
        df.to_pickle(pickle_name)




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
    def ConvertToInt(path, pickle_name):

        dataset = pd.read_pickle(path)
        logging.info('[DataManipulator] dataset shape: ' + str(dataset.shape))

        h, w, c = DataManipulator.GetSizeDataFromDataFrame(dataset)
        sizes = DataManipulator.CreateSizeDataFrame(h, w, c)

        x_set = dataset['x'].values
        y_set = dataset['y'].values
        z_set = dataset['z'].values
        t_set = dataset['t'].values
        p_set = dataset['p'].values


        x_set = np.vstack(x_set[:])
        x_set = np.reshape(x_set, (-1, h, w, c))


        x_augset = []
        y_augset = []
        z_augset = []
        t_augset = []
        p_augset = []

        for i in range(len(x_set)):

            y = y_set[i]
            z = z_set[i]
            t = t_set[i]
            x = x_set[i]
            p = p_set[i]
            x = np.reshape(x, (h, w)).astype("uint8")

            p_augset.append(p)
            x_augset.append(x)
            y_augset.append(y)
            z_augset.append(z)
            t_augset.append(t)

        data = pd.DataFrame(data={'x': x_augset, 'y': y_augset, 'z': z_augset, 't': t_augset, 'p': p_augset})
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
        hor_offset = int((w - desiresSize[1])/2)

        for i in range(len(x_set)):

            y = y_set[i]
            z = z_set[i]
            t = t_set[i]
            x = x_set[i]
            x = np.reshape(x, (h, w)).astype("uint8")

            for p in range(factor):

                crop_offset = np.random.randint(0, vertical_range)
                img = it.ApplyVignette(x, np.random.randint(25, 50))
                img = img[crop_offset:(crop_offset + desiresSize[0]), hor_offset:desiresSize[1]]

                # crop_offset = np.random.randint(20, 80)
                # img = it.ApplyVignette(x, np.random.randint(25, 50))
                # img = img[crop_offset:(crop_offset + desiresSize[0]), hor_offset:(hor_offset+desiresSize[1])]


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
    def AugmentAndCropMirko(path, pickle_name, desiresSize, factor=1):
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
        aug_id = []
        frame_id = []

        vertical_range = h - desiresSize[0]
        hor_offset = int((w - desiresSize[1]) / 2)

        for p in range(factor):

            Blur = np.random.choice([True, False])
            Exposure = np.random.choice([True, False])
            Exposure_param = np.random.uniform(0.7, 2.0)
            Gamma = np.random.choice([True, False])
            DR = np.random.choice([True, False])
            DR_high = np.random.uniform(0.7, 0.9)
            DR_low = np.random.uniform(0.0, 0.2)
            vignette = np.random.randint(25, 50)
            crop_offset = np.random.randint(0, vertical_range)

            for i in range(len(x_set)):
                y = y_set[i]
                z = z_set[i]
                t = t_set[i]
                x = x_set[i]
                x = np.reshape(x, (h, w)).astype("uint8")

                img = it.ApplyVignette(x, vignette)
                img = img[crop_offset:(crop_offset + desiresSize[0]), hor_offset:(hor_offset + desiresSize[1])]

                if Blur:
                    img = it.ApplyBlur(img, 3)
                if Exposure:
                    img = it.ApplyExposure(img, Exposure_param)
                if Gamma:
                    img = it.ApplyGamma(img, 0.4, 2.0)
                elif DR:
                    img = it.ApplyDynamicRange(img, DR_high, DR_low)

                img = img.astype("uint8")
                p_augset.append(crop_offset)
                x_augset.append(img)
                y_augset.append(y)
                z_augset.append(z)
                t_augset.append(t)
                aug_id.append(p)
                frame_id.append(i)

    #     # imv = X.astype("uint8")
    #     # cv2.imshow("frame", imv)
    #     # cv2.waitKey()

        data = pd.DataFrame(data={'frame_id': frame_id, 'aug_id': aug_id , 'frame': x_augset, 'timestamp': t_augset, 'pitch': p_augset, 'rel_pose' :y_augset, 'drone_pose': z_augset})
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
        p_set = dataset['p'].values

        if num_split <1:
            num_split = int(len(x_set) * num_split)



        data1 = pd.DataFrame(data={'x': x_set[:num_split], 'y': y_set[:num_split],
                                   'z': z_set[:num_split], 't': t_set[:num_split], 'p': p_set[:num_split]})
        data2 = pd.DataFrame(data={'x': x_set[num_split:-1], 'y': y_set[num_split:-1],
                                   'z': z_set[num_split:-1], 't': t_set[num_split:-1], 'p': p_set[num_split:-1]})
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

    @staticmethod
    def JoinDatasetFromList(pathlist, new_path):
        """Divide dataset

              Parameters
            ----------
            path1 : str list
                list of paths to pickles to be joined
            new_path : str
                The file location of the new .pickle

           """

        dataset1 = pd.read_pickle(pathlist[0])
        logging.info('[DataManipulator] dataset1 shape: ' + str(dataset1.shape))


        h, w, c = DataManipulator.GetSizeDataFromDataFrame(dataset1)
        sizes = DataManipulator.CreateSizeDataFrame(h, w, c)

        x_set = dataset1['x'].values
        y_set = dataset1['y'].values
        z_set = dataset1['z'].values
        t_set = dataset1['t'].values
        #p_set = dataset1['p'].values


        for i in range(1,len(pathlist)):
            print(i)
            dataset = pd.read_pickle(pathlist[i])
            logging.info("[DataManipulator] dataset {} shape: {}".format(i,  dataset.shape))

            x_set = np.concatenate((x_set, dataset['x'].values), axis=0)
            y_set = np.concatenate((y_set, dataset['y'].values), axis=0)
            z_set = np.concatenate((z_set, dataset['z'].values), axis=0)
            t_set = np.concatenate((t_set, dataset['t'].values), axis=0)
            #p_set = np.concatenate((p_set, dataset['p'].values), axis=0)

        #data = pd.DataFrame(data={'x': x_set, 'y': y_set, 'z': z_set, 't': t_set, 'p': p_set})
        data = pd.DataFrame(data={'x': x_set, 'y': y_set, 'z': z_set, 't': t_set})
        df2 = pd.concat([data, sizes], axis=1)
        df2.to_pickle(new_path)

    @staticmethod
    def DownsampleDataset(pickle_path, ds_size, ds_type, new_path):
        """Downsample Dataset

              Parameters
            ----------
            pickle_path : str
                The file location of the first .pickle
            ds_size: (int height, int width)
                target size after downsampling
            ds_type: cv2 interpolation flag
                the method of downsampling
            new_path : str
                The name/path of the newly created dataset

           """

        dataset = pd.read_pickle(pickle_path)
        logging.info('[DataManipulator] dataset shape: ' + str(dataset.shape))

        h, w, c = DataManipulator.GetSizeDataFromDataFrame(dataset)
        sizes = DataManipulator.CreateSizeDataFrame(ds_size[0], ds_size[1], c)

        x_set = dataset['x'].values
        y_set = dataset['y'].values
        z_set = dataset['z'].values
        t_set = dataset['t'].values
        #p_set = dataset['p'].values

        x_set = np.vstack(x_set[:])
        x_set = np.reshape(x_set, (-1, h, w, c))

        x_ds= []

        for i in range(len(x_set)):
            img = x_set[i]
            img = cv2.resize(img, (ds_size[1], ds_size[0]), interpolation=ds_type)
            img = img.astype(np.uint8)
            x_ds.append(img)


        #data = pd.DataFrame(data={'x': x_ds, 'y': y_set, 'z': z_set, 't': t_set, 'p': p_set})
        data = pd.DataFrame(data={'x': x_ds, 'y': y_set, 'z': z_set, 't': t_set})
        df = pd.concat([data, sizes], axis=1)
        df.to_pickle(new_path)

    @staticmethod
    def FoveateDataset(pickle_path, new_path):
        """Foveate Dataset

            This function is hardocded  because the computations here are headache that I don't need to deal with right now.
            ok?ok.

              Parameters
            ----------
            pickle_path : str
                The file location of the .pickle of size 160x90
            new_path : str
                The name/path of the newly created dataset which is 108x60

           """

        dataset = pd.read_pickle(pickle_path)
        logging.info('[DataManipulator] dataset shape: ' + str(dataset.shape))

        h, w, c = DataManipulator.GetSizeDataFromDataFrame(dataset)
        sizes = DataManipulator.CreateSizeDataFrame(60, 108, c)

        x_set = dataset['x'].values
        y_set = dataset['y'].values
        z_set = dataset['z'].values
        t_set = dataset['t'].values
        p_set = dataset['p'].values

        x_set = np.vstack(x_set[:])
        x_set = np.reshape(x_set, (-1, h, w, c))

        x_fov = []

        for i in range(len(x_set)):
            img = x_set[i]
            img = np.reshape(img, (h, w)).astype("uint8")
            fov_img = np.zeros((60, 108), dtype="uint8")

            #in  corners res is half on both axes

            fov_img[0:15, 0:27] = cv2.resize(img[0:30, 0:53], (27, 15), cv2.INTER_LINEAR)
            fov_img[-15:, 0:27] = cv2.resize(img[-30:, 0:53], (27, 15), cv2.INTER_LINEAR)
            fov_img[0:15, -27:] = cv2.resize(img[0:30, -53:], (27, 15), cv2.INTER_LINEAR)
            fov_img[-15:, -27:] = cv2.resize(img[-30:, -53:], (27, 15), cv2.INTER_LINEAR)

            #top/bottom center - every second row
            fov_img[0:15, 27:81] = cv2.resize(img[0:30, 53:107], (54, 15), cv2.INTER_LINEAR)
            fov_img[-15:, 27:81] = cv2.resize(img[-30:, 53:107], (54, 15), cv2.INTER_LINEAR)

            # left/right center - everey second column
            fov_img[15:45, 0:27] = cv2.resize(img[30:60, 0:53], (27, 30), cv2.INTER_LINEAR)
            fov_img[15:45, -27:] = cv2.resize(img[30:60, -53:], (27, 30), cv2.INTER_LINEAR)

            # center is full res
            fov_img[15:45, 27:81] = img[30:60, 53:107]

            x_fov.append(fov_img)

        data = pd.DataFrame(data={'x': x_fov, 'y': y_set, 'z': z_set, 't': t_set, 'p': p_set})
        #data = pd.DataFrame(data={'x': x_fov, 'y': y_set, 'z': z_set, 't': t_set})
        df = pd.concat([data, sizes], axis=1)
        df.to_pickle(new_path)

    @staticmethod
    def FoveateDataset80x48(pickle_path, new_path):
        """Foveate Dataset

            This function is hardocded  because the computations here are headache that I don't need to deal with right now.
            ok?ok.

              Parameters
            ----------
            pickle_path : str
                The file location of the .pickle of size 160x90
            new_path : str
                The name/path of the newly created dataset which is 80x48

           """

        dataset = pd.read_pickle(pickle_path)
        logging.info('[DataManipulator] dataset shape: ' + str(dataset.shape))

        h, w, c = DataManipulator.GetSizeDataFromDataFrame(dataset)
        sizes = DataManipulator.CreateSizeDataFrame(48, 80, c)

        x_set = dataset['x'].values
        y_set = dataset['y'].values
        z_set = dataset['z'].values
        t_set = dataset['t'].values
        p_set = dataset['p'].values

        x_set = np.vstack(x_set[:])
        x_set = np.reshape(x_set, (-1, h, w, c))

        x_fov = []

        for i in range(len(x_set)):
            img = x_set[i]
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

            x_fov.append(fov_img)

        data = pd.DataFrame(data={'x': x_fov, 'y': y_set, 'z': z_set, 't': t_set, 'p': p_set})
        # data = pd.DataFrame(data={'x': x_fov, 'y': y_set, 'z': z_set, 't': t_set})
        df = pd.concat([data, sizes], axis=1)
        df.to_pickle(new_path)

    @staticmethod
    def BlurBySamplingDataset(pickle_path, ds_size, new_path):
        """Downsample Dataset

              Parameters
            ----------
            pickle_path : str
                The file location of the first .pickle
            ds_size: (int height, int width)
                target size after downsampling
            ds_type: cv2 interpolation flag
                the method of downsampling
            new_path : str
                The name/path of the newly created dataset

           """

        dataset = pd.read_pickle(pickle_path)
        logging.info('[DataManipulator] dataset shape: ' + str(dataset.shape))

        h, w, c = DataManipulator.GetSizeDataFromDataFrame(dataset)
        sizes = DataManipulator.CreateSizeDataFrame(h, w, c)

        x_set = dataset['x'].values
        y_set = dataset['y'].values
        z_set = dataset['z'].values
        t_set = dataset['t'].values
       # p_set = dataset['p'].values

        x_set = np.vstack(x_set[:])
        x_set = np.reshape(x_set, (-1, h, w, c))

        x_ds = []

        for i in range(len(x_set)):
            img = x_set[i]
            img = cv2.resize(img, (ds_size[1], ds_size[0]), interpolation=cv2.INTER_LINEAR)
            img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)
            img = img.astype(np.uint8)
            x_ds.append(img)

       # data = pd.DataFrame(data={'x': x_ds, 'y': y_set, 'z': z_set, 't': t_set, 'p': p_set})
        data = pd.DataFrame(data={'x': x_ds, 'y': y_set, 'z': z_set, 't': t_set})
        df = pd.concat([data, sizes], axis=1)
        df.to_pickle(new_path)

    @staticmethod
    def PruneBadFrames(pickle_path, pickle_name):
        """Prune Dataset

                      Parameters
                    ----------
                    pickle_path : str
                        The file location of the first .pickle
                    pickle_name : str
                        The name/path of the newly created dataset

                   """

        dataset = pd.read_pickle(pickle_path)
        logging.info('[DataManipulator] dataset shape: ' + str(dataset.shape))

        h, w, c = DataManipulator.GetSizeDataFromDataFrame(dataset)
        sizes = DataManipulator.CreateSizeDataFrame(h, w, c)

        x_set = dataset['x'].values
        y_set = dataset['y'].values
        z_set = dataset['z'].values
        t_set = dataset['t'].values

        x_augset = []
        y_augset = []
        z_augset = []
        t_augset = []


        for i in range(len(x_set)):
            rel_pose = y_set[i]
            if ((rel_pose[0] > 1.0) and (rel_pose[0] < 3.5)):
                if ((rel_pose[3] > -np.pi/2.5) and (rel_pose[3] < np.pi/2.5)):
                    if ((rel_pose[2] > -0.5) and (rel_pose[2] < 0.5)):
                        if ((rel_pose[1] > -2) and (rel_pose[1] < 2)):
                            x_augset.append(x_set[i])
                            y_augset.append(y_set[i])
                            z_augset.append(z_set[i])
                            t_augset.append(t_set[i])


        data = pd.DataFrame(data={'x': x_augset, 'y': y_augset, 'z': z_augset, 't': t_augset})
        df = pd.concat([data, sizes], axis=1)
        df.to_pickle(pickle_name)

    @staticmethod
    def TrimDataset(pickle_path, pickle_name, start=None, end=None):
        """Prune Dataset

                      Parameters
                    ----------
                    pickle_path : str
                        The file location of the first .pickle
                    pickle_name : str
                        The name/path of the newly created dataset

                   """

        dataset = pd.read_pickle(pickle_path)
        logging.info('[DataManipulator] dataset shape: ' + str(dataset.shape))

        h, w, c = DataManipulator.GetSizeDataFromDataFrame(dataset)
        sizes = DataManipulator.CreateSizeDataFrame(h, w, c)

        x_set = dataset['x'].values
        y_set = dataset['y'].values
        z_set = dataset['z'].values
        t_set = dataset['t'].values

        x_augset = []
        y_augset = []
        z_augset = []
        t_augset = []
        if start is None:
            start = -1
        if end is None:
            end = len(x_set)


        for i in range(len(x_set)):
            # if(i > 2680 and i < 2720) or (i > 2900 and i < 2920):
            #     continue
            if (i > start) and (i < end):
            #if (i > 9 and i < 75) or (i > 80 and i < 395) or (i > 570 and i < 1390):
                x_augset.append(x_set[i])
                y_augset.append(y_set[i])
                z_augset.append(z_set[i])
                t_augset.append(t_set[i])

        data = pd.DataFrame(data={'x': x_augset, 'y': y_augset, 'z': z_augset, 't': t_augset})
        df = pd.concat([data, sizes], axis=1)
        df.to_pickle(pickle_name)
