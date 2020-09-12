from DataManipulator import DataManipulator
import cv2
import logging
import pandas as pd
import numpy as np

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

    DATA_PATH = "/Users/usi/PycharmProjects/data/160x160/"
    DATA_PATH2 = "/Users/usi/PycharmProjects/data/160x96/"
    # print("creating data set")
    # DataManipulator.AugmentAndCropMirko(DATA_PATH+"160x160HimaxTrain16_4_2020.pickle", DATA_PATH2+"MirkoTrain.pickle", [96, 160], 20)
    #
    # print("testing data set")
    # dataset = pd.read_pickle(DATA_PATH2+"MirkoTest.pickle")
    # frames = dataset['frame'].values
    # frame_ids = dataset['frame_id'].values
    # pitch = dataset['pitch'].values
    # aug_ids = dataset['aug_id'].values
    # drone_pose = dataset['drone_pose'].values
    # rel_pose = dataset['rel_pose'].values
    # t = dataset['timestamp'].values
    # h = int(dataset['h'].values[0])
    # w = int(dataset['w'].values[0])
    # c = int(dataset['c'].values[0])
    #
    # for i in range(len(frame_ids)):
    #     print("frame id : {}, aug_id : {}, pitch : {}, drone_pose: {}, rel_pose : {}, t : {}".format( frame_ids[i],
    #           aug_ids[i], pitch[i], drone_pose[i],  rel_pose[i], t[i]))
    #     cv2.imshow("frame", frames[i])
    #     cv2.waitKey()
    #

    testPath = DATA_PATH2 + "160x96PaperTestsetPrune2.pickle"

    test_set = pd.read_pickle(testPath)
    h = int(test_set['h'].values[0])
    w = int(test_set['w'].values[0])
    c = int(test_set['c'].values[0])
    x_test = test_set['x'].values

    x_test = np.vstack(x_test[:]).astype(np.float32)
    x_test = np.reshape(x_test, (-1, h, w, c))
    y_test = test_set['y'].values
    y_test = np.vstack(y_test[:]).astype(np.float32)
    z_test = test_set['z'].values
    z_test = np.vstack(z_test[:]).astype(np.float32)

    # h = x_test.shape[2]
    # w = x_test.shape[3]
    x_test = np.reshape(x_test, (-1, h, w))

    print("blah")

if __name__ == '__main__':
    main()