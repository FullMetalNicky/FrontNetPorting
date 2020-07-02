#from DataManipulator import DataManipulator
import cv2
import logging
import pandas as pd

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
    #print("creating data set")
    #DataManipulator.AugmentAndCropMirko(DATA_PATH+"160x160HimaxTest16_4_2020.pickle", DATA_PATH2+"MirkoTest.pickle", [96, 160], 20)

    print("testing data set")
    dataset = pd.read_pickle(DATA_PATH2+"MirkoTest.pickle")
    frames = dataset['frame'].values
    frame_ids = dataset['frame_id'].values
    pitch = dataset['pitch'].values
    aug_ids = dataset['aug_id'].values
    drone_pose = dataset['drone_pose'].values
    rel_pose = dataset['rel_pose'].values
    t = dataset['timestamp'].values
    h = int(dataset['h'].values[0])
    w = int(dataset['w'].values[0])
    c = int(dataset['c'].values[0])

    for i in range(len(frame_ids)):
        print("frame id : {}, aug_id : {}, pitch : {}, drone_pose: {}, rel_pose : {}, t : {}".format( frame_ids[i],
              aug_ids[i], pitch[i], drone_pose[i],  rel_pose[i], t[i]))
        cv2.imshow("frame", frames[i])
        cv2.waitKey()



if __name__ == '__main__':
    main()