from ImageIO import ImageIO
from CameraSynchronizer import CameraSynchronizer
from ImageTransformer import ImageTransformer
from DatasetCreator import DatasetCreator
import pandas as pd

import sys
sys.path.append("/home/usi/Documents/Drone/FrontNetPorting/pulp")
from CameraCalibration import CameraCalibration 
sys.path.append("/home/usi/Documents/Drone/FrontNetPorting")
import config

def TestImageIO():

	images = ImageIO.ReadImagesFromFolder(config.folder_path + "/data/himax/", '.jpg')
	ImageIO.WriteImagesToFolder(images, config.folder_path + "/data/test/", '.jpg')


def TestCameraCalibration():
	cc = CameraCalibration()
	images = cc.CaptureCalibration("image_pipe")
	cc.CalibrateImages(images, "test.yaml")


#def TestCameraSynchronizer():
	#re-write test

def TestImageTransformer():

	himaxImages = ImageIO.ReadImagesFromFolder(config.folder_path + "/data/himax/", '.jpg')
	bebopImages = ImageIO.ReadImagesFromFolder(config.folder_path + "/data/bebop/", '.jpg')
	it = ImageTransformer()
	himaxTransImages, bebopTransImages = it. TransformImages(config.folder_path + "/data/calibration.yaml", config.folder_path + "/data/bebop_calibration.yaml", himaxImages, bebopImages)
	ImageIO.WriteImagesToFolder(himaxTransImages, config.folder_path + "/data/test/", '.jpg')

def TestDatasetCreator():
	dc = DatasetCreator(config.folder_path + '/data/nickyrighthand.bag')
#	start_frame, end_frame = dc.FrameSelector()
	dc.CreateBebopDataset(0, True, "trainBebopHand.pickle", 1567175068652886038, 1567175152238984312)
	#dc.CreateHimaxDataset(config.himax_delay, False, "trainHimaxHead.pickle")
	#train_set = pd.read_pickle("trainHimaxHead.pickle").values
	
	#x_train = train_set[:, 0]
	#y_train = train_set[:, 1]
	#print(type(y_train))
	#print(y_train[0])


def main():
	#TestImageIO()
	#TestRosbagUnpacker()
	#TestCameraSynchronizer()
	#TestImageTransformer()
	TestDatasetCreator()


if __name__ == '__main__':
    main()
