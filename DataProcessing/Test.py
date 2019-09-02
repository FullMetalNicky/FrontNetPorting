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
	# subject_name = "nicky2"
	# dc = DatasetCreator(config.folder_path + "/data/Hand/" + subject_name + ".bag")
	# start_frame, end_frame = dc.FrameSelector()
	# dc.CreateBebopDataset(0, True, config.folder_path + "/data/Hand/" + subject_name + "Hand.pickle", start_frame, end_frame)
	#dc.CreateHimaxDataset(config.himax_delay, False, "trainHimaxHead.pickle")
	folderPath = config.folder_path + "/data/Hand/"
	fileList = ["lilithHand.pickle", "dario1Hand.pickle", "dario2Hand.pickle", "nicky1Hand.pickle", 
	"nicky2Hand.pickle", "mirko1Hand.pickle", "mirko2Hand.pickle"]
	DatasetCreator.JoinPickleFiles(fileList, config.folder_path + "/data/Hand/Hand.pickle", folderPath)
	

def main():
	#TestImageIO()
	#TestRosbagUnpacker()
	#TestCameraSynchronizer()
	#TestImageTransformer()
	TestDatasetCreator()


if __name__ == '__main__':
    main()
