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

	#himaxImages = ImageIO.ReadImagesFromFolder(config.folder_path + "/data/himax/", '.jpg')
	#bebopImages = ImageIO.ReadImagesFromFolder(config.folder_path + "/data/bebop/", '.jpg')
	it = ImageTransformer()
	himax_fovx, himax_fovy, himax_h, himax_w = it.CalculateFOVfromCalibration(config.folder_path + "/data/calibration.yaml")
	bebop_fovx, bebop_fovy, bebop_h, bebop_w = it.CalculateFOVfromCalibration(config.folder_path + "/data/bebop_calibration.yaml")
	x_ratio = bebop_fovx / himax_fovx
	y_ratio = bebop_fovy / himax_fovy

	new_size, shift_x, shift_y = it.calc_crop_parameters(x_ratio, y_ratio, himax_h, himax_w)
	print("new size:{}, shift_x:{}, shift_y:{}".format(new_size, shift_x, shift_y))
	#new size:(306, 183), shift_x:9.0, shift_y:30.5
	#himaxTransImages, bebopTransImages = it. TransformImages(config.folder_path + "/data/calibration.yaml", config.folder_path + "/data/bebop_calibration.yaml", himaxImages, bebopImages)
	#ImageIO.WriteImagesToFolder(himaxTransImages, config.folder_path + "/data/test/", '.jpg')

def TestDatasetCreator():
	subject_name = "davide1"
	dc = DatasetCreator(config.folder_path + "/data/Hand/" + subject_name + ".bag")
	start_frame, end_frame = dc.FrameSelector()
	dc.CreateBebopDataset(0, config.folder_path + "/data/Hand/" + subject_name + "Hand.pickle", start_frame, end_frame)

	subject_name = "davide2"
	dc2 = DatasetCreator(config.folder_path + "/data/Hand/" + subject_name + ".bag")
	start_frame, end_frame = dc2.FrameSelector()
	dc2.CreateBebopDataset(0, config.folder_path + "/data/Hand/" + subject_name + "Hand.pickle", start_frame, end_frame)

	folderPath = config.folder_path + "/data/Hand/"
	fileList = ["davide1Hand.pickle", "davide2Hand.pickle"]
	DatasetCreator.JoinPickleFiles(fileList, config.folder_path + "/data/Hand/DavideHand.pickle", folderPath)
	

	

def main():
	#TestImageIO()
	#TestRosbagUnpacker()
	#TestCameraSynchronizer()
	TestImageTransformer()
	#TestDatasetCreator()


if __name__ == '__main__':
    main()
