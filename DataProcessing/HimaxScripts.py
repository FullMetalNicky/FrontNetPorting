from DatasetCreator import DatasetCreator
import sys
import pandas as pd
import os

sys.path.append("/home/usi/Documents/Drone/FrontNetPorting")
import config



def CreatePickle(subject_name, rosbagfolder, delay=0):


	rosbagName = rosbagfolder +  subject_name + ".bag"
	pickleName = config.folder_path + "/../Pickles/16_4_2020/" + subject_name + ".pickle"
	dc = DatasetCreator(rosbagName)

	dc.CreateHimaxDataset(delay,pickleName, "/image_raw", "optitrack/gapuino", ["optitrack/head"])


def JoinPickles(fileList, picklename):

	picklefolder = config.folder_path + "/../Pickles/16_4_2020/"
	#fileList = {"Clip1.pickle", "Clip2.pickle", "Clip3.pickle", "Clip4.pickle", "Clip5.pickle", "Clip6.pickle"}
	DatasetCreator.JoinPickleFiles(fileList, picklefolder + picklename, picklefolder)

def AddColumnsToDataSet(picklename, height, width, channels ):

	picklefolder = config.folder_path + "/../Pickles/Dynamic/160x160/"
	dataset = pd.read_pickle(picklefolder + picklename)
	df = pd.DataFrame({
    'c': channels,
    'w' : width,
    'h' : height
}, index=[0])

	new = pd.concat([dataset, df], axis=1) 
	print(new.head)
	print("dataframe ready")
	new.to_pickle(picklefolder + picklename)

def ProcessAllFilesInFolder(rosbag_folder, pickle_folder):
	files = os.listdir(rosbag_folder)

	for f in files:

		#print(f)
		dc = DatasetCreator(rosbag_folder+f)
		pickleName = pickle_folder + os.path.splitext(os.path.basename(f))[0]+ ".pickle"
		#print(pickleName)
		dc.CreateHimaxDataset(0, pickleName, "/image_raw", "optitrack/gapuino", ["optitrack/head"])


def main():



	# fileList = []
	# for i in range(1,3):
	# 	name = "Pitch{}".format(i)
	# 	CreatePickle(name, rosbagfolder = config.folder_path+"/../Rosbags/16_4_2020/nicky/")
	# 	name = name + ".pickle"
	# 	#print(name)
	# 	fileList.append(name)
	ProcessAllFilesInFolder("/home/usi/Documents/Drone/Rosbags/16_4_2020/Regular/", "/home/usi/Documents/Drone/Pickles/16_4_2020/Regular/")
	
	# #CreatePickle("Clip6")
	# fileList = ["160x160HimaxDynamic_12_03_20.pickle", "160x160HimaxTest.pickle" ]
	#JoinPickles(fileList, "160x160Pitch.pickle")
	#CreatePickle("Test", rosbagfolder = config.folder_path+"/../Rosbags/normal/alessandro/")
	#AddColumnsToDataSet("train_grey.pickle", 60, 108, 1)


if __name__ == '__main__':
    main()