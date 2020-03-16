from DatasetCreator import DatasetCreator
import sys
import pandas as pd
sys.path.append("/home/usi/Documents/Drone/FrontNetPorting")
import config



def CreatePickle(subject_name, rosbagfolder, delay=0):


	rosbagName = rosbagfolder +  subject_name + ".bag"
	pickleName = config.folder_path + "/../Pickles/" + subject_name + ".pickle"
	dc = DatasetCreator(rosbagName)

	dc.CreateHimaxDataset(delay,pickleName, "/image_raw", "/optitrack/gapuino", ["/optitrack/head"], pose = True )


def JoinPickles(fileList, picklename):

	picklefolder = config.folder_path + "/../Pickles/"
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



def main():


	# fileList = []
	# for i in range(1,6):
	# 	name = "Clip{}".format(i)
	# 	CreatePickle(name, rosbagfolder = config.folder_path+"/../Rosbags/normal/alessandro/")
	# 	fileList.append(name + ".pickle")
	
	# #CreatePickle("Clip6")
	# JoinPickles(fileList, "HimaxDynamic_12_03_20.pickle")
	#CreatePickle("Test", rosbagfolder = config.folder_path+"/../Rosbags/normal/alessandro/")
	AddColumnsToDataSet("Test.pickle", 160, 160, 1)


if __name__ == '__main__':
    main()