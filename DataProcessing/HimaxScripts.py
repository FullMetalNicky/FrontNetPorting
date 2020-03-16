from DatasetCreator import DatasetCreator
import sys
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



def main():


	# fileList = []
	# for i in range(1,6):
	# 	name = "Clip{}".format(i)
	# 	CreatePickle(name, rosbagfolder = config.folder_path+"/../Rosbags/normal/alessandro/")
	# 	fileList.append(name + ".pickle")
	
	# #CreatePickle("Clip6")
	# JoinPickles(fileList, "HimaxDynamic_12_03_20.pickle")
	CreatePickle("Test", rosbagfolder = config.folder_path+"/../Rosbags/normal/alessandro/")


if __name__ == '__main__':
    main()