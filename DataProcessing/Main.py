

from DatasetCreator import DatasetCreator
import sys
sys.path.append("/home/usi/Documents/Drone/FrontNetPorting")
import config


def main():
	# subject_name = "davide1"
	# rosbagName = config.folder_path + "/data/Hand/" + subject_name + ".bag"
	# pickleName = config.folder_path + "/data/Hand/" + subject_name + "Hand.pickle"
	# CreateDatasetFromRosbag(rosbagName, pickleName, isBebop=True, start_frame=None, end_frame=None)

	subject_name = "session3"
	rosbagName = config.folder_path + "/data/compressed/" + subject_name + ".bag"
	pickleName = config.folder_path + "/data/compressed/" + subject_name + ".pickle"
	#CreateDatasetFromDarioRosbag(rosbagName, pickleName, start_frame=None, end_frame=None)

	dc = DatasetCreator(rosbagName)
	#dc.CreateBebopDataset(0, pickleName, "bebop/image_raw/compressed", "optitrack/drone", ["optitrack/head", "optitrack/hand"])
	dc.CreateHimaxDataset(config.himax_delay, pickleName, "himax_camera", "bebop/image_raw/compressed", "optitrack/drone", ["optitrack/head", "optitrack/hand"])

	# If you wish to join several .pickle files into one big .pickle, use
	# DatasetCreator.JoinPickleFiles(fileList, newPickleName, folderPath)


if __name__ == '__main__':
    main()