

from DatasetCreator import DatasetCreator
import sys
sys.path.append("/home/usi/Documents/Drone/FrontNetPorting")
import config




def CreateDatasetFromRosbag(rosbagName, pickleName, isBebop=True, start_frame=None, end_frame=None):
	"""Converts rosbag to format suitable for training/testing. 
	if start_frame, end_frame are unknown, FrameSelector will help you choose how to trim the video

    Parameters
    ----------
    rosbagName : str
        The file location of the rosbag
    pickleName : str
        name of the new .pickle file
    isBebop : bool, optional
        True if you want RGB dataset for the bebop, False if you want Himax-tailored dataset
    start_frame : int, optional
        if known, the timestamp in ns of the frame you wish to start from 
    end_frame : int, optional
        if known, the timestamp in ns of the frame you wish to finish at
    """

	dc = DatasetCreator(rosbagName)
	if (start_frame is None) or (end_frame is None):
		start_frame, end_frame = dc.FrameSelector()

	if isBebop == True:
		dc.CreateBebopDataset(0, pickleName, start_frame, end_frame)
	else:
		dc.CreateHimaxDataset(config.himax_delay, pickleName, start_frame, end_frame)


def CreateDatasetFromDarioRosbag(rosbagName, pickleName, start_frame=None, end_frame=None):
	"""Converts Dario's rosbag to format suitable for training/testing. 
	if start_frame, end_frame are unknown, FrameSelector will help you choose how to trim the video

    Parameters
    ----------
    rosbagName : str
        The file location of the rosbag
    pickleName : str
        name of the new .pickle file
    start_frame : int, optional
        if known, the timestamp in ns of the frame you wish to start from 
    end_frame : int, optional
        if known, the timestamp in ns of the frame you wish to finish at
    """

	dc = DatasetCreator(rosbagName)
	if (start_frame is None) or (end_frame is None):
		start_frame, end_frame = dc.FrameSelector(True)

	dc.CreateBebopDarioDataset(0, pickleName, start_frame, end_frame)

		




def main():
	# subject_name = "davide1"
	# rosbagName = config.folder_path + "/data/Hand/" + subject_name + ".bag"
	# pickleName = config.folder_path + "/data/Hand/" + subject_name + "Hand.pickle"
	# CreateDatasetFromRosbag(rosbagName, pickleName, isBebop=True, start_frame=None, end_frame=None)

	subject_name = "3"
	rosbagName = config.folder_path + "/data/" + subject_name + ".bag"
	pickleName = config.folder_path + "/data/" + subject_name + ".pickle"
	CreateDatasetFromDarioRosbag(rosbagName, pickleName, start_frame=None, end_frame=None)

	# If you wish to join several .pickle files into one big .pickle, use
	# DatasetCreator.JoinPickleFiles(fileList, newPickleName, folderPath)


if __name__ == '__main__':
    main()