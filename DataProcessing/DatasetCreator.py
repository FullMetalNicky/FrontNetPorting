# !/usr/bin/env python

import numpy as np 
import pandas as pd
import rosbag
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

import sys
sys.path.append("../pulp/")
from TimestampSynchronizer import TimestampSynchronizer


class DatasetCreator:


	def __init__(self, bagName):
		self.node = rospy.init_node('DatasetCreator', anonymous=True)
		self.bagName = bagName


	

		if str(msg._type) == "geometry_msgs/PoseStamped"

		

	def CreateBebopDataset(self, delay, isHand, datasetName):
		ts = TimestampSynchronizer(bagName)
		if isHand == True:
			optitrackTopic = "optitrack/hand"
		else:
			optitrackTopic = "optitrack/head"
		bebop_stamps, optitrack_stamps = ts.UnpackBagStamps("bebop/image_raw", optitrackTopic)
		sync_bebop_ids, sync_optitrack_ids = ts.SyncStamps(bebop_stamps, optitrack_stamps, delay)
		bebop_msgs, optitrack_msgs = ts.SyncTopicsByStamps("bebop/image_raw", optitrackTopic, sync_bebop_ids, sync_optitrack_ids)
		
		bebop_stamps, drone_stamps = ts.UnpackBagStamps("bebop/image_raw", "optitrack/drone")
		sync_bebop_ids, sync_drone_ids = ts.SyncStamps(bebop_stamps, drone_stamps, delay)
		bebop_msgs, drone_msgs = ts.SyncTopicsByStamps("bebop/image_raw", "optitrack/drone", sync_bebop_ids, sync_drone_ids)

		bridge = CvBridge()
		
		x_dataset = []
		y_dataset = []
		for i in range(len(bebop_msgs)):
			cv_image = bridge.imgmsg_to_cv2(bebop_msgs[i])
			cv_image = cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR)
			x_dataset.append(cv_image)
			part_pose = optitrack_msgs[i].pose.position
			drone_pose = drone_msgs[i].pose.position
			rel_pose = part_pose - drone_pose
			y_dataset.append([int(isHand), rel_pose])


		dataset = (x_dataset, y_dataset)
    df = pd.DataFrame(dataset)
    df.to_pickle(datasetName)

		
		
	



