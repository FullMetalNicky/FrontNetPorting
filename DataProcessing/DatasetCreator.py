# !/usr/bin/env python

import numpy as np 
import pandas as pd
import rosbag
import rospy
import cv2
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

import sys
sys.path.append("../pulp/")
from TimestampSynchronizer import TimestampSynchronizer
from ImageTransformer import ImageTransformer


class DatasetCreator:


	def __init__(self, bagName):
		self.bagName = bagName
		self.ts = TimestampSynchronizer(self.bagName)
		self.drone_topic = "optitrack/drone"
		self.camera_topic = "bebop/image_raw"
		self.body_topic = "optitrack/hand"


	def Try(self, delay):

		#unpack the stamps
		bebop_stamps = self.ts.UnpackBagStampsSingle(self.camera_topic)
		optitrack_stamps = self.ts.UnpackBagStampsSingle(self.body_topic)
		drone_stamps = self.ts.UnpackBagStampsSingle(self.drone_topic )
		if((drone_stamps < bebop_stamps ) or (optitrack_stamps < bebop_stamps)):
			print("Error:recording data corrupted. not enough MoCap stamps.") 
			return

		print("unpacked stamps")
		
		#get the sync ids 
		otherTopics = [optitrack_stamps, drone_stamps]
		sync_bebop_ids, sync_other_ids = self.ts.SyncStampsToMain(bebop_stamps, otherTopics, delay)
		sync_optitrack_ids = sync_other_ids[0]
		sync_drone_ids = sync_other_ids[1]	
		print("synced ids")

		return sync_bebop_ids, sync_optitrack_ids, sync_drone_ids

	def CreateBebopDataset(self, delay, isHand, datasetName):
	
		if isHand == True:
			self.body_topic = "optitrack/hand"
		else:
			self.body_topic = "optitrack/head"

		
		sync_bebop_ids, sync_optitrack_ids, sync_drone_ids = self.Try(delay)
		optitrack_msgs = self.ts.GetMessages(self.body_topic)
		drone_msgs = self.ts.GetMessages(self.drone_topic)
		
		bridge = CvBridge()
		
		
		x_dataset = []
		y_dataset = []

		#read in chunks because memory is low
		bebop_msgs_count = self.ts.GetMessagesCount(self.camera_topic)
		chunk_size = 1000
		chunks = (bebop_msgs_count/chunk_size) + 1
		for chunk in range(chunks):
			bebop_msgs = self.ts.GetMessages(self.camera_topic, chunk * chunk_size + 1, (chunk+1) * chunk_size)

			for i in range(len(bebop_msgs)):
				cv_image = bridge.imgmsg_to_cv2(bebop_msgs[i])
				cv_image = cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR)
				cv_image = cv2.resize(cv_image, (108, 60), cv2.INTER_AREA)
				x_dataset.append(cv_image)		
	
				optitrack_id = sync_optitrack_ids[chunk * chunk_size + i]		
				drone_id = sync_drone_ids[chunk * chunk_size + i]
				bebop_id = sync_bebop_ids[chunk * chunk_size + i]
				#print("opti_id={}/{}, drone_id={}/{}, bebop_id={}".format(optitrack_id, len(optitrack_msgs), drone_id, len(drone_msgs), bebop_id))
				
				part_pose = optitrack_msgs[optitrack_id].pose.position
				drone_pose = drone_msgs[drone_id].pose.position
				rel_pose = PoseStamped().pose.position
				rel_pose.x = part_pose.x - drone_pose.x
				rel_pose.y = part_pose.y - drone_pose.y
				rel_pose.z = part_pose.z - drone_pose.z

				#print("isHand={}, part_pose={}".format(int(isHand), part_pose))
				#print("drone_pose={}".format(drone_pose))
				#print("rel_pose={}".format(rel_pose))
				#print([int(isHand), rel_pose.x, rel_pose.y, rel_pose.z])
				y_dataset.append([int(isHand), rel_pose.x, rel_pose.y, rel_pose.z])

		print("dataset ready x:{} y:{}".format(len(x_dataset), len(y_dataset)))
		dataset = (x_dataset, y_dataset)
		print(sys.getsizeof(x_dataset))
		#df = pd.DataFrame(dataset)
		#print("dataframe ready")
		#df.to_pickle(datasetName)

		
		
	



