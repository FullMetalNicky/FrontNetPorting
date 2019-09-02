# !/usr/bin/env python

import numpy as np 
import pandas as pd
import rosbag
import rospy
import cv2
import tf
from tf.transformations import euler_from_quaternion, quaternion_from_euler
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from TimestampSynchronizer import TimestampSynchronizer
from ImageEffects import ImageEffects
import subprocess

import sys
sys.path.append("../")
import config

class DatasetCreator:


	def __init__(self, bagName):
		self.bagName = bagName
		self.ts = TimestampSynchronizer(self.bagName)
		self.drone_topic = "optitrack/drone"
		self.camera_topic = "bebop/image_raw"
		self.body_topic = "optitrack/hand"
		self.rate = rospy.Rate(50.0)


	def Sync(self, delay):

		print("unpacking...")
		#unpack the stamps
		camera_stamps = self.ts.ExtractStampsFromHeader(self.camera_topic)
		optitrack_stamps = self.ts.ExtractStampsFromRosbag(self.body_topic)
		drone_stamps = self.ts.ExtractStampsFromRosbag(self.drone_topic )
		if((len(drone_stamps) < len(camera_stamps) ) or (len(optitrack_stamps) < len(camera_stamps))):
			print("Error:recording data corrupted. not enough MoCap stamps.") 
			return

		print("unpacked stamps")
		
		#get the sync ids 
		otherTopics = [optitrack_stamps, drone_stamps]
		sync_camera_ids, sync_other_ids = self.ts.SyncStampsToMain(camera_stamps, otherTopics, delay)
		sync_optitrack_ids = sync_other_ids[0]
		sync_drone_ids = sync_other_ids[1]	
		print("synced ids")

		return sync_camera_ids, sync_optitrack_ids, sync_drone_ids


	def BroadcastTF(self, msg, name):
		br = tf.TransformBroadcaster()
		br.sendTransform((msg.pose.position.x+config.dronemarker_offset , msg.pose.position.y, msg.pose.position.z),
		(msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w),
		rospy.Time.now(), name, "World")
		self.rate.sleep()


	def CalculateRelativePose(self, optitrack_msg, drone_msg):

		part_orient = optitrack_msg.pose.orientation
		drone_orient = drone_msg.pose.orientation
		part_pose = optitrack_msg.pose.position
		drone_pose = drone_msg.pose.position

		listener = tf.TransformListener()
		trans = None
		rot = None

		while ((trans == None) or (rot == None)):
			self.BroadcastTF(optitrack_msg, "/part")
			self.BroadcastTF(drone_msg, "/drone")
			try:
				now = rospy.Time(0)
				(trans,rot) = listener.lookupTransform("/drone", "/part", now)
		 	except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
				continue
	
		euler = tf.transformations.euler_from_quaternion(rot)
  		x = trans[0]
  		y = trans[1]
  		z = trans[2]
  		yaw = euler[2] 

		#print("part_pose={}".format(part_pose))
		#print("drone_pose={}".format(drone_pose))
		#print("rel_pose={}, {}, {}, {}".format(x,y,z,yaw))

		return x, y, z, yaw

	def FrameSelector(self):

		bridge = CvBridge()
		bebop_msgs_count = self.ts.GetMessagesCount(self.camera_topic)
		chunk_size = 1000
		chunks = (bebop_msgs_count/chunk_size) + 1
		start_frame = None
		end_frame = None
		for chunk in range(chunks):
			bebop_msgs = self.ts.GetMessages(self.camera_topic, chunk * chunk_size + 1, (chunk+1) * chunk_size)

			for i in range(len(bebop_msgs)):
				cv_image = bridge.imgmsg_to_cv2(bebop_msgs[i])
				bebop_id = chunk * chunk_size + i
				t = bebop_msgs[i].header.stamp.to_nsec()
				if  start_frame is None:
					cv2.imshow("hand", cv_image)
					key = cv2.waitKey(0)
					#print(t)
					if key == ord('s'):
						start_frame = t
				else:
					if (bebop_id > (bebop_msgs_count -200)) and (end_frame is None):
						cv2.imshow("hand", cv_image)
						key = cv2.waitKey(0)
						#print(t)
						if key == ord('s'):
							end_frame = t
							print("start={}, end={}".format(start_frame, end_frame)) 
							print("recording time {} seconds".format((end_frame-start_frame)/1000000000))
							cv2.destroyAllWindows()
							return start_frame, end_frame

		print("start={}, end={}".format(start_frame, t)) 
		print("recording time {} seconds".format((t-start_frame)/1000000000))
		cv2.destroyAllWindows()
		return start_frame, t


	def CreateBebopDataset(self, delay, isHand, datasetName, start = 0, end = sys.maxint):
	
		if isHand == True:
			self.body_topic = "optitrack/hand"
		else:
			self.body_topic = "optitrack/head"

		sync_bebop_ids, sync_optitrack_ids, sync_drone_ids = self.Sync(delay)
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
				t = bebop_msgs[i].header.stamp.to_nsec()
				if (t >= start) and (t <=end):
					cv_image = bridge.imgmsg_to_cv2(bebop_msgs[i])
					cv_image = cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR)
					cv_image = cv2.resize(cv_image, (config.input_width, config.input_height), cv2.INTER_AREA)
					x_dataset.append(cv_image)		
		
					optitrack_id = sync_optitrack_ids[chunk * chunk_size + i]		
					drone_id = sync_drone_ids[chunk * chunk_size + i]
					bebop_id = sync_bebop_ids[chunk * chunk_size + i]
					#print("opti_id={}/{}, drone_id={}/{}, bebop_id={}".format(optitrack_id, len(optitrack_msgs), drone_id, len(drone_msgs), bebop_id))

					x, y, z, yaw = self.CalculateRelativePose(optitrack_msgs[optitrack_id], drone_msgs[drone_id])
					if isHand == True:
						yaw = 0.0
					
					#y_dataset.append([int(isHand), x, y, z, yaw])
					y_dataset.append([x, y, z, yaw])

		print("dataset ready x:{} y:{}".format(len(x_dataset), len(y_dataset)))
		df = pd.DataFrame(data={'x': x_dataset, 'y': y_dataset})
		print("dataframe ready")
		df.to_pickle(datasetName)


	def CreateHimaxDataset(self, delay, isHand, datasetName, start = 0, end = sys.maxint):
	
		if isHand == True:
			self.body_topic = "optitrack/hand"
		else:
			self.body_topic = "optitrack/head"

		
		optitrack_msgs = self.ts.GetMessages(self.body_topic)
		drone_msgs = self.ts.GetMessages(self.drone_topic)
		bridge = CvBridge()

		#sync_bebop_ids, sync_optitrack_ids, sync_drone_ids = self.Sync(config.optitrack_delay)
		sync_bebop_ids, sync_optitrack_ids, sync_drone_ids = self.Sync(0)

		x_dataset = []
		y_dataset = []
	
		gammaLUT = ImageEffects.GetGammaLUT(0.6)
		vignetteMask = ImageEffects.GetVignetteMask(config.himax_width, config.himax_width)

		#read in chunks because memory is low
		bebop_msgs_count = self.ts.GetMessagesCount(self.camera_topic)
		chunk_size = 1000
		chunks = (bebop_msgs_count/chunk_size) + 1
		for chunk in range(chunks):
			bebop_msgs = self.ts.GetMessages(self.camera_topic, chunk * chunk_size + 1, (chunk+1) * chunk_size)

			for i in range(len(bebop_msgs)):
				cv_image = bridge.imgmsg_to_cv2(bebop_msgs[i])

				# image transform
				cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
				cv_image = cv2.LUT(cv_image, gammaLUT)
				cv_image = cv2.GaussianBlur(cv_image,(5,5),0)
				cv_image = cv2.resize(cv_image, (config.himax_width, config.himax_height), cv2.INTER_AREA)
				cv_image = cv_image *  vignetteMask[40:284, 0:324]
				cv_image = cv2.resize(cv_image, (config.input_width, config.input_height), cv2.INTER_NEAREST)
				x_dataset.append(cv_image)		
	
				optitrack_id = sync_optitrack_ids[chunk * chunk_size + i]		
				drone_id = sync_drone_ids[chunk * chunk_size + i]
				bebop_id = sync_bebop_ids[chunk * chunk_size + i]
				#print("opti_id={}/{}, drone_id={}/{}, bebop_id={}".format(optitrack_id, len(optitrack_msgs), drone_id, len(drone_msgs), bebop_id))
				#print("bebop id={}". format(bebop_id))
				#print("track_t={}, drone_t={}". format(drone_msgs[drone_id].header.stamp, optitrack_msgs[optitrack_id].header.stamp))

				x, y, z, yaw = self.CalculateRelativePose(optitrack_msgs[optitrack_id], drone_msgs[drone_id])
				if isHand == True:
					yaw = 0.0
				#y_dataset.append([int(isHand), x, y, z, yaw])
				y_dataset.append([x, y, z, yaw])

		print("finished transformed bebop")
		# self.camera_topic = "himax_camera"
		# himax_msgs = self.ts.GetMessages(self.camera_topic)
		# sync_himax_ids, sync_optitrack_ids, sync_drone_ids = self.Sync(delay)		

		# for i in range(len(himax_msgs)):
		# 	cv_image = bridge.imgmsg_to_cv2(himax_msgs[i])
		# 	cv_image = cv2.resize(cv_image, (config.input_width, config.input_height), cv2.INTER_AREA)
		# 	x_dataset.append(cv_image)		

		# 	optitrack_id = sync_optitrack_ids[i]		
		# 	drone_id = sync_drone_ids[i]
		# 	himax_id = sync_himax_ids[i]
		# 	#print("opti_id={}/{}, drone_id={}/{}, bebop_id={}".format(optitrack_id, len(optitrack_msgs), drone_id, len(drone_msgs), bebop_id))
			
		# 	x, y, z, yaw = self.CalculateRelativePose(optitrack_msgs[optitrack_id], drone_msgs[drone_id])
		# 	if isHand == True:
		# 			yaw = 0.0	
		# 	#y_dataset.append([int(isHand), x, y, z, yaw])
		# 	y_dataset.append([x, y, z, yaw])

		print("dataset ready x:{} y:{}".format(len(x_dataset), len(y_dataset)))
		df = pd.DataFrame(data={'x': x_dataset, 'y': y_dataset})
		print("dataframe ready")
		df.to_pickle(datasetName)


	@staticmethod
	def JoinPickleFiles(fileList, datasetName, folderPath=""):
		x_dataset = []
		y_dataset = []

		for file in fileList:
			dataset = pd.read_pickle(folderPath + file).values
			print(len(dataset[:, 0]))
			x_dataset.extend(dataset[:, 0])
			y_dataset.extend(dataset[:, 1])

		print("dataset ready x:{} y:{}".format(len(x_dataset), len(y_dataset)))
		df = pd.DataFrame(data={'x': x_dataset, 'y': y_dataset})
		print("dataframe ready")
		df.to_pickle(datasetName)



		
		

		
		
	



