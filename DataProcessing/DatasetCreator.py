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
from tqdm import tqdm
import PyKDL

import sys
sys.path.append("../")
import config

def _frame(stamped_pose):
	q = stamped_pose.pose.orientation
	p = stamped_pose.pose.position
	rotation = PyKDL.Rotation.Quaternion(q.x, q.y, q.z, q.w)
	translation = PyKDL.Vector(p.x, p.y, p.z)
	return PyKDL.Frame(rotation, translation)

def _pose(frame, reference_frame='reference'):
    pose = PoseStamped()
    pose.pose.position.x = frame[(0, 3)]
    pose.pose.position.y = frame[(1, 3)]
    pose.pose.position.z = frame[(2, 3)]
    o = pose.pose.orientation
    (o.x, o.y, o.z, o.w) = frame.M.GetQuaternion()
    pose.header.frame_id = reference_frame
    return pose

def relative_pose(stamped_pose, reference_pose, reference_frame='reference'):
	frame = _frame(stamped_pose)
	ref_frame = _frame(reference_pose)
	rel_frame = ref_frame.Inverse() * frame
	res = _pose(rel_frame, reference_frame)

	position = res.pose.position
	x, y, z = position.x , position.y, position.z
	q = res.pose.orientation
	_, _, yaw = tf.transformations.euler_from_quaternion((q.x, q.y, q.z, q.w))
	yaw = yaw - np.pi # 0 pointing towards the drone
	yaw = (yaw + np.pi) % (2 * np.pi) - np.pi

	return x, y, z, yaw


class DatasetCreator:


	def __init__(self, bagName):
		self.bagName = bagName
		self.ts = TimestampSynchronizer(self.bagName)
		self.drone_topic = "optitrack/drone"
		self.camera_topic = "bebop/image_raw"
		self.body_topic = "optitrack/hand"

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


	def CreateBebopDataset(self, delay, datasetName, start = 0, end = sys.maxint):
	
		print("unpacking...")
		#unpack the stamps
		camera_stamps = self.ts.ExtractStampsFromHeader(self.camera_topic)
		hand_stamps = self.ts.ExtractStampsFromRosbag("optitrack/hand")
		head_stamps = self.ts.ExtractStampsFromRosbag("optitrack/head")
		drone_stamps = self.ts.ExtractStampsFromRosbag(self.drone_topic)

		if((len(drone_stamps) < len(camera_stamps) ) or (len(hand_stamps) < len(camera_stamps)) or (len(head_stamps) < len(camera_stamps))):
			print("Error:recording data corrupted. not enough MoCap stamps.") 
			return

		print("unpacked stamps")
		
		#get the sync ids 
		otherTopics = [hand_stamps, head_stamps, drone_stamps]
		sync_bebop_ids, sync_other_ids = self.ts.SyncStampsToMain(camera_stamps, otherTopics, delay)
		sync_hand_ids = sync_other_ids[0]
		sync_head_ids = sync_other_ids[1]
		sync_drone_ids = sync_other_ids[2]	

		print("synced ids")

		hand_msgs = self.ts.GetMessages("optitrack/hand")
		head_msgs = self.ts.GetMessages("optitrack/head")
		drone_msgs = self.ts.GetMessages(self.drone_topic)
		bridge = CvBridge()
		
		x_dataset = []
		hand_dataset = []
		head_dataset = []

		#read in chunks because memory is low
		bebop_msgs_count = self.ts.GetMessagesCount(self.camera_topic)
		chunk_size = 1000
		chunks = (bebop_msgs_count/chunk_size) + 1
		for chunk in tqdm(range(chunks)):
			bebop_msgs = self.ts.GetMessages(self.camera_topic, chunk * chunk_size + 1, (chunk+1) * chunk_size)

			for i in tqdm(range(len(bebop_msgs))):
				t = bebop_msgs[i].header.stamp.to_nsec()
				if (t >= start) and (t <=end):
					cv_image = bridge.imgmsg_to_cv2(bebop_msgs[i])
					cv_image = cv2.resize(cv_image, (config.input_width, config.input_height), cv2.INTER_AREA)
					x_dataset.append(cv_image)		
		
					hand_id = sync_hand_ids[chunk * chunk_size + i]		
					head_id = sync_head_ids[chunk * chunk_size + i]		
					drone_id = sync_drone_ids[chunk * chunk_size + i]
					bebop_id = sync_bebop_ids[chunk * chunk_size + i]
					#print("opti_id={}/{}, drone_id={}/{}, bebop_id={}".format(optitrack_id, len(optitrack_msgs), drone_id, len(drone_msgs), bebop_id))

					x, y, z, yaw = relative_pose(hand_msgs[hand_id], drone_msgs[drone_id])
					hand_dataset.append([x, y, z, yaw])

					x, y, z, yaw = relative_pose(head_msgs[head_id], drone_msgs[drone_id])
					head_dataset.append([x, y, z, yaw])

		print("dataset ready x:{} hand:{} head:{}".format(len(x_dataset), len(hand_dataset), len(head_dataset)))
		df = pd.DataFrame(data={'x': x_dataset, 'y': hand_dataset, 'z' : head_dataset})
		print("dataframe ready")
		df.to_pickle(datasetName)


	def CreateHimaxDataset(self, delay, isHand, datasetName, start = 0, end = sys.maxint):
	
		print("unpacking...")
		#unpack the stamps
		camera_stamps = self.ts.ExtractStampsFromHeader(self.camera_topic)
		hand_stamps = self.ts.ExtractStampsFromRosbag("optitrack/hand")
		head_stamps = self.ts.ExtractStampsFromRosbag("optitrack/head")
		drone_stamps = self.ts.ExtractStampsFromRosbag(self.drone_topic)

		if((len(drone_stamps) < len(camera_stamps) ) or (len(hand_stamps) < len(camera_stamps)) or (len(head_stamps) < len(camera_stamps))):
			print("Error:recording data corrupted. not enough MoCap stamps.") 
			return

		print("unpacked stamps")
		
		#get the sync ids 
		otherTopics = [hand_stamps, head_stamps, drone_stamps]
		sync_bebop_ids, sync_other_ids = self.ts.SyncStampsToMain(camera_stamps, otherTopics, delay)
		sync_hand_ids = sync_other_ids[0]
		sync_head_ids = sync_other_ids[1]
		sync_drone_ids = sync_other_ids[2]	

		hand_msgs = self.ts.GetMessages("optitrack/hand")
		head_msgs = self.ts.GetMessages("optitrack/head")
		drone_msgs = self.ts.GetMessages(self.drone_topic)
		bridge = CvBridge()
		
		x_dataset = []
		hand_dataset = []
		head_dataset = []
	
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
	
				hand_id = sync_hand_ids[chunk * chunk_size + i]		
				head_id = sync_head_ids[chunk * chunk_size + i]		
				drone_id = sync_drone_ids[chunk * chunk_size + i]
				bebop_id = sync_bebop_ids[chunk * chunk_size + i]
				#print("opti_id={}/{}, drone_id={}/{}, bebop_id={}".format(optitrack_id, len(optitrack_msgs), drone_id, len(drone_msgs), bebop_id))

				x, y, z, yaw = relative_pose(hand_msgs[hand_id], drone_msgs[drone_id])
				hand_dataset.append([x, y, z, yaw])

				x, y, z, yaw = relative_pose(head_msgs[head_id], drone_msgs[drone_id])
				head_dataset.append([x, y, z, yaw])

		print("finished transformed bebop")
		self.camera_topic = "himax_camera"
		camera_stamps = self.ts.ExtractStampsFromHeader(self.camera_topic)
		himax_msgs = self.ts.GetMessages(self.camera_topic)
		#get the sync ids 
		otherTopics = [hand_stamps, head_stamps, drone_stamps]
		sync_himax_ids, sync_other_ids = self.ts.SyncStampsToMain(camera_stamps, otherTopics, delay)
		sync_hand_ids = sync_other_ids[0]
		sync_head_ids = sync_other_ids[1]
		sync_drone_ids = sync_other_ids[2]		

		for i in range(len(himax_msgs)):
			cv_image = bridge.imgmsg_to_cv2(himax_msgs[i])
			#need to crop too?
			cv_image = cv2.resize(cv_image, (config.input_width, config.input_height), cv2.INTER_AREA)
			x_dataset.append(cv_image)		

			optitrack_id = sync_optitrack_ids[i]		
			head_id = sync_head_ids[i]
			hand_id = sync_hand_ids[i]
			himax_id = sync_himax_ids[i]

			x, y, z, yaw = relative_pose(hand_msgs[hand_id], drone_msgs[drone_id])
			hand_dataset.append([x, y, z, yaw])

			x, y, z, yaw = relative_pose(head_msgs[head_id], drone_msgs[drone_id])
			head_dataset.append([x, y, z, yaw])
			

		print("dataset ready x:{} hand:{} head:{}".format(len(x_dataset), len(hand_dataset), len(head_dataset)))
		df = pd.DataFrame(data={'x': x_dataset, 'y': hand_dataset, 'z' : head_dataset})
		print("dataframe ready")
		df.to_pickle(datasetName)


	@staticmethod
	def JoinPickleFiles(fileList, datasetName, folderPath=""):
		x_dataset = []
		y_dataset = []
		z_dataset = []

		for file in fileList:
			dataset = pd.read_pickle(folderPath + file).values
			print(len(dataset[:, 0]))
			x_dataset.extend(dataset[:, 0])
			y_dataset.extend(dataset[:, 1])
			z_dataset.extend(dataset[:, 2])

		print("dataset ready x:{} hand:{} head:{}".format(len(x_dataset), len(y_dataset), len(z_dataset)))
		df = pd.DataFrame(data={'x': x_dataset, 'y': y_dataset, 'z' : z_dataset})
		print("dataframe ready")
		df.to_pickle(datasetName)




		
		

		
		
	



