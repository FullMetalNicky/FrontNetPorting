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
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge
from TimestampSynchronizer import TimestampSynchronizer
from ImageEffects import ImageEffects
import subprocess
from tqdm import tqdm
import PyKDL
import os

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

def ExtractPoseFromMessage(Pose):
	#print(Pose)
	position = Pose.pose.position
	x, y, z = position.x , position.y, position.z
	q = Pose.pose.orientation
	_, _, yaw = tf.transformations.euler_from_quaternion((q.x, q.y, q.z, q.w))

	return x, y, z, yaw



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

	def FrameSelector(self, isCompressed):
		"""Helps to select where to start and stop the video
		When you wish to mark a frame for start/end, press 's', for viewing the next frame press any other key
		"""

		# if isDarioBag==True:
		# 	self.camera_topic = "bebop/image_raw/compressed"
		# else:
		# 	self.camera_topic = "bebop/image_raw"


		bridge = CvBridge()
		bebop_msgs_count = self.ts.GetMessagesCount(self.camera_topic)
		chunk_size = 1000
		chunks = (bebop_msgs_count/chunk_size) + 1
		start_frame = None
		end_frame = None
		for chunk in range(chunks):
			bebop_msgs = self.ts.GetMessages(self.camera_topic, chunk * chunk_size + 1, (chunk+1) * chunk_size)

			for i in range(len(bebop_msgs)):
				if isCompressed==True:
					cv_image = bridge.compressed_imgmsg_to_cv2(bebop_msgs[i])
				else:
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


	# def CreateBebopDataset(self, delay, datasetName, start = 0, end = sys.maxint):
	# 	"""Converts rosbag to format suitable for training/testing. 
	# 	if start_frame, end_frame are unknown, FrameSelector will help you choose how to trim the video
	# 	the output .pickle is organized as 'x': video frames, 'y': hand poses, 'z' : head poses

	#     Parameters
	#     ----------
	#     delay : int
	#         The delay between the camera and the optitrack 
	#     datasetName : str
	#         name of the new .pickle file
	#     start_frame : int, optional
	#         if known, the timestamp in ns of the frame you wish to start from 
	#     end_frame : int, optional
	#         if known, the timestamp in ns of the frame you wish to finish at
	#     """
	
	# 	self.camera_topic = "bebop/image_raw"
	# 	self.drone_topic = "optitrack/drone"
	# 	print("unpacking...")
	# 	#unpack the stamps
	# 	camera_stamps = self.ts.ExtractStampsFromHeader(self.camera_topic)
	# 	hand_stamps = self.ts.ExtractStampsFromRosbag("optitrack/hand")
	# 	head_stamps = self.ts.ExtractStampsFromRosbag("optitrack/head")
	# 	drone_stamps = self.ts.ExtractStampsFromRosbag(self.drone_topic)

	# 	if((len(drone_stamps) < len(camera_stamps) ) or (len(hand_stamps) < len(camera_stamps)) or (len(head_stamps) < len(camera_stamps))):
	# 		print("Error:recording data corrupted. not enough MoCap stamps.") 
	# 		return

	# 	print("unpacked stamps")
		
	# 	#get the sync ids 
	# 	otherTopics = [hand_stamps, head_stamps, drone_stamps]
	# 	sync_bebop_ids, sync_other_ids = self.ts.SyncStampsToMain(camera_stamps, otherTopics, delay)
	# 	sync_hand_ids = sync_other_ids[0]
	# 	sync_head_ids = sync_other_ids[1]
	# 	sync_drone_ids = sync_other_ids[2]	

	# 	print("synced ids")

	# 	hand_msgs = self.ts.GetMessages("optitrack/hand")
	# 	head_msgs = self.ts.GetMessages("optitrack/head")
	# 	drone_msgs = self.ts.GetMessages(self.drone_topic)
	# 	bridge = CvBridge()
		
	# 	x_dataset = []
	# 	hand_dataset = []
	# 	head_dataset = []

	# 	#read in chunks because memory is low
	# 	bebop_msgs_count = self.ts.GetMessagesCount(self.camera_topic)
	# 	chunk_size = 1000
	# 	chunks = (bebop_msgs_count/chunk_size) + 1
	# 	for chunk in tqdm(range(chunks)):
	# 		bebop_msgs = self.ts.GetMessages(self.camera_topic, chunk * chunk_size + 1, (chunk+1) * chunk_size)

	# 		for i in tqdm(range(len(bebop_msgs))):
	# 			t = bebop_msgs[i].header.stamp.to_nsec()
	# 			if (t >= start) and (t <=end):
	# 				cv_image = bridge.imgmsg_to_cv2(bebop_msgs[i])
	# 				cv_image = cv2.resize(cv_image, (config.input_width, config.input_height), cv2.INTER_AREA)
	# 				x_dataset.append(cv_image)		
		
	# 				hand_id = sync_hand_ids[chunk * chunk_size + i]		
	# 				head_id = sync_head_ids[chunk * chunk_size + i]		
	# 				drone_id = sync_drone_ids[chunk * chunk_size + i]
	# 				bebop_id = sync_bebop_ids[chunk * chunk_size + i]
	# 				#print("opti_id={}/{}, drone_id={}/{}, bebop_id={}".format(optitrack_id, len(optitrack_msgs), drone_id, len(drone_msgs), bebop_id))

	# 				x, y, z, yaw = relative_pose(hand_msgs[hand_id], drone_msgs[drone_id])
	# 				hand_dataset.append([x, y, z, yaw])

	# 				x, y, z, yaw = relative_pose(head_msgs[head_id], drone_msgs[drone_id])
	# 				head_dataset.append([x, y, z, yaw])

	# 	print("dataset ready x:{} hand:{} head:{}".format(len(x_dataset), len(hand_dataset), len(head_dataset)))
	# 	df = pd.DataFrame(data={'x': x_dataset, 'y': hand_dataset, 'z' : head_dataset})
	# 	print("dataframe ready")
	# 	df.to_pickle(datasetName)


	# def CreateHimaxDataset(self, delay, datasetName, start = 0, end = sys.maxint):
	# 	"""Converts rosbag to format suitable for training/testing. 
	# 	if start_frame, end_frame are unknown, FrameSelector will help you choose how to trim the video
	# 	the output .pickle is organized as 'x': video frames, 'y': hand poses, 'z' : head poses

	#     Parameters
	#     ----------
	#     delay : int
	#         The delay between the himax camera and the optitrack or bebop camera
	#     datasetName : str
	#         name of the new .pickle file
	#     start_frame : int, optional
	#         if known, the timestamp in ns of the frame you wish to start from 
	#     end_frame : int, optional
	#         if known, the timestamp in ns of the frame you wish to finish at
	#     """
	
	# 	self.camera_topic = "bebop/image_raw"
	# 	self.drone_topic = "optitrack/drone"
	# 	print("unpacking...")
	# 	#unpack the stamps
	# 	camera_stamps = self.ts.ExtractStampsFromHeader(self.camera_topic)
	# 	hand_stamps = self.ts.ExtractStampsFromRosbag("optitrack/hand")
	# 	head_stamps = self.ts.ExtractStampsFromRosbag("optitrack/head")
	# 	drone_stamps = self.ts.ExtractStampsFromRosbag(self.drone_topic)

	# 	if((len(drone_stamps) < len(camera_stamps) ) or (len(hand_stamps) < len(camera_stamps)) or (len(head_stamps) < len(camera_stamps))):
	# 		print("Error:recording data corrupted. not enough MoCap stamps.") 
	# 		return

	# 	print("unpacked stamps")
		
	# 	#get the sync ids 
	# 	otherTopics = [hand_stamps, head_stamps, drone_stamps]
	# 	sync_bebop_ids, sync_other_ids = self.ts.SyncStampsToMain(camera_stamps, otherTopics, 0)
	# 	sync_hand_ids = sync_other_ids[0]
	# 	sync_head_ids = sync_other_ids[1]
	# 	sync_drone_ids = sync_other_ids[2]	

	# 	hand_msgs = self.ts.GetMessages("optitrack/hand")
	# 	head_msgs = self.ts.GetMessages("optitrack/head")
	# 	drone_msgs = self.ts.GetMessages(self.drone_topic)
	# 	bridge = CvBridge()
		
	# 	x_dataset = []
	# 	hand_dataset = []
	# 	head_dataset = []
	
	# 	gammaLUT = ImageEffects.GetGammaLUT(0.6)
	# 	vignetteMask = ImageEffects.GetVignetteMask(config.himax_width, config.himax_width)

	# 	#read in chunks because memory is low
	# 	bebop_msgs_count = self.ts.GetMessagesCount(self.camera_topic)
	# 	chunk_size = 1000
	# 	chunks = (bebop_msgs_count/chunk_size) + 1
	# 	for chunk in range(chunks):
	# 		bebop_msgs = self.ts.GetMessages(self.camera_topic, chunk * chunk_size + 1, (chunk+1) * chunk_size)

	# 		for i in range(len(bebop_msgs)):
	# 			cv_image = bridge.imgmsg_to_cv2(bebop_msgs[i])

	# 			# image transform
	# 			cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
	# 			cv_image = cv2.LUT(cv_image, gammaLUT)
	# 			cv_image = cv2.GaussianBlur(cv_image,(5,5),0)
	# 			cv_image = cv2.resize(cv_image, (config.himax_width, config.himax_height), cv2.INTER_AREA)
	# 			cv_image = cv_image *  vignetteMask[40:284, 0:324]
	# 			cv_image = cv2.resize(cv_image, (config.input_width, config.input_height), cv2.INTER_NEAREST)
	# 			x_dataset.append(cv_image)		
	
	# 			hand_id = sync_hand_ids[chunk * chunk_size + i]		
	# 			head_id = sync_head_ids[chunk * chunk_size + i]		
	# 			drone_id = sync_drone_ids[chunk * chunk_size + i]
	# 			bebop_id = sync_bebop_ids[chunk * chunk_size + i]
	# 			#print("opti_id={}/{}, drone_id={}/{}, bebop_id={}".format(optitrack_id, len(optitrack_msgs), drone_id, len(drone_msgs), bebop_id))

	# 			x, y, z, yaw = relative_pose(hand_msgs[hand_id], drone_msgs[drone_id])
	# 			hand_dataset.append([x, y, z, yaw])

	# 			x, y, z, yaw = relative_pose(head_msgs[head_id], drone_msgs[drone_id])
	# 			head_dataset.append([x, y, z, yaw])

	# 	print("finished transformed bebop")
	# 	self.camera_topic = "himax_camera"
	# 	camera_stamps = self.ts.ExtractStampsFromHeader(self.camera_topic)
	# 	himax_msgs = self.ts.GetMessages(self.camera_topic)
	# 	#get the sync ids 
	# 	otherTopics = [hand_stamps, head_stamps, drone_stamps]
	# 	sync_himax_ids, sync_other_ids = self.ts.SyncStampsToMain(camera_stamps, otherTopics, delay)
	# 	sync_hand_ids = sync_other_ids[0]
	# 	sync_head_ids = sync_other_ids[1]
	# 	sync_drone_ids = sync_other_ids[2]		

	# 	for i in range(len(himax_msgs)):
	# 		cv_image = bridge.imgmsg_to_cv2(himax_msgs[i])
	# 		#need to crop too?
	# 		cv_image = cv2.resize(cv_image, (config.input_width, config.input_height), cv2.INTER_AREA)
	# 		x_dataset.append(cv_image)		

	# 		optitrack_id = sync_optitrack_ids[i]		
	# 		head_id = sync_head_ids[i]
	# 		hand_id = sync_hand_ids[i]
	# 		himax_id = sync_himax_ids[i]

	# 		x, y, z, yaw = relative_pose(hand_msgs[hand_id], drone_msgs[drone_id])
	# 		hand_dataset.append([x, y, z, yaw])

	# 		x, y, z, yaw = relative_pose(head_msgs[head_id], drone_msgs[drone_id])
	# 		head_dataset.append([x, y, z, yaw])
			

	# 	print("dataset ready x:{} hand:{} head:{}".format(len(x_dataset), len(hand_dataset), len(head_dataset)))
	# 	df = pd.DataFrame(data={'x': x_dataset, 'y': hand_dataset, 'z' : head_dataset})
	# 	print("dataframe ready")
	# 	df.to_pickle(datasetName)


	# def CreateBebopDarioDataset(self, delay, datasetName, start = 0, end = sys.maxint):
	# 	"""Converts rosbag to format suitable for training/testing. 
	# 	if start_frame, end_frame are unknown, FrameSelector will help you choose how to trim the video
	# 	the output .pickle is organized as 'x': video frames, 'y': hand poses, 'z' : head poses

	#     Parameters
	#     ----------
	#     delay : int
	#         The delay between the camera and the optitrack 
	#     datasetName : str
	#         name of the new .pickle file
	#     start_frame : int, optional
	#         if known, the timestamp in ns of the frame you wish to start from 
	#     end_frame : int, optional
	#         if known, the timestamp in ns of the frame you wish to finish at
	#     """
	# 	self.camera_topic = "/bebop/image_raw/compressed"
	# 	self.drone_topic = "/optitrack/bebop"
	# 	print("unpacking...")
	# 	#unpack the stamps
	# 	camera_stamps = self.ts.ExtractStampsFromHeader(self.camera_topic)
	# 	head_stamps = self.ts.ExtractStampsFromRosbag("/optitrack/head")
	# 	drone_stamps = self.ts.ExtractStampsFromRosbag(self.drone_topic)

	# 	if((len(drone_stamps) < len(camera_stamps)) or (len(head_stamps) < len(camera_stamps))):
	# 		print("Error:recording data corrupted. not enough MoCap stamps.") 
	# 		return

	# 	print("unpacked stamps")
		
	# 	#get the sync ids 
	# 	otherTopics = [head_stamps, drone_stamps]
	# 	sync_bebop_ids, sync_other_ids = self.ts.SyncStampsToMain(camera_stamps, otherTopics, delay)
	# 	sync_head_ids = sync_other_ids[0]
	# 	sync_drone_ids = sync_other_ids[1]	

	# 	print("synced ids")

	# 	head_msgs = self.ts.GetMessages("/optitrack/head")
	# 	drone_msgs = self.ts.GetMessages(self.drone_topic)
	# 	bridge = CvBridge()
		
	# 	x_dataset = []
	# 	head_dataset = []

	# 	#read in chunks because memory is low
	# 	bebop_msgs_count = self.ts.GetMessagesCount(self.camera_topic)
	# 	chunk_size = 1000
	# 	chunks = (bebop_msgs_count/chunk_size) + 1
	# 	for chunk in tqdm(range(chunks)):
	# 		bebop_msgs = self.ts.GetMessages(self.camera_topic, chunk * chunk_size + 1, (chunk+1) * chunk_size)

	# 		for i in tqdm(range(len(bebop_msgs))):
	# 			t = bebop_msgs[i].header.stamp.to_nsec()
	# 			if (t >= start) and (t <=end):
	# 				cv_image = bridge.compressed_imgmsg_to_cv2(bebop_msgs[i])
	# 				cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
	# 				cv_image = cv2.resize(cv_image, (config.input_width, config.input_height), cv2.INTER_AREA)
	# 				x_dataset.append(cv_image)		
		
	# 				head_id = sync_head_ids[chunk * chunk_size + i]		
	# 				drone_id = sync_drone_ids[chunk * chunk_size + i]
	# 				bebop_id = sync_bebop_ids[chunk * chunk_size + i]
	# 				#print("opti_id={}/{}, drone_id={}/{}, bebop_id={}".format(optitrack_id, len(optitrack_msgs), drone_id, len(drone_msgs), bebop_id))

	# 				x, y, z, yaw = relative_pose(head_msgs[head_id], drone_msgs[drone_id])
	# 				head_dataset.append([x, y, z, yaw])

	# 	print("dataset ready x:{} head:{}".format(len(x_dataset), len(head_dataset)))
	# 	df = pd.DataFrame(data={'x': x_dataset, 'y': head_dataset})
	# 	print("dataframe ready")
	# 	df.to_pickle(datasetName)


	def SaveToDataFrame(self, x_dataset, y_dataset, datasetName):

		if len(x_dataset) == len(y_dataset):
			print("dataset ready x:{} y:{}".format(len(x_dataset), len(y_dataset)))
			df = pd.DataFrame(data={'x': x_dataset, 'y': y_dataset})
		else:
			if len(y_dataset) == 2:
				print("dataset ready x:{} y:{} z:{}".format(len(x_dataset), len(y_dataset[0]), len(y_dataset[1])))
				df = pd.DataFrame(data={'x': x_dataset, 'y': y_dataset[0], 'z':y_dataset[1]})
			elif len(y_dataset) == 3:
				print("dataset ready x:{} y:{} z:{} w:{}".format(len(x_dataset), len(y_dataset[0]), len(y_dataset[1]), len(y_dataset[2])))
				df = pd.DataFrame(data={'x': x_dataset, 'y': y_dataset[0], 'z':y_dataset[1], 'w':y_dataset[2]})
			elif len(y_dataset) == 1:
				print("dataset ready x:{} y:{}".format(len(x_dataset), len(y_dataset[0])))
				df = pd.DataFrame(data={'x': x_dataset, 'y': y_dataset[0]})

			# add your own line if you have more tracking markers.....
			
		print("dataframe ready")
		df.to_pickle(datasetName)

	def SaveInfoFile(self, datasetName, topic_list):

		if datasetName.find(".pickle"):
			datasetName = datasetName.replace(".pickle",'')
		f= open(datasetName + ".txt","w+")

		for topic in topic_list:
			 f.write(topic + os.linesep)

		f.close()


	def CreateBebopDataset(self, delay, datasetName, camera_topic, drone_topic, tracking_topic_list, start = None, end = None, pose = None, outputs = None, throttle = 1):
		"""Converts rosbag to format suitable for training/testing. 
		if start_frame, end_frame are unknown, FrameSelector will help you choose how to trim the video
		the output.
		Additionally to .pickle, this function also creates a .txt file with the topics (in the correct order) contained in the dataset

	    Parameters
	    ----------
	    delay : int
	        The delay between the himax camera and the optitrack or bebop camera
	    datasetName : str
	        name of the new .pickle file
	    camera_topic : str
	        name of the camera (video) topic
	    drone_topic : str
	        name of the optitrack msg of the dron's pose
	    tracking_topic_list : list
	        list of names, specifying all the tracked marker topics (hand, head, etc)
	    start_frame : int, optional
	        if known, the timestamp in ns of the frame you wish to start from 
	    end_frame : int, optional
	        if known, the timestamp in ns of the frame you wish to finish at
	    """

		self.camera_topic = camera_topic
		self.other_topic_list = tracking_topic_list
		self.drone_topic = drone_topic


		if (self.camera_topic.find("compressed")) > -1:
			isCompressed = True
		else:
			isCompressed = False


		if (start is None) or (end is None):
			start, end  = self.FrameSelector(isCompressed)

		print("unpacking...")
		#unpack the stamps
		camera_stamps = self.ts.ExtractStampsFromHeader(self.camera_topic)
		drone_stamps = self.ts.ExtractStampsFromRosbag(self.drone_topic)

		other_stamps_list = []
		for topic in tracking_topic_list:
			topic_stamps = self.ts.ExtractStampsFromRosbag(topic)
			other_stamps_list.append(topic_stamps)
			if (len(topic_stamps) < len(camera_stamps)):
				print("Error:recording data corrupted. not enough MoCap stamps.") 
				return
		print("unpacked stamps")

		other_stamps_list.append(drone_stamps)
		if outputs is not None:
			output_msgs = self.ts.GetMessages("/bebop/output")
		
		#get the sync ids 
		sync_camera_ids, sync_other_ids = self.ts.SyncStampsToMain(camera_stamps, other_stamps_list, delay)
		sync_drone_ids = sync_other_ids.pop()

		print("synced ids")

		drone_msgs = self.ts.GetMessages(self.drone_topic)
		other_messages = []
		for topic in tracking_topic_list:
			msgs = self.ts.GetMessages(topic)
			other_messages.append(msgs)

		bridge = CvBridge()
		
		x_dataset = []
		y_dataset = [None] * len(tracking_topic_list)
		for i in range(len(y_dataset)):
			y_dataset[i] = []
		z_dataset =  []
		t_dataset = []
		o_dataset = []

		#read in chunks because memory is low
		camera_msgs_count = self.ts.GetMessagesCount(self.camera_topic)
		chunk_size = 1000
		chunks = (camera_msgs_count/chunk_size) + 1
		for chunk in tqdm(range(chunks)):
			camera_msgs = self.ts.GetMessages(self.camera_topic, chunk * chunk_size + 1, (chunk+1) * chunk_size)

			for i in tqdm(range(0, len(camera_msgs), throttle)):
				t = camera_msgs[i].header.stamp.to_nsec()
				if (t >= start) and (t <=end):
					if isCompressed==True:
						cv_image = bridge.compressed_imgmsg_to_cv2(camera_msgs[i])
					else:
						cv_image = bridge.imgmsg_to_cv2(camera_msgs[i])
					cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
					cv_image = cv2.cvtColor(cv_image, cv2.COLOR_RGB2GRAY)
					cv_image = cv2.resize(cv_image, (config.input_width, config.input_height), cv2.INTER_AREA)
					x_dataset.append(cv_image)	
					camera_id = sync_camera_ids[chunk * chunk_size + i]
					drone_id = sync_drone_ids[chunk * chunk_size + i]

					for id in range(len(sync_other_ids)):
						topic_msgs = other_messages[id]
						topic_id = sync_other_ids[id][chunk * chunk_size + i]
						x, y, z, yaw = relative_pose(topic_msgs[topic_id], drone_msgs[drone_id])
						y_dataset[id].append([x, y, z, yaw])


					x, y, z, yaw = ExtractPoseFromMessage(drone_msgs[drone_id])
					z_dataset.append([x, y, z, yaw])	
					t_dataset.append(t)
					if outputs is not None:
						output = output_msgs[i].data
						x, y, z, yaw = output
						o_dataset.append([x, y, z, yaw])
		
					#print("opti_id={}/{}, drone_id={}/{}, bebop_id={}".format(optitrack_id, len(optitrack_msgs), drone_id, len(drone_msgs), bebop_id))

										
		df = pd.DataFrame(data={'x': x_dataset, 'y': y_dataset[0], 'z' : z_dataset, 't': t_dataset, 'o': o_dataset})
		print("dataframe ready, frames: {}".format(len(x_dataset)))
		df.to_pickle(datasetName)

		#self.SaveToDataFrame(x_dataset, y_dataset, datasetName)
		topic_list = []
		topic_list.append("video")
		topic_list = topic_list + tracking_topic_list
		self.SaveInfoFile(datasetName, topic_list)


	def CreateHimaxDataset(self, delay, datasetName, camera_topic_himax, drone_topic, tracking_topic_list, start = None, end = None, pose = False):

		"""Converts rosbag to format suitable for training/testing. 
		if start_frame, end_frame are unknown, FrameSelector will help you choose how to trim the video
		the output.
		Additionally to .pickle, this function also creates a .txt file with the topics (in the correct order) contained in the dataset

	    Parameters
	    ----------
	    delay : int
	        The delay between the himax camera and the optitrack or bebop camera
	    datasetName : str
	        name of the new .pickle file
	    camera_topic_himax : str
	        name of the himax (video) topic
	    drone_topic : str
	        name of the optitrack msg of the dron's pose
	    tracking_topic_list : list
	        list of names, specifying all the tracked marker topics (hand, head, etc)
	    start_frame : int, optional
	        if known, the timestamp in ns of the frame you wish to start from 
	    end_frame : int, optional
	        if known, the timestamp in ns of the frame you wish to finish at
	    """

		self.camera_topic = camera_topic_himax
		self.other_topic_list = tracking_topic_list
		self.drone_topic = drone_topic

		if (self.camera_topic.find("compressed")) > -1:
			isCompressed = True
		else:
			isCompressed = False

		if (start is None) or (end is None):
			start, end  = self.FrameSelector(isCompressed)


		print("unpacking...")
		#unpack the stamps
		camera_stamps = self.ts.ExtractStampsFromHeader(self.camera_topic)
		drone_stamps = self.ts.ExtractStampsFromRosbag(self.drone_topic)

		other_stamps_list = []
		for topic in tracking_topic_list:
			topic_stamps = self.ts.ExtractStampsFromRosbag(topic)
			other_stamps_list.append(topic_stamps)
			if (len(topic_stamps) < len(camera_stamps)):
				print("Error:recording data corrupted. not enough MoCap stamps.") 
				return

		other_stamps_list.append(drone_stamps)

		drone_msgs = self.ts.GetMessages(self.drone_topic)
		other_messages = []
		for topic in tracking_topic_list:
			msgs = self.ts.GetMessages(topic)
			other_messages.append(msgs)

		bridge = CvBridge()

		x_dataset = []
		y_dataset = [None] * len(tracking_topic_list)
		z_dataset = []
		t_dataset = []
		for i in range(len(y_dataset)):
			y_dataset[i] = []

		self.camera_topic = camera_topic_himax

		#unpack the stamps
		camera_stamps = self.ts.ExtractStampsFromHeader(self.camera_topic)
		print("unpacked stamps")
		print("Getting camera messages")

		himax_msgs = self.ts.GetMessages(self.camera_topic)
		print("Synching IDs")
		sync_camera_ids, sync_other_ids = self.ts.SyncStampsToMain(camera_stamps, other_stamps_list, delay)
		sync_drone_ids = sync_other_ids.pop()

		print("Preparing DataFrame")
		for i in range(len(himax_msgs)):
			t = himax_msgs[i].header.stamp.to_nsec()
			if (t >= start) and (t <=end):

				cv_image = bridge.imgmsg_to_cv2(himax_msgs[i])
				#need to crop too?
				#cv_image = cv2.resize(cv_image, (config.input_width, config.input_height), cv2.INTER_AREA)
				x_dataset.append(cv_image)	
				himax_id = sync_camera_ids[i]
				drone_id = sync_drone_ids[i]

				for id in range(len(sync_other_ids)):
					topic_msgs = other_messages[id]
					topic_id = sync_other_ids[id][i]
					x, y, z, yaw = relative_pose(topic_msgs[topic_id], drone_msgs[drone_id])
					y_dataset[id].append([x, y, z, yaw]) 
				
				x, y, z, yaw = ExtractPoseFromMessage(drone_msgs[drone_id])
				z_dataset.append([x, y, z, yaw])	
				t_dataset.append(t)
			
		
		if pose is False:								
			self.SaveToDataFrame(x_dataset, y_dataset, datasetName)
		else:
			df = pd.DataFrame(data={'x': x_dataset, 'y': y_dataset[0], 'z' : z_dataset, 't': t_dataset})
			#df = pd.DataFrame(data={'x': x_dataset, 'y': y_dataset[0], 'z' : z_dataset})
			print("dataframe ready, frames: {}".format(len(x_dataset)))
			df.to_pickle(datasetName)

		topic_list = []
		topic_list.append("video")
		topic_list = topic_list + tracking_topic_list
		self.SaveInfoFile(datasetName, topic_list)


	def CreateMixedDataset(self, delay, datasetName, camera_topic_himax, camera_topic_bebop, drone_topic, tracking_topic_list, start = None, end = None):

		"""Converts rosbag to format suitable for training/testing. 
		if start_frame, end_frame are unknown, FrameSelector will help you choose how to trim the video
		the output.
		Additionally to .pickle, this function also creates a .txt file with the topics (in the correct order) contained in the dataset

	    Parameters
	    ----------
	    delay : int
	        The delay between the himax camera and the optitrack or bebop camera
	    datasetName : str
	        name of the new .pickle file
	    camera_topic_himax : str
	        name of the himax (video) topic
	    camera_topic_bebop : str
	        name of the bebop (video) topic
	    drone_topic : str
	        name of the optitrack msg of the dron's pose
	    tracking_topic_list : list
	        list of names, specifying all the tracked marker topics (hand, head, etc)
	    start_frame : int, optional
	        if known, the timestamp in ns of the frame you wish to start from 
	    end_frame : int, optional
	        if known, the timestamp in ns of the frame you wish to finish at
	    """

		self.camera_topic = camera_topic_bebop
		self.other_topic_list = tracking_topic_list
		self.drone_topic = drone_topic


		if (self.camera_topic.find("compressed")) > -1:
			isCompressed = True
		else:
			isCompressed = False

		if (start is None) or (end is None):
			start, end  = self.FrameSelector(isCompressed)

		print("unpacking...")
		#unpack the stamps
		camera_stamps = self.ts.ExtractStampsFromHeader(self.camera_topic)
		drone_stamps = self.ts.ExtractStampsFromRosbag(self.drone_topic)

		other_stamps_list = []
		for topic in tracking_topic_list:
			topic_stamps = self.ts.ExtractStampsFromRosbag(topic)
			other_stamps_list.append(topic_stamps)
			if (len(topic_stamps) < len(camera_stamps)):
				print("Error:recording data corrupted. not enough MoCap stamps.") 
				return
		print("unpacked stamps")

		other_stamps_list.append(drone_stamps)
		
		#get the sync ids 
		sync_camera_ids, sync_other_ids = self.ts.SyncStampsToMain(camera_stamps, other_stamps_list, 0)
		sync_drone_ids = sync_other_ids.pop()

		print("synced ids")

		drone_msgs = self.ts.GetMessages(self.drone_topic)
		other_messages = []
		for topic in tracking_topic_list:
			msgs = self.ts.GetMessages(topic)
			other_messages.append(msgs)

		bridge = CvBridge()

		gammaLUT = ImageEffects.GetGammaLUT(0.6)
		vignetteMask = ImageEffects.GetVignetteMask(config.himax_width, config.himax_width)
		
		x_dataset = []
		y_dataset = [None] * len(tracking_topic_list)
		for i in range(len(y_dataset)):
			y_dataset[i] = []

		#read in chunks because memory is low
		camera_msgs_count = self.ts.GetMessagesCount(self.camera_topic)
		chunk_size = 1000
		chunks = (camera_msgs_count/chunk_size) + 1
		for chunk in tqdm(range(chunks)):
			camera_msgs = self.ts.GetMessages(self.camera_topic, chunk * chunk_size + 1, (chunk+1) * chunk_size)

			for i in tqdm(range(len(camera_msgs))):
				t = camera_msgs[i].header.stamp.to_nsec()
				if (t >= start) and (t <=end):
					if isCompressed==True:
						cv_image = bridge.compressed_imgmsg_to_cv2(camera_msgs[i])
					else:
						cv_image = bridge.imgmsg_to_cv2(camera_msgs[i])
					# image transform
					cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
					cv_image = cv2.LUT(cv_image, gammaLUT)
					cv_image = cv2.GaussianBlur(cv_image,(5,5),0)
					cv_image = cv2.resize(cv_image, (config.himax_width, config.himax_height), cv2.INTER_AREA)
					cv_image = cv_image *  vignetteMask[40:284, 0:324]
					cv_image = cv2.resize(cv_image, (config.input_width, config.input_height), cv2.INTER_NEAREST)
					x_dataset.append(cv_image)	

					camera_id = sync_camera_ids[chunk * chunk_size + i]
					drone_id = sync_drone_ids[chunk * chunk_size + i]

					for id in range(len(sync_other_ids)):
						topic_msgs = other_messages[id]
						topic_id = sync_other_ids[id][chunk * chunk_size + i]
						x, y, z, yaw = relative_pose(topic_msgs[topic_id], drone_msgs[drone_id])
						y_dataset[id].append([x, y, z, yaw])
		
					#print("opti_id={}/{}, drone_id={}/{}, bebop_id={}".format(optitrack_id, len(optitrack_msgs), drone_id, len(drone_msgs), bebop_id))

		print("finished transformed bebop")
		self.camera_topic = camera_topic_himax

		print("unpacking...")
		#unpack the stamps
		camera_stamps = self.ts.ExtractStampsFromHeader(self.camera_topic)
		himax_msgs = self.ts.GetMessages(self.camera_topic)
		sync_camera_ids, sync_other_ids = self.ts.SyncStampsToMain(camera_stamps, other_stamps_list, delay)
		sync_drone_ids = sync_other_ids.pop()

		for i in range(len(himax_msgs)):
			cv_image = bridge.imgmsg_to_cv2(himax_msgs[i])
			#need to crop too?
			#cv_image = cv2.resize(cv_image, (config.input_width, config.input_height), cv2.INTER_AREA)
			x_dataset.append(cv_image)	
			himax_id = sync_camera_ids[i]
			drone_id = sync_drone_ids[i]

			for id in range(len(sync_other_ids)):
				topic_msgs = other_messages[id]
				topic_id = sync_other_ids[id][i]
				x, y, z, yaw = relative_pose(topic_msgs[topic_id], drone_msgs[drone_id])
				y_dataset[id].append([x, y, z, yaw])			
										
		self.SaveToDataFrame(x_dataset, y_dataset, datasetName)
		topic_list = []
		topic_list.append("video")
		topic_list = topic_list + tracking_topic_list
		self.SaveInfoFile(datasetName, topic_list)




	@staticmethod
	def JoinPickleFiles(fileList, datasetName, folderPath=""):
		"""joins several pickle files into one big .pickle file

	    Parameters
	    ----------
	    fileList : list
	        list of the file locations of all the .pickle
	    datasetName : str
	        name of the new .pickle file
	    isBebop : bool, optional
	        True if you want RGB dataset for the bebop, False if you want Himax-tailored dataset
	    folderPath : str, optional
	        if all files are in the same folder, you can only specify the folder path once
	    """
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
		# print("dataset ready x:{} head:{}".format(len(x_dataset), len(y_dataset)))
		# df = pd.DataFrame(data={'x': x_dataset, 'y': y_dataset})
		print("dataframe ready")
		df.to_pickle(datasetName)




		
		

		
		
	



