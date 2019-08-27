import rospy
import rosbag
from sensor_msgs.msg import Image
import cv2
import numpy as np
from cv_bridge import CvBridge
from ImageEffects import ImageEffects

class CameraSynchronizer:

	def __init__(self, bagName):
		self.node = rospy.init_node('sync', anonymous=True)
		self.bagName = bagName

	def ConvertMsgstoImages(self, himax_msgs, bebop_msgs):
		himax_images = []
		bebop_images = []
		bridge = CvBridge()
		for i in range(len(himax_msgs)):
			himax_image = bridge.imgmsg_to_cv2(himax_msgs[i])
			himax_images.append(himax_image)
			bebop_image = bridge.imgmsg_to_cv2(bebop_msgs[i])
			bebop_image = cv2.cvtColor(bebop_image, cv2.COLOR_RGB2BGR)
			bebop_images.append(bebop_image)

		return himax_images, bebop_images


	def CreateSyncVideo(self, frames, videoName, fps=1):
		
		height, width, layers = frames[0].shape
		size = (width,height)
		out = cv2.VideoWriter(videoName, cv2.VideoWriter_fourcc(*'DIVX'), fps, size)

		for i in range(len(frames)):
				out.write(frames[i])
		out.release()

	def GetSyncConcatFrames(self, sync_himax_images, sync_bebop_images):

		if(len(sync_himax_images) != len(sync_bebop_images)):
			print("Error, images not in the same length")

		frames =[]
		for i in range(len(sync_himax_images)):
			viz = ImageEffects.ConcatImages(sync_himax_images[i], sync_bebop_images[i])
			frames.append(viz)
		
		return frames
			


