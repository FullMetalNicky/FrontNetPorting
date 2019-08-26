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


	def SyncImages(self, himax_images, bebop_images, himax_stamps, bebop_stamps, delay):
		
		sync_himax_images= [] 
		sync_bebop_images = []

		for i, himax_t in enumerate(himax_stamps):
			t = himax_t.to_nsec() + delay
			ind = self.closest(bebop_stamps, t)
			sync_himax_images.append(himax_images[i])
			sync_bebop_images.append(bebop_images[ind])

		return sync_himax_images, sync_bebop_images

	def SyncImagesByStamps(self, sync_himax_ids, sync_bebop_ids):

		bag = rosbag.Bag(self.bagName)
		bridge = CvBridge()
		himax_images = []
		bebop_images = []
		himax_cnt = 0
		bebop_cnt = 0

		for topic, msg, t in bag.read_messages(topics=['himax_camera', 'bebop/image_raw']):
			if((topic == 'himax_camera') and (len(sync_himax_ids)>0)):
				if (himax_cnt == sync_himax_ids[0]):
					cv_image = bridge.imgmsg_to_cv2(msg)
					himax_images.append(cv_image)
					sync_himax_ids.pop(0)
				himax_cnt = himax_cnt + 1

			elif((topic == 'bebop/image_raw') and (len(sync_bebop_ids)>0)):
				if (bebop_cnt == sync_bebop_ids[0]):
					cv_image = bridge.imgmsg_to_cv2(msg)
					cv_image = cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR)
					bebop_images.append(cv_image)
					sync_bebop_ids.pop(0)
				bebop_cnt = bebop_cnt+ 1

			if ((len(sync_himax_ids)==0) and (len(sync_bebop_ids)==0)):
				break
		
		bag.close()

		return himax_images, bebop_images


	def CreateSyncVideo(self, sync_himax_images, sync_bebop_images, videoName, fps=1):

		if(len(sync_himax_images) != len(sync_bebop_images)):
			print("Error, images not in the same length")

		frames =[]
		for i in range(len(sync_himax_images)):
			viz = ImageEffects.ConcatImages(sync_himax_images[i], sync_bebop_images[i])
			frames.append(viz)

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
			


