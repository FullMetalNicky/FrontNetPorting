import rospy
import rosbag

from sensor_msgs.msg import Image
from sensor_msgs.msg import CameraInfo
from cv_bridge import CvBridge
import cv2
import numpy as np


class RosbagUnpacker:

	def init(self):
		self.node = rospy.init_node('unbag', anonymous=True)

	def UnpackBag(self, bagName, stopNum=np.inf):

		bag = rosbag.Bag(bagName)
		himax_msgs = []
		bebop_msgs = []
		himax_cnt = 1

		for topic, msg, t in bag.read_messages(topics=['himax_camera', 'bebop/image_raw']):
	
			if(topic == 'himax_camera'):
				himax_msgs.append(msg)
				himax_cnt = himax_cnt + 1

			elif(topic == 'bebop/image_raw'):
				bebop_msgs.append(msg)
	
			if himax_cnt > stopNum:
				break
		
		bag.close()

		return himax_msgs, bebop_msgs

	def MessagesToImages(self, himax_msgs, bepop_msgs):

		himax_images = []
		himax_stamps = []
		bebop_images = []
		bebop_stamps = []
		bridge = CvBridge()

		for msg in himax_msgs:
			cv_image = bridge.imgmsg_to_cv2(msg)
			himax_images.append(cv_image)
			himax_stamps.append(msg.header.stamp)

		for msg in bepop_msgs:
			cv_image = bridge.imgmsg_to_cv2(msg)
			cv_image = cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR)
			bebop_images.append(cv_image)
			bebop_stamps.append(msg.header.stamp)

		return himax_images, bebop_images, himax_stamps, bebop_stamps
		
