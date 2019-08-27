import rospy
import numpy as np
from sensor_msgs.msg import Image
from sensor_msgs.msg import CameraInfo
from cv_bridge import CvBridge
import cv2
import sys
sys.path.append("/home/usi/Documents/Drone/FrontNetPorting/DataProcessing")
from ImageEffects import ImageEffects



class Visualizaer:

	def __init__(self):
		rospy.init_node('pub_concat_cameras', anonymous=True)
		self.image_pub = rospy.Publisher("concat_cameras",Image)
		self.himax_sub = rospy.Subscriber("/himax_camera", Image, self.himax_cb)
		self.bebop_sub = rospy.Subscriber("/bebop/image_raw", Image, self.bebop_cb)
		self.bridge = CvBridge()
		self.himax_image = None
		self.bebop_image = None

	def himax_cb(self, data):
		self.himax_image = self.bridge.imgmsg_to_cv2(data)

	def bebop_cb(self, data):
		self.bebop_image = self.bridge.imgmsg_to_cv2(data)
		


	def Run(self):
		while not rospy.is_shutdown():

			if ((self.himax_image is not None) and (self.bebop_image is not None)):
				size = self.himax_image.shape
				self.bebop_image = cv2.resize(self.bebop_image, (size[1], size[0]))
				viz = ImageEffects.ConcatImages(self.himax_image, self.bebop_image)
				msg = self.bridge.cv2_to_imgmsg(viz, "rgb8")
				msg.header.stamp = rospy.Time.now()
				self.image_pub.publish(msg)
				rospy.sleep(0)
				


def main():
	viz = Visualizaer()
	viz.Run()



if __name__ == '__main__':
    main()
