#!/usr/bin/env python

import rospy
import numpy as np
from sensor_msgs.msg import Image
from sensor_msgs.msg import CameraInfo
from cv_bridge import CvBridge
import os
import ctypes as c
from time import time
import cv2

height = 244
width = 324
page_size = 4096


def read_from_pipe(pipein):
	remaining_size = height * width
	
	data = []
	while(remaining_size >= page_size):
		output =  os.read(pipein, page_size)
		remaining_size = remaining_size - len(output)
		data.append(output)

	data.append(os.read(pipein, max(0, remaining_size)))
	data=''.join(data)

	if (len(data) < height*width):
			rospy.loginfo("Error, expecting {} bytes, received {}.".format(height*width, len(data)))
			return None
	#else:
		#rospy.loginfo("Received buffer.")

	data = np.frombuffer(data, dtype=np.uint8)

	return data
	

def main():
	rospy.init_node('pub_gap_camera', anonymous=True)
	image_pub = rospy.Publisher("image_topic_2",Image)
	bridge = CvBridge()

	cv_file = cv2.FileStorage("calibration.yaml", cv2.FILE_STORAGE_READ)
	k = cv_file.getNode("k").mat()
	D = cv_file.getNode("D").mat()
	size = cv_file.getNode("size").mat()
	cv_file.release()

	k = k.flatten().tolist()
	D = D.flatten().tolist()
	camera_info = CameraInfo()
	camera_info.K = k
	camera_info.D = D
	camera_info.height = size[0][0]
	camera_info.width = size[1][0]
	print(camera_info)
	
	pipe_name = "image_pipe"
	if not os.path.exists(pipe_name):
		os.mkfifo(pipe_name)

	pipein = os.open(pipe_name, os.O_RDONLY)

	frame_id = 1
	while not rospy.is_shutdown():
		data = read_from_pipe(pipein)
		if data is not None:
			cv_image = np.reshape(data, (height, width))
			image_pub.publish(bridge.cv2_to_imgmsg(cv_image))
			rospy.sleep(0)
			
	os.close(pipein)

if __name__ == '__main__':
    main()
