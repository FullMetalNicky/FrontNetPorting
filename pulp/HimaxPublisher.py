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
import fcntl
import sys

sys.path.append("/home/usi/Documents/Drone/FrontNetPorting")
import config
sys.path.append("/home/usi/Documents/Drone/FrontNetPorting/DataProcessing")
from ImageTransformer import ImageTransformer


page_size = 4096
F_GETPIPE_SZ = 1032  # Linux 2.6.35+
F_SETPIPE_SZ = 1031  # Linux 2.6.35+

def read_from_pipe(pipein):
	remaining_size = config.himax_height * config.himax_width
	
	data = []
	while(remaining_size > 0):
		output =  os.read(pipein, min(remaining_size, page_size))
		remaining_size = remaining_size - len(output)
		data.append(output)

	data=''.join(data)

	if (len(data) < config.himax_height*config.himax_width):
			rospy.loginfo("Error, expecting {} bytes, received {}.".format(config.himax_height*config.himax_width, len(data)))
			return None

	data = np.frombuffer(data, dtype=np.uint8)

	return data



def main():
	rospy.init_node('pub_gap_camera', anonymous=True)
	image_pub = rospy.Publisher("himax_camera",Image)
	bridge = CvBridge()
	
	cv_file = cv2.FileStorage(config.folder_path +"/data/calibration.yaml", cv2.FILE_STORAGE_READ)
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

	it = ImageTransformer()
	new_size, shift_x, shift_y = it.get_crop_parameters(config.folder_path + "/data/calibration.yaml", config.folder_path + "/data/bebop_calibration.yaml")
	shift_x = int(shift_x)
	shift_y = int(shift_y)

	
	pipe_name = config.folder_path + "/pulp/image_pipe"
	if not os.path.exists(pipe_name):
		os.mkfifo(pipe_name)
		
	pipein = os.open(pipe_name, os.O_RDONLY)
	fcntl.fcntl(pipein, F_SETPIPE_SZ, 1000000)

	frame_id = 1
	while not rospy.is_shutdown():
		data = read_from_pipe(pipein)
		if data is not None:
			cv_image = np.reshape(data, (config.himax_height, config.himax_width))
			cv_image = cv_image[shift_y:shift_y+new_size[1], shift_x:shift_x+ new_size[0]]	
			msg = bridge.cv2_to_imgmsg(cv_image)
			msg.header.stamp = rospy.Time.now()
			image_pub.publish(msg)
			rospy.sleep(0)
			
	os.close(pipein)

if __name__ == '__main__':
    main()
