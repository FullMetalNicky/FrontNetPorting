#!/usr/bin/env python

import rospy
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import os
import ctypes as c
#from libc.stdint import uintptr_t

height = 244
width = 324
page_size = 4096

def read_from_pipe(pipein):
	remaining_size = height * width

	data = ''
	while(remaining_size >= page_size):
		output =  os.read(pipein, page_size)
		remaining_size = remaining_size - len(output)
		data = data + output

	data = data + os.read(pipein, max(0, remaining_size))
	if (len(data) < height*width):
			print("Error, expecting {} bytes, received {}.".format(height*width, len(data)))
			return None
	#print(len(data))
	data = np.frombuffer(data, dtype=np.uint8)
	#print(data.shape)

	return data
	

def main():
	rospy.init_node('pub_gap_camera', anonymous=True)
	image_pub = rospy.Publisher("image_topic_2",Image)
	bridge = CvBridge()

	pipe_name = "image_pipe"
	if not os.path.exists(pipe_name):
		os.mkfifo(pipe_name)

	pipein = os.open(pipe_name, os.O_RDONLY)

	while not rospy.is_shutdown():
		#data = os.read(pipein, height * width)
		#print(len(data))
		#data = np.frombuffer(data, dtype=np.uint8)
		#print(data.shape)
		#data_ptr = c.cast(data, c.POINTER(c.c_ubyte))
		#cv_image = np.ctypeslib.as_array(data_ptr, shape=(height, width))
		#image_pub.publish(bridge.cv2_to_imgmsg(cv_image))
		data = read_from_pipe(pipein)
		if data is not None:
			print(data.shape)
			cv_image = np.reshape(data, (height, width))
			image_pub.publish(bridge.cv2_to_imgmsg(cv_image))

	os.close(pipein)

if __name__ == '__main__':
    main()
