import rospy
import rosbag

from sensor_msgs.msg import Image
from sensor_msgs.msg import CameraInfo
from cv_bridge import CvBridge
import cv2
import numpy as np

def closest(list, Number):
    aux = []
    for valor in list:
        aux.append(abs(Number-valor.to_nsec()))

    return aux.index(min(aux))

def concat_images(img1, img2):

	if len(img1.shape) < 3:
		img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2RGB)
	if len(img2.shape) < 3:
		img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2RGB)
	h1, w1 = img1.shape[:2]
	h2, w2 = img2.shape[:2]
	
	vis = np.zeros((max(h1, h2), w1+w2,3), np.uint8)

	vis[:h1, :w1,:3] = img1
	vis[:h2, w1:w1+w2,:3] = img2
	return vis
	

def main():
	rospy.init_node('unbag', anonymous=True)
	folder_path = "../"
	himax_cnt = 1
	bridge = CvBridge()
	#delay = (-1795935273 -1637871369 -1874250084 -1817123289-1817327532) /10
	delay = -1817123289
	print(delay)
	
	himax_images = []
	himax_stamps = []
	bebop_images = []
	bebop_stamps = []

	bag = rosbag.Bag('../data/2019-08-08-05-06-09.bag')
	for topic, msg, t in bag.read_messages(topics=['himax_camera', 'bebop/image_raw']):
	
		cv_image = bridge.imgmsg_to_cv2(msg)
		if(topic == 'himax_camera'):
			himax_images.append(cv_image)
			himax_stamps.append(msg.header.stamp)
			himax_cnt = himax_cnt + 1

		elif(topic == 'bebop/image_raw'):
			cv_image = cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR)
			bebop_images.append(cv_image)
			bebop_stamps.append(msg.header.stamp)

		if himax_cnt > 50:
			break
		
	bag.close()
	frames =[]
	for i, himax_t in enumerate(himax_stamps):
		t = himax_t.to_nsec() + delay
		ind = closest(bebop_stamps, t)
		print("himax ind {}, bebop ind {}".format(i, ind))
		viz = concat_images(himax_images[i], bebop_images[ind])
		cv2.imshow("compare", viz)
		cv2.waitKey(0)
		frames.append(viz)

	height, width, layers = frames[0].shape
    	size = (width,height)
	print(size)
	out = cv2.VideoWriter('sync.avi',cv2.VideoWriter_fourcc(*'DIVX'), 1, size)
 
	for i in range(len(frames)):
	    out.write(frames[i])
	out.release()



if __name__ == '__main__':
    main()
