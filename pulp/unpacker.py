import rospy
import rosbag

from sensor_msgs.msg import Image
from sensor_msgs.msg import CameraInfo
from cv_bridge import CvBridge
import cv2


def main():
	rospy.init_node('unbag', anonymous=True)
	himax_folder = "../data/himax/"
	bebop_folder = "../data/bebop/"
	himax_cnt=1
	bebop_cnt=1
	bridge = CvBridge()

	bag = rosbag.Bag('../data/2019-08-08-02-34-35.bag')
	for topic, msg, t in bag.read_messages(topics=['himax_camera', 'bebop/image_raw']):
		#print("topic {}, time stamp {}".format(topic, t))
		if(topic == 'himax_camera'):
			cv_image = bridge.imgmsg_to_cv2(msg)
			image_name = himax_folder + "{}.jpg".format(himax_cnt)
			himax_cnt = himax_cnt + 1
			cv2.imwrite(image_name, cv_image)
		elif(topic == 'bebop/image_raw'):
			cv_image = bridge.imgmsg_to_cv2(msg)
			cv_image = cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR)
			image_name = bebop_folder + "{}.jpg".format(bebop_cnt)
			bebop_cnt = bebop_cnt + 1
			cv2.imwrite(image_name, cv_image)
		if (himax_cnt == 9) and  (topic == 'himax_camera'):
			print("topic {}, time stamp {}".format(topic, t))
		if (bebop_cnt == 121) and (topic == 'bebop/image_raw') :
			print("topic {}, time stamp {}".format(topic, t))
		if himax_cnt > 10:
			break
		
	bag.close()



if __name__ == '__main__':
    main()
