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
	bebop_stamp = None
	himax_stamp = None
	h_stop = 12
	b_stop = 612

	bag = rosbag.Bag('../data/2019-08-08-08-17-30.bag')
	for topic, msg, t in bag.read_messages(topics=['himax_camera', 'bebop/image_raw']):
		#print("topic {}, time stamp {}".format(topic, t))
	
		cv_image = bridge.imgmsg_to_cv2(msg)
		if(topic == 'himax_camera'):
			image_name = himax_folder + "{}.jpg".format(himax_cnt)
			if (himax_cnt == h_stop):
				print("topic {}, time stamp    {}".format(topic, msg.header.stamp))
				himax_stamp = msg.header.stamp
			himax_cnt = himax_cnt + 1

		elif(topic == 'bebop/image_raw'):
			cv_image = cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR)
			image_name = bebop_folder + "{}.jpg".format(bebop_cnt)
			if (bebop_cnt == b_stop):
				print("topic {}, time stamp {}".format(topic, msg.header.stamp))
				bebop_stamp = msg.header.stamp
			bebop_cnt = bebop_cnt + 1

		if((himax_stamp is not None) and (bebop_stamp is not None)):
			delay = long(bebop_stamp.to_nsec()) - long(himax_stamp.to_nsec())
			print("Himax frame {}, Bebop frame {}".format(h_stop, b_stop))
			print("delay {} ns".format(delay))
			break

		#if himax_cnt < 100:
		#	continue
		#cv2.imwrite(image_name, cv_image)
		
		if himax_cnt > 20:
			break
		
	bag.close()



if __name__ == '__main__':
    main()
