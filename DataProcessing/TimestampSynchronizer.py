import rospy
import rosbag
import numpy as np


class TimestampSynchronizer:

	def __init__(self, bagName):
		self.node = rospy.init_node('sync', anonymous=True)
		self.bagName = bagName

	def closest(self, list, Number):
	    aux = []
	    for valor in list:
		aux.append(abs(Number-valor.to_nsec()))

	    return aux.index(min(aux))

	def SyncStamps(self, topic1_stamps, topic2_stamps, delay):
		
		sync_topic1_ids = [] 
		sync_topic2_ids = []

		for i, topic1_t in enumerate(topic1_stamps):
			t = topic1_t.to_nsec() + delay
			ind = self.closest(topic2_stamps, t)
			sync_topic1_ids.append(i)
			sync_topic2_ids.append(ind)

		return sync_topic1_ids, sync_topic2_ids

	def SyncStampsToMain(self, main_topic_stamps, topics_stamps_list, delay):
		
		sync_main_topic_ids = [] 
		sync_other_topics_ids = []
		for i in range(len(topics_stamps_list)):
			sync_other_topics_ids.append(list())	

		for i, topic1_t in enumerate(main_topic_stamps):
			t = topic1_t.to_nsec() + delay
			ind_vec = []
			for topic in range(len(topics_stamps_list)):
				ind = self.closest(topics_stamps_list[topic], t)
				sync_other_topics_ids[topic].append(ind)
			
			sync_main_topic_ids.append(i)

		return sync_main_topic_ids, sync_other_topics_ids


	def UnpackBagStamps(self, topic1 = "himax_camera", topic2= "bebop/image_raw", stopNum=np.inf):
	
		bag = rosbag.Bag(self.bagName)
		topic1_stamps = []
		topic2_stamps = []
		topic1_cnt = 1

		for topic, msg, t in bag.read_messages(topics=[topic1, topic2]):

			if(topic == topic1):
				topic1_stamps.append(msg.header.stamp)
				topic1_cnt = topic1_cnt + 1

			elif(topic == topic2):
				topic2_stamps.append(msg.header.stamp)

			if topic1_cnt > stopNum:
				break
	
		bag.close()

		return topic1_stamps, topic2_stamps

	def ExtractStampsFromHeader(self, topic1 = "himax_camera", stopNum=np.inf):
	
		bag = rosbag.Bag(self.bagName)
		topic1_stamps = []
		topic1_cnt = 1

		for topic, msg, t in bag.read_messages(topics=[topic1]):

			if(topic == topic1):
				topic1_stamps.append(msg.header.stamp)
				topic1_cnt = topic1_cnt + 1

			if topic1_cnt > stopNum:
				break
	
		bag.close()

		return topic1_stamps


	def ExtractStampsFromRosbag(self, topic1 = "himax_camera", stopNum=np.inf):
	
		bag = rosbag.Bag(self.bagName)
		topic1_stamps = []
		topic1_cnt = 1

		for topic, msg, t in bag.read_messages(topics=[topic1]):

			if(topic == topic1):
				topic1_stamps.append(t)
				topic1_cnt = topic1_cnt + 1

			if topic1_cnt > stopNum:
				break

		bag.close()

		return topic1_stamps


	def SyncTopicsByStamps(self, topic1, topic2, sync_topic1_ids, sync_topic2_ids):
		bag = rosbag.Bag(self.bagName)
		topic1_msgs = []
		topic2_msgs = []
		topic1_cnt = 0
		topic2_cnt = 0

		for topic, msg, t in bag.read_messages(topics=[topic1, topic2]):
			if((topic == topic1) and (len(sync_topic1_ids)>0)):
				if (topic1_cnt == sync_topic1_ids[0]):
					topic1_msgs.append(msg)
					sync_topic1_ids.pop(0)
				topic1_cnt = topic1_cnt + 1

			elif((topic == topic2) and (len(sync_topic2_ids)>0)):
		
				if (topic2_cnt == sync_topic2_ids[0]):
					topic2_msgs.append(msg)
					sync_topic2_ids.pop(0)
				topic2_cnt = topic2_cnt+ 1

			if ((len(sync_topic1_ids)==0) and (len(sync_topic2_ids)==0)):
				break
		
		bag.close()

		return topic1_msgs, topic2_msgs





	def GetMessages(self, topic1, startNum = 1, stopNum=np.inf):
		bag = rosbag.Bag(self.bagName)
		topic_msgs = []
		topic_cnt = 1

		for topic, msg, t in bag.read_messages(topics=[topic1]):
			if(topic == topic1):
				if topic_cnt >= startNum:
					topic_msgs.append(msg)
				topic_cnt = topic_cnt + 1

			if topic_cnt > stopNum:
				break
	
		bag.close()

		return topic_msgs

	def GetMessagesCount(self, topic1):
		bag = rosbag.Bag(self.bagName)
		topic_cnt = 0

		for topic, msg, t in bag.read_messages(topics=[topic1]):
			if(topic == topic1):
				topic_cnt = topic_cnt + 1
	
		bag.close()

		return topic_cnt














