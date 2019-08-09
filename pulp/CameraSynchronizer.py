import rospy
from sensor_msgs.msg import Image
import cv2
import numpy as np


class CameraSynchronizer:

	def closest(self, list, Number):
	    aux = []
	    for valor in list:
		aux.append(abs(Number-valor.to_nsec()))

	    return aux.index(min(aux))

	def concat_images(self, img1, img2):

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


	def SyncImages(self, himax_images, bebop_images, himax_stamps, bebop_stamps, delay):
		
		sync_himax_images= [] 
		sync_bebop_images = []

		for i, himax_t in enumerate(himax_stamps):
			t = himax_t.to_nsec() + delay
			ind = self.closest(bebop_stamps, t)
			sync_himax_images.append(himax_images[i])
			sync_bebop_images.append(bebop_images[ind])

		return sync_himax_images, sync_bebop_images

	def CreateSyncVideo(self, sync_himax_images, sync_bebop_images, videoName, fps=1):

		if(len(sync_himax_images) != len(sync_bebop_images)):
			print("Error, images not in the same length")

		frames =[]
		for i in range(len(sync_himax_images)):
			viz = self.concat_images(sync_himax_images[i], sync_bebop_images[i])
			frames.append(viz)

		height, width, layers = frames[0].shape
  		size = (width,height)
		out = cv2.VideoWriter(videoName, cv2.VideoWriter_fourcc(*'DIVX'), fps, size)
	 
		for i in range(len(frames)):
			  out.write(frames[i])
		out.release()
			





