import cv2
import numpy as np
import os


class ImageTransformer:

	def CalculateFOVfromCalibration(self, calibFile):

		cv_file = cv2.FileStorage(calibFile, cv2.FILE_STORAGE_READ)
		k = cv_file.getNode("k").mat()
		D = cv_file.getNode("D").mat()
		size = cv_file.getNode("size").mat()
		h = size[0][0]
		w = size[1][0]
		cv_file.release()
		fovx, fovy, focalLength, principalPoint, aspectRatio = cv2.calibrationMatrixValues(k, (w, h), 1, 1) 
		print("{} : fovx {} fovy {}".format(calibFile, fovx, fovy))
		return fovx, fovy, h, w

	

	def get_crop_parameters(self, x_ratio, y_ratio, h, w):

		new_size = (int(w * x_ratio) , int(h * y_ratio))
		#print ("new size {}".format(new_size))
		shift_x = 0.5*(w - new_size[0])
		shift_y = 0.5*(h - new_size[1])

		return new_size, shift_x, shift_y


	def TransformImages(self, himaxCalibFile, bebopCalibFile, himaxImages, bebopImages):

		himax_fovx, himax_fovy, himax_h, himax_w = self.CalculateFOVfromCalibration(himaxCalibFile)
		bebop_fovx, bebop_fovy, bebop_h, bebop_w = self.CalculateFOVfromCalibration(bebopCalibFile)
		x_ratio = bebop_fovx / himax_fovx
		y_ratio = bebop_fovy / himax_fovy

		new_size, shift_x, shift_y = self.get_crop_parameters(x_ratio, y_ratio, himax_h, himax_w)

		himaxTransImages = []
		bebopTransImages = []

		for img in himaxImages:
			crop_img = img[shift_y:shift_y+new_size[1], shift_x:shift_x+ new_size[0]]	
			himaxTransImages.append(crop_img)

		for img in bebopImages:
			gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
			bebopTransImages.append(gray)

		return himaxTransImages, bebopTransImages
