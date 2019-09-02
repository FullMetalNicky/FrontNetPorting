import cv2
import numpy as np
import os
import sys
sys.path.append("../")
import config

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

	def adjust_gamma(self, gamma=1.0):
		invGamma = 1.0 / gamma
		table = np.array([((i / 255.0) ** invGamma) * 255
			for i in np.arange(0, 256)]).astype("uint8")
	
		return table

	def ApplyVignette(self, rows, cols, sigma=150):

		# generating vignette mask using Gaussian kernels
		kernel_x = cv2.getGaussianKernel(cols,sigma)
		kernel_y = cv2.getGaussianKernel(rows,sigma)
		kernel = kernel_y * kernel_x.T
		#mask = 255 * kernel / np.linalg.norm(kernel)
		mask = kernel / kernel.max()
		return mask
		 #output[:,:,i] = output[:,:,i] * mask
	

	def get_crop_parameters(self, himaxCalibFile, bebopCalibFile):

		himax_fovx, himax_fovy, himax_h, himax_w = self.CalculateFOVfromCalibration(himaxCalibFile)
		bebop_fovx, bebop_fovy, bebop_h, bebop_w = self.CalculateFOVfromCalibration(bebopCalibFile)
		x_ratio = bebop_fovx / himax_fovx
		y_ratio = bebop_fovy / himax_fovy

		new_size, shift_x, shift_y = self.calc_crop_parameters(x_ratio, y_ratio, himax_h, himax_w)

		return new_size, shift_x, shift_y
		

	def calc_crop_parameters(self, x_ratio, y_ratio, h, w):

		new_size = (int(w * x_ratio) , int(h * y_ratio))
		shift_x = 0.5*(w - new_size[0])
		shift_y = 0.5*(h - new_size[1])

		return new_size, shift_x, shift_y



	def TransformImages(self, himaxCalibFile, bebopCalibFile, himaxImages, bebopImages):

		himax_fovx, himax_fovy, himax_h, himax_w = self.CalculateFOVfromCalibration(himaxCalibFile)
		bebop_fovx, bebop_fovy, bebop_h, bebop_w = self.CalculateFOVfromCalibration(bebopCalibFile)
		x_ratio = bebop_fovx / himax_fovx
		y_ratio = bebop_fovy / himax_fovy

		new_size, shift_x, shift_y = self.calc_crop_parameters(x_ratio, y_ratio, himax_h, himax_w)

		himaxTransImages = []
		bebopTransImages = []
		resize_h = config.input_height
		resize_w = config.input_width
		w_himax = config.himax_width
		h_himax = config.himax_height

		for img in himaxImages:
			#crop_img = img[shift_y:shift_y+new_size[1], shift_x:shift_x+ new_size[0]]	
			crop_img = img
			crop_img = cv2.resize(crop_img, (resize_w, resize_h), cv2.INTER_NEAREST)
			himaxTransImages.append(crop_img)

		table = self.adjust_gamma(0.6)
		mask = self.ApplyVignette(w_himax, w_himax, 150)
		for img in bebopImages:
			gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
			gray = cv2.LUT(gray, table)
			gray = cv2.GaussianBlur(gray,(5,5),0)
			gray = cv2.resize(gray, (w_himax, h_himax), cv2.INTER_AREA)
			gray = gray *  mask[40:284, 0:324]
			gray = cv2.resize(gray, (resize_w, resize_h), cv2.INTER_NEAREST)
			bebopTransImages.append(gray)

		return himaxTransImages, bebopTransImages
