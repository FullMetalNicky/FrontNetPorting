# !/usr/bin/env python


import numpy as np
import cv2
import os
import sys


class ImageEffects:

	@staticmethod
	def ConcatImages(img1, img2):
	"""Writes a list of images to folder

        Parameters
        ----------
        img1 : OpenCV image type
            First image to be concatenated
        img2 : OpenCV image type
            second image to be concatenated
        
        Returns
        -------
        OpenCV image type
            concatenated image
        """
		if len(img1.shape) < 3:
			img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2RGB)
		if len(img2.shape) < 3:
			img2=img2.astype(np.uint8)
			img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2RGB)
		h1, w1 = img1.shape[:2]
		h2, w2 = img2.shape[:2]

		vis = np.zeros((max(h1, h2), w1+w2,3), np.uint8)

		vis[:h1, :w1,:3] = img1
		vis[:h2, w1:w1+w2,:3] = img2
		return vis

	@staticmethod
	def GetGammaLUT(gamma=1.0):
		invGamma = 1.0 / gamma
		table = np.array([((i / 255.0) ** invGamma) * 255
			for i in np.arange(0, 256)]).astype("uint8")
	
		return table

	@staticmethod
	def ApplyGamma(img, gammaLUT):

		return cv2.LUT(img, gammaLUT)

	@staticmethod
	def GetVignetteMask(rows, cols, sigma=150):
		"""Generates a vignette mask

        Parameters
        ----------
        rows : int
            height of the mask
        cols : int
            width of the mask
        sigma : int, optional 
        	the sigma of the Gaussian kernel
        
        Returns
        -------
        OpenCV image type
            vignette mask
        """

		kernel_x = cv2.getGaussianKernel(cols,sigma)
		kernel_y = cv2.getGaussianKernel(rows,sigma)
		kernel = kernel_y * kernel_x.T
		#mask = 255 * kernel / np.linalg.norm(kernel)
		mask = kernel / kernel.max()
		return mask

	@staticmethod
	def ApplyVignette(img, vignetteMask):

		return img * vignetteMask

	@staticmethod
	def GenerateRandomDynamicRange():
		dr = np.random.uniform(0.4, 0.8)  # dynamic range
		low = np.random.uniform(0, 0.3)
		high = min(1.0, low + dr)

		return low, high

	@staticmethod
	def ApplyDynamicRange(img, low, high):

		img = np.interp(img/255.0, [0, low, high, 1], [0, 0, 1, 1])
		return img * 255


	

