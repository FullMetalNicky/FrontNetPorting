#!/usr/bin/env python


import numpy as np
import cv2
import os


class CameraCalibration:


	def read_from_pipe(self, pipein):
		remaining_size = height * width
	
		data = []
		while(remaining_size >= page_size):
			output =  os.read(pipein, page_size)
			remaining_size = remaining_size - len(output)
			data.append(output)

		data.append(os.read(pipein, max(0, remaining_size)))
		data=''.join(data)

		if (len(data) < height*width):
				print("Error, expecting {} bytes, received {}.".format(height*width, len(data)))
				return None

		data = np.frombuffer(data, dtype=np.uint8)

		return data

	def CaptureCalibration(self, pipe_name):

		if not os.path.exists(pipe_name):
			os.mkfifo(pipe_name)

		pipein = os.open(pipe_name, os.O_RDONLY)
		images = []

		while (1):
			data = read_from_pipe(pipein)
			if data is not None:
				cv_image = np.reshape(data, (height, width))
				cv2.imshow("Calibration", cv_image)
				c = cv2.waitKey(10)
				if 'c' == chr(c & 255):
					images.append(cv_image)
				if 'q' == chr(c & 255):
					break

		os.close(pipein)
		return images

	def CalibrateImages(self, images, calibFileName):

		criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
		objp = np.zeros((4*6,3), np.float32)
		objp[:,:2] = np.mgrid[0:6,0:4].T.reshape(-1,2)
		objpoints = [] 
		imgpoints = [] 
		for img in images:
			gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
			ret, corners = cv2.findChessboardCorners(gray, (6,4),cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE)

			if ret == True:
				objpoints.append(objp)
				corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
				imgpoints.append(corners2)

				img = cv2.drawChessboardCorners(img, (6,4), corners2,ret)
				cv2.imshow('img',img)
				cv2.waitKey(50)

		cv2.destroyAllWindows()
		ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)
		
		cv_file = cv2.FileStorage(calibFileName, cv2.FILE_STORAGE_WRITE)
		cv_file.write("k", mtx)
		cv_file.write("D", dist)
		cv_file.write("size", np.array([h, w]))
		cv_file.release()


