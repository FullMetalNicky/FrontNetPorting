import cv2
import numpy as np
import os


cv_file = cv2.FileStorage("calibration.yaml", cv2.FILE_STORAGE_READ)
k = cv_file.getNode("k").mat()
D = cv_file.getNode("D").mat()
size = cv_file.getNode("size").mat()
h_himax = size[0][0]
w_himax = size[1][0]
cv_file.release()
fovx_himax, fovy_himax, focalLength, principalPoint, aspectRatio = cv2.calibrationMatrixValues(k, (w_himax, h_himax), 0.0036, 0.0036) 
print("himax fovx {} fovy {}".format(fovx_himax, fovy_himax))

cv_file = cv2.FileStorage("bebop_calibration.yaml", cv2.FILE_STORAGE_READ)
k = cv_file.getNode("k").mat()
D = cv_file.getNode("D").mat()
size = cv_file.getNode("size").mat()
h_bebop = size[0][0]
w_bebop = size[1][0]
cv_file.release()

fovx_bebop, fovy_bebop, focalLength, principalPoint, aspectRatio = cv2.calibrationMatrixValues(k, (w_bebop, h_bebop), 1, 1) 
print("bebop fovx {} fovy {}".format(fovx_bebop, fovy_bebop))

x_ratio = fovx_bebop / fovx_himax
y_ratio = fovy_bebop / fovy_himax
new_size = (int(w_himax * x_ratio) , int(h_himax * y_ratio))
print ("x ratio {}, y ration {}".format(x_ratio, y_ratio))
print ("new size {}".format(new_size))

himax_folder = "../data/himax/"
himax_processed_folder = "../data/himax_processed/"
bebop_folder = "../data/bebop/"
bebop_processed_folder = "../data/bebop_processed/"

#crop all the himax images

files = []
shift_x = 0.5*(w_himax - new_size[0])
shift_y = 0.5*(h_himax - new_size[1])

for r, d, f in os.walk(himax_folder):
	for file in f:
	    if '.jpg' in file:
	        files.append(os.path.join(r, file))
for f in files:
		img = cv2.imread(f)	
		crop_img = img[shift_y:shift_y+new_size[1], shift_x:shift_x+ new_size[0]]	
		cv2.imwrite(himax_processed_folder + os.path.basename(f), crop_img)
		#print(himax_processed_folder + os.path.basename(f))
		

#grey scale all bebop images
files = []
for r, d, f in os.walk(bebop_folder):
	for file in f:
	    if '.jpg' in file:
	        files.append(os.path.join(r, file))

for f in files:
		img = cv2.imread(f)	
		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		cv2.imwrite(bebop_processed_folder + os.path.basename(f), gray)










