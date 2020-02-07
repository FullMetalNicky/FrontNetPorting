# !/usr/bin/env python

import os
import subprocess
import config


#streaming only himax images from the GAP and rosbagging them with optitrack data
proc1 = subprocess.Popen(['gnome-terminal', '-e', "python pulp/HimaxPublisher.py"])
proc2 = subprocess.Popen(['gnome-terminal', '-e' ,"make clean all run"], cwd=config.folder_path +"/pulp/camera_to_fifo/")
proc3 = subprocess.Popen(['gnome-terminal', '-e', "rosrun rqt_image_view rqt_image_view /himax_camera"])
proc4 = subprocess.Popen(['gnome-terminal', '-e', "rosrun rqt_image_view rqt_image_view /bebop/image_raw"])
#proc3 = subprocess.Popen(['gnome-terminal', '-e', "python pulp/Visualizer.py"])
#proc4 = subprocess.Popen(['gnome-terminal', '-e', "rosrun rqt_image_view rqt_image_view /concat_cameras"])
#proc5 = subprocess.Popen(['gnome-terminal', '-e', "rosbag record himax_camera bebop/image_raw optitrack/hand optitrack/head optitrack/drone"])
proc5 = subprocess.Popen(['gnome-terminal', '-e', "rosbag record himax_camera bebop/image_raw optitrack/head optitrack/drone"])


#streaming only himax images from the PULP SHEILD and rosbagging them
# proc1 = subprocess.Popen(['gnome-terminal', '-e', "python pulp/HimaxPublisher.py"])
# proc2 = subprocess.Popen(['gnome-terminal', '-e' ,"plpbridge --chip=gap --cable=ftdi --binary=build/gap/test/test load ioloop reqloop start wait"], cwd=config.folder_path +"/pulp/camera_to_fifo/")
# pro3 = subprocess.Popen(['gnome-terminal', '-e', "rosrun rqt_image_view rqt_image_view /himax_camera"])
# proc4 = subprocess.Popen(['gnome-terminal', '-e', "rosbag record himax_camera"])

