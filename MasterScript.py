# !/usr/bin/env python

import os
import subprocess
import config


proc1 = subprocess.Popen(['gnome-terminal', '-e', "python pulp/HimaxPublisher.py"])
proc2 = subprocess.Popen(['gnome-terminal', '-e' ,"make clean all run"], cwd=config.folder_path +"/pulp/camera_to_fifo/")
proc3 = subprocess.Popen(['gnome-terminal', '-e', "rosrun rqt_image_view rqt_image_view /himax_camera"])
proc4 = subprocess.Popen(['gnome-terminal', '-e', "rosrun rqt_image_view rqt_image_view /bebop/image_raw"])
#proc3 = subprocess.Popen(['gnome-terminal', '-e', "python pulp/Visualizer.py"])
#proc4 = subprocess.Popen(['gnome-terminal', '-e', "rosrun rqt_image_view rqt_image_view /concat_cameras"])
proc5 = subprocess.Popen(['gnome-terminal', '-e', "rosbag record himax_camera bebop/image_raw optitrack/hand optitrack/head optitrack/drone"])
