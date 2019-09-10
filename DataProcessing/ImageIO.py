# !/usr/bin/env python


import numpy as np
import cv2
import os
import sys


class ImageIO:

    @staticmethod
    def WriteImagesToFolder(images, folderName, imgType='.jpg'):
         """Writes a list of images to folder

            Parameters
            ----------
            images : list
                list of images
            folderName : str
                location for the images to be saved
            imgType : str, optional
                type of image format)
            """
        frame_id = 1

        for img in images:
            img_name = folderName + "{}".format(frame_id) + imgType
            frame_id = frame_id + 1
            cv2.imwrite(img_name, img)

    @staticmethod
    def ReadImagesFromFolder(folderName, imgType='.jpg', read_mode=1):
        """Writes a list of images to folder

            Parameters
            ----------
            folderName : str
                location for the images to be saved
            imgType : str, optional
                type of image format
            read_mode: int, optional
                OpenCV reading mode 
            
            Returns
            -------
            list
                list of images found in the folder
            """

        files = []
        images = []
        for r, d, f in os.walk(folderName):
            for file in f:
                if imgType in file:
                    files.append(os.path.join(r, file))
        if sys.version_info[0] >= 3:
            files.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
        else:
            files.sort(key=lambda f: int(filter(str.isdigit, f)))
        for f in files:
            img = cv2.imread(f, read_mode)
            images.append(img)

        return images
