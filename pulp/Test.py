from ImageIO import ImageIO
from CameraCalibration import CameraCalibration 
from RosbagUnpacker import RosbagUnpacker



def TestImageIO():

	images = ImageIO.ReadImagesFromFolder("../data/himax/", '.jpg')
	ImageIO.WriteImagesToFolder(images, "../data/test/", '.jpg')


def TestCameraCalibration():
	cc = CameraCalibration()
	images = cc.CaptureCalibration("image_pipe")
	cc.CalibrateImages(images, "test.yaml")

def TestRosbagUnpacker():
	rbu = RosbagUnpacker()
	himax_msgs, bepop_msgs = rbu.UnpackBag('../data/2019-08-08-08-17-30.bag', stopNum=3)
	himax_images, bepop_images = rbu.MessagesToImages(himax_msgs, bepop_msgs)
	ImageIO.WriteImagesToFolder(himax_images, "../data/test/", '.jpg')	


def main():
	#TestImageIO()
	TestRosbagUnpacker()



if __name__ == '__main__':
    main()
