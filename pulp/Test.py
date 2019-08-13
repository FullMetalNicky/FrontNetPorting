from ImageIO import ImageIO
from CameraCalibration import CameraCalibration 
from RosbagUnpacker import RosbagUnpacker
from CameraSynchronizer import CameraSynchronizer
from ImageTransformer import ImageTransformer

def TestImageIO():

	images = ImageIO.ReadImagesFromFolder("../data/himax/", '.jpg')
	ImageIO.WriteImagesToFolder(images, "../data/test/", '.jpg')


def TestCameraCalibration():
	cc = CameraCalibration()
	images = cc.CaptureCalibration("image_pipe")
	cc.CalibrateImages(images, "test.yaml")

def TestRosbagUnpacker():
	rbu = RosbagUnpacker()
	himax_msgs, bebop_msgs = rbu.UnpackBag('../data/2019-08-08-08-17-30.bag', stopNum=3)
	himax_images, bebop_images, himax_stamps, bebop_stamps = rbu.MessagesToImages(himax_msgs, bebop_msgs)
	ImageIO.WriteImagesToFolder(himax_images, "../data/test/", '.jpg')	


def TestCameraSynchronizer():
	rbu = RosbagUnpacker()
	himax_msgs, bebop_msgs = rbu.UnpackBag('../data/2019-08-08-08-17-30.bag', stopNum=3)
	himax_images, bebop_images, himax_stamps, bebop_stamps = rbu.MessagesToImages(himax_msgs, bebop_msgs)
	cs = CameraSynchronizer()
	sync_himax_images, sync_bebop_images = cs.SyncImages(himax_images, bebop_images, himax_stamps, bebop_stamps, -1817123289)
	cs.CreateSyncVideo(sync_himax_images, sync_bebop_images, "test.avi")

def TestImageTransformer():


	himaxImages = ImageIO.ReadImagesFromFolder("../data/himax/", '.jpg')
	bebopImages = ImageIO.ReadImagesFromFolder("../data/bebop/", '.jpg')
	it = ImageTransformer()
	himaxTransImages, bebopTransImages = it. TransformImages("../data/calibration.yaml", "../data/bebop_calibration.yaml", himaxImages, bebopImages)
	ImageIO.WriteImagesToFolder(himaxTransImages, "../data/test/", '.jpg')

def main():
	#TestImageIO()
	#TestRosbagUnpacker()
	#TestCameraSynchronizer()
	TestImageTransformer()



if __name__ == '__main__':
    main()
