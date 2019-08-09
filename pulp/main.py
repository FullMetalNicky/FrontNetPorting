from ImageIO import ImageIO
from CameraCalibration import CameraCalibration 
from RosbagUnpacker import RosbagUnpacker
from CameraSynchronizer import CameraSynchronizer
from ImageTransformer import ImageTransformer


#flow example
def main():
	
	rbu = RosbagUnpacker()
	himax_msgs, bebop_msgs = rbu.UnpackBag('../data/2019-08-08-08-17-30.bag', stopNum=3)
	himax_images, bebop_images, himax_stamps, bebop_stamps = rbu.MessagesToImages(himax_msgs, bebop_msgs)
	cs = CameraSynchronizer()
	sync_himax_images, sync_bebop_images = cs.SyncImages(himax_images, bebop_images, himax_stamps, bebop_stamps, -1817123289)
	cs.CreateSyncVideo(sync_himax_images, sync_bebop_images, "test.avi")
	it = ImageTransformer()
	himaxTransImages, bebopTransImages = it. TransformImages("../data/calibration.yaml", "../data/bebop_calibration.yaml", sync_himax_images, sync_bebop_images)
	ImageIO.WriteImagesToFolder(himaxTransImages, "../data/test/", '.jpg')
	

if __name__ == '__main__':
    main()
