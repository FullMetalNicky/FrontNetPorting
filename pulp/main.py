from ImageIO import ImageIO
from CameraCalibration import CameraCalibration 
from RosbagUnpacker import RosbagUnpacker
from CameraSynchronizer import CameraSynchronizer
from ImageTransformer import ImageTransformer


#flow example
def main():
	
	cs = CameraSynchronizer('../data/2019-08-08-05-06-09.bag')
	himax_stamps, bebop_stamps = cs.UnpackBagStamps()
	print("get stamps")
	sync_himax_ids, sync_bebop_ids = cs.SyncStamps(himax_stamps, bebop_stamps, -1817123289)
	print("synched stamps")
	sync_himax_images, sync_bebop_images = cs.SyncImagesByStamps(sync_himax_ids, sync_bebop_ids)
	print("synched images")
	it = ImageTransformer()
	himaxTransImages, bebopTransImages = it. TransformImages("../data/calibration.yaml", "../data/bebop_calibration.yaml", sync_himax_images, sync_bebop_images)
	print("transformed")
#	cs.CreateSyncVideo(himaxTransImages, bebopTransImages, "test.avi")
	ImageIO.WriteImagesToFolder(himaxTransImages, "../data/himax_processed/", '.jpg')
	ImageIO.WriteImagesToFolder(bebopTransImages, "../data/bebop_processed/", '.jpg')
	

if __name__ == '__main__':
    main()
