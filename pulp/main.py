from ImageIO import ImageIO
from CameraCalibration import CameraCalibration 
from RosbagUnpacker import RosbagUnpacker
from CameraSynchronizer import CameraSynchronizer
from ImageTransformer import ImageTransformer
import subprocess

#flow example
def main():
	
	cs = CameraSynchronizer('../data/2019-08-09-04-27-04.bag')
	himax_stamps, bebop_stamps = cs.UnpackBagStamps(30)
	print("get stamps")
	sync_himax_ids, sync_bebop_ids = cs.SyncStamps(himax_stamps, bebop_stamps, -1817123289)
	print("synched stamps")
	sync_himax_images, sync_bebop_images = cs.SyncImagesByStamps(sync_himax_ids, sync_bebop_ids)
	print("synched images")
	it = ImageTransformer()
	himaxTransImages, bebopTransImages = it. TransformImages("../data/calibration.yaml", "../data/bebop_calibration.yaml", sync_himax_images, sync_bebop_images)
	print("transformed")
	#cs.CreateSyncVideo(himaxTransImages, bebopTransImages, "test.avi", 1)
	frames = cs.GetSyncConcatFrames(himaxTransImages, bebopTransImages)
	ImageIO.WriteImagesToFolder(frames, "../data/video/", '.jpg')
	
	ImageIO.WriteImagesToFolder(himaxTransImages, "../data/himax_processed/", '.jpg')
	ImageIO.WriteImagesToFolder(bebopTransImages, "../data/bebop_processed/", '.jpg')

	cmd = 'convert -delay 100 ../data/video/*.jpg -loop 0 ../data/video/sync.gif'
	subprocess.call(cmd, shell=True)
	

if __name__ == '__main__':
    main()
