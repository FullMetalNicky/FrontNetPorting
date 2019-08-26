from ImageIO import ImageIO
from CameraCalibration import CameraCalibration 
from RosbagUnpacker import RosbagUnpacker
from CameraSynchronizer import CameraSynchronizer
from TimestampSynchronizer import TimestampSynchronizer
from ImageTransformer import ImageTransformer
import subprocess

#flow example
def main():

	cmd = 'rm -f ../data/himax_processed/*.jpg'
	subprocess.call(cmd, shell=True)
	cmd = 'rm -f ../data/bebop_processed/*.jpg'
	subprocess.call(cmd, shell=True)
	

	ts = TimestampSynchronizer('../data/monster.bag')
	himax_stamps, bebop_stamps = ts.UnpackBagStamps()
	print("get stamps")
	sync_himax_ids, sync_bebop_ids = ts.SyncStamps(himax_stamps, bebop_stamps, -1817123289)
	print("synched stamps")

	cs = CameraSynchronizer('../data/monster.bag')
	sync_himax_images, sync_bebop_images = cs.SyncImagesByStamps(sync_himax_ids, sync_bebop_ids)
	print("synched images")
	it = ImageTransformer()
	himaxTransImages, bebopTransImages = it. TransformImages("../data/calibration.yaml", "../data/bebop_calibration.yaml", sync_himax_images, sync_bebop_images)
	print("transformed")

	ImageIO.WriteImagesToFolder(himaxTransImages, "../data/himax_processed/", '.jpg')
	ImageIO.WriteImagesToFolder(bebopTransImages, "../data/bebop_processed/", '.jpg')

	cs.CreateSyncVideo(himaxTransImages, bebopTransImages, "monster.avi", 1)
	#frames = cs.GetSyncConcatFrames(himaxTransImages, bebopTransImages)
	#ImageIO.WriteImagesToFolder(frames, "../data/video/", '.jpg')
	#cmd = 'convert -delay 100 ../data/video/*.jpg -loop 0 ../data/video/sync.gif'
	#subprocess.call(cmd, shell=True)
	

if __name__ == '__main__':
    main()
