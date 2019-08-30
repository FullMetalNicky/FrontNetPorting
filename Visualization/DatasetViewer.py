import numpy as np 
import pandas as pd
import cv2
import matplotlib.patches as patches
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

class DatasetViewer:

	def LoadDataset(self, fileName):
		test_set = pd.read_pickle(fileName).values

		self.frames = test_set[:, 0]
		self.labels = test_set[:, 1]

		#self.frames = np.reshape(frames, (len(frames), 60, 108))
		#self.labels = np.reshape(labels, (-1, 4))

	def PlotTrackingAndDisplayVideo(self, video=False):
		
		fig = plt.figure(888, figsize=(15, 5))

		h = 9
		w = 16

		ax0 = plt.subplot2grid((h, w), (0, 0), colspan=7, rowspan=2)
		ax0.set_ylim([0, 9])
		annotation = ax0.annotate("poop", xy=(0, 5.5), size=10)
		annotation.set_animated(True)
		ax0.axis('off')

		ax1 = plt.subplot2grid((h, w), (2, 0), colspan=8, rowspan=7)
		ax1.set_title('Relative Pose (x,y)')
		ax1.yaxis.set_ticks([0, 1.5, 3])  # set y-ticks
		ax1.xaxis.set_ticks([-3.0, -1.5, 0, 1.5, 3.0])  # set y-ticks
		ax1.xaxis.tick_top()  # and move the X-Axis
		ax1.yaxis.tick_left()  # remove right y-Ticks
		ax1.spines['right'].set_visible(False)
		ax1.spines['bottom'].set_visible(False)
		ax1.invert_xaxis()
		trianglex = [2, 0, -2, 2]
		triangley = [3, 0, 3, 3]
		#collection = plt.fill(trianglex, triangley, facecolor='lightskyblue')

		plot1gt, = plt.plot([], [], color='green', label='GT', linestyle='None', marker='o', markersize=10)
		#arr1gt = ax1.arrow([], [], np.cos([]), np.sin([]), head_width=0.1, head_length=0.1, color='green', animated=True)
		#arr1pr = ax1.arrow([], [], np.cos([]), np.sin([]), head_width=0.1, head_length=0.1, color='blue', animated=True)
		plt.legend(loc='lower right', bbox_to_anchor=(0.8, 0.2, 0.25, 0.25))

		ax2 = plt.subplot2grid((h, w), (2, 8), rowspan=7)
		#ax2.set_title('Relative z', pad=20)
		ax2.yaxis.tick_right()
		ax2.set_xlim([-0.5, 0.5])
		ax2.set_xticklabels([])
		ax2.yaxis.set_ticks([-1, 0, 1])  # set y-ticks
		ax2.xaxis.set_ticks_position('none')
		scatter2gt = plt.scatter([], [], color='green', label='GT', s=100)

		ax3 = plt.subplot2grid((h, w), (2, 9), rowspan=7, colspan=7)
		ax3.axis('off')
		frame = self.frames[0].astype(np.uint8)
		imgplot = plt.imshow(frame,cmap="gray", vmin=0, vmax=255)

		ax4 = plt.subplot2grid((h, w), (0, 9), colspan=7)
		ax4.set_xlim([0, 8])
		annotation2 = ax4.annotate("poop", xy=(3, 0.1), size=14, weight='bold')
		annotation2.set_animated(True)
		ax4.axis('off')

		plt.subplots_adjust(wspace=1.5)

		img = mpimg.imread('minidrone.jpg')
		newax = fig.add_axes([0.26, 0.0, 0.1, 0.1], anchor='S')
		newax.imshow(img)
		newax.axis('off')

		#Writer = animation.writers['ffmpeg']
		#writer = Writer(fps=2, metadata=dict(artist='FullMetalNicky'))


		def animate(id):
		    label = self.labels[id]
		    x_gt, y_gt, z_gt, phi_gt = label[0], label[1], label[2], label[3]

		    str1 = "x_gt={:05.3f}, y_gt={:05.3f}, z_gt={:05.3f}, phi_gt={:05.3f} {}".format(x_gt, y_gt, z_gt, phi_gt, "\n")

		    phi_gt = phi_gt - np.pi / 2

		    annotation.set_text(str1)

		    plot1gt.set_data(np.array([y_gt, x_gt]))

		  #  if(len(ax1.patches) > 1):
		   #     ax1.patches.pop()
		    #    ax1.patches.pop()

		    #patch1 = patches.FancyArrow(y_gt, x_gt, 0.5*np.cos(phi_gt), 0.5*np.sin(phi_gt), head_width=0.05, head_length=0.05, color='green')
		    #patch2 = patches.FancyArrow(y_pred, x_pred, 0.5*np.cos(phi_pred), 0.5*np.sin(phi_pred), head_width=0.05, head_length=0.05, color='blue')
		    #ax1.add_patch(patch1)
		    #ax1.add_patch(patch2)

		    scatter2gt.set_offsets(np.array([-0.05, z_gt]))

		    frame = self.frames[id].astype(np.uint8)
		    imgplot.set_array(frame)

		    annotation2.set_text('Frame {}'.format(id))

		    #use the first one for viz on screen and second one for video recording
		    #return plot1gt, imgplot
		    #return plot1gt, plot1pr, patch1, patch2, scatter2gt, scatter2pr, imgplot, annotation, annotation2
		    #return plot1gt, plot1pr, scatter2gt, scatter2pr, imgplot, annotation, annotation2


		ani = animation.FuncAnimation(fig, animate, frames=len(self.frames), interval=1, blit=True)
		if video == True:
			ani.save('head.avi', writer=writer)
		plt.show()




def main():
	dsv = DatasetViewer()
	dsv.LoadDataset("../DataProcessing/trainHimaxHead.pickle")
	dsv.PlotTrackingAndDisplayVideo(False)


if __name__ == '__main__':
    main()
