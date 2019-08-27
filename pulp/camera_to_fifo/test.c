/* I got help from ETH guys, but I did write most of it myself 
 or at least, copy-pasted cleverly from different projects :) */


#include "rt/rt_api.h"
#include "rt/rt_time.h"
#include "rt/data/rt_data_camera.h"
#include "ImgIO.h"


// This strange resolution comes from the himax camera
#define WIDTH     324
#define HEIGHT    244

#ifdef USE_RAW
#define FB_FORMAT RT_FB_FORMAT_RAW
#else
#define FB_FORMAT RT_FB_FORMAT_GRAY
#endif


static unsigned char* 	L2_image;
static int				imgTransferDone = 0;
static rt_event_t *		event_capture;
static rt_camera_t *	camera;
static int frame_id = 0;


static void end_of_frame() 
{

	rt_cam_control(camera, CMD_PAUSE, NULL);

	//WriteImageToFifo("../../../image_pipe", WIDTH, HEIGHT, L2_image);
	WriteImageToFifo("/home/usi/Documents/Drone/FrontNetPorting/pulp/image_pipe", WIDTH, HEIGHT, L2_image);


	imgTransferDone = 1;
}

static void enqueue_capture() 
{

	rt_cam_control(camera, CMD_START, NULL);

	rt_camera_capture(camera, (unsigned char*)L2_image, WIDTH*HEIGHT*sizeof(unsigned char), rt_event_get(NULL, end_of_frame, NULL));
}



int main()
{
  printf("Entering main controller\n");

  // First wait until the external bridge is connected to the platform
  printf("Connecting to bridge\n");
  rt_bridge_connect(1, NULL);
  printf("Connection done\n");

  // Allocate 3 buffers, there will be 2 for the camera double-buffering and one
  // for flushing an image to the framebuffer.
  
	L2_image = rt_alloc(RT_ALLOC_L2_CL_DATA, WIDTH*HEIGHT*sizeof(unsigned char));

  printf("Finished allocation\n");


  // We'll need one event per buffer
  if (rt_event_alloc(NULL,1)) return -1;

  // Configure Himax camera on interface 0
  rt_cam_conf_t cam_conf;
  rt_camera_conf_init(&cam_conf);
  cam_conf.id = 0;
  cam_conf.control_id = 1;
  cam_conf.type = RT_CAM_TYPE_HIMAX;
  cam_conf.resolution = QVGA;
  cam_conf.format = HIMAX_MONO_COLOR;
  cam_conf.fps = fps30;
  cam_conf.slice_en = DISABLE;
  cam_conf.shift = 0;
  cam_conf.frameDrop_en = DISABLE;
  cam_conf.frameDrop_value = 0;
  cam_conf.cpiCfg = UDMA_CHANNEL_CFG_SIZE_8;



  printf("camera configured\n");
  // Open the camera
  camera = rt_camera_open(NULL, &cam_conf, 0);
  if (camera == NULL) return -1;

  printf("camera opened\n");
  rt_cam_control(camera, CMD_INIT, 0);
  //rt_time_wait_us(1000000); //Wait camera calibration
  
  printf("start camera\n");
  // Start it
  rt_cam_control(camera, CMD_START, 0);
	if(rt_platform() == ARCHI_PLATFORM_BOARD)
		rt_time_wait_us(1000000);
  printf("camera started\n");

 
  enqueue_capture();

	// wait on input image transfer 
	while(imgTransferDone==0) 
	{
		rt_event_yield(NULL);
	}


  // The main loop is not doing anything, everything will be done through event callbacks
  while(1)
  {
    // wait on input image transfer 
		while(imgTransferDone==0) 
		{
			rt_event_yield(NULL);
		}
		imgTransferDone=0;

		++frame_id;
		printf("Transferred frame %d\n", frame_id);

		event_capture = rt_event_get(NULL, enqueue_capture, NULL);
		rt_event_push(event_capture);
  }

  rt_camera_close(camera, 0);
	rt_free(RT_ALLOC_L2_CL_DATA, L2_image, WIDTH*HEIGHT*sizeof(unsigned char));


  return 0;
}
