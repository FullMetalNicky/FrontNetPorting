

#include <stdio.h>
#include <fcntl.h>
#include "ImgIO.h"
#include <stdlib.h> 
 #include <unistd.h> 

static int frame_id = 0;
#define WIDTH     324
#define HEIGHT    244

int main()
{
  unsigned int width = 324, height = 244;

	unsigned char *outBuffer = (unsigned char *) malloc(WIDTH*HEIGHT*sizeof(unsigned char));
	if (outBuffer == NULL) 
  {
		printf("outBuffer no memory\n");
		return -1;
  }

	if(access("../image_pipe", F_OK ))
	{
    int res = mkfifo("../image_pipe", 0666); 
		if (res == -1) 
		{
			printf("Could not create pipe\n");
			return -1;
		}
	}
	
 while(1)
{
	printf("Reading from pipe...\n");
	
	unsigned char* ptr = ReadImageFromFifo("../image_pipe", WIDTH, HEIGHT, outBuffer);
	if (NULL != ptr)
	{

		++frame_id;
		char ImageName[15]; 
		sprintf(ImageName, "%d.ppm", frame_id);

		int ret = WriteImageToFile(ImageName,WIDTH,HEIGHT,outBuffer);
		printf("Finished writing to file, ret %d bytes.\n", ret);
	}
	memset(outBuffer, 0, WIDTH*HEIGHT*sizeof(unsigned char));
}

	free(outBuffer);
  printf("Test success\n");

  return 0;
}
