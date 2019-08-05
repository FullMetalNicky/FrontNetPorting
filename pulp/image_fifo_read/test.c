

#include <stdio.h>
#include <fcntl.h>
#include "ImgIO.h"
#include <stdlib.h> 
 #include <unistd.h> 


int main()
{
  unsigned int width = 322, height = 242;

	unsigned char *outBuffer = (unsigned char *) malloc(width*height*sizeof(unsigned char));
	if (outBuffer == NULL) 
  {
		printf("outBuffer no memory\n");
		return -1;
  }
 
	printf("Reading from pipe...\n");
	
	ReadImageFromFifo("../image_pipe", width, height, outBuffer);

  int ret = WriteImageToFile("Pedestrian_OUT.ppm",width,height,outBuffer);
	printf("Finished writing to file, ret %d bytes.\n", ret);
	

	free(outBuffer);
  printf("Test success\n");

  return 0;
}
