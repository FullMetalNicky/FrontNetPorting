/*
 * Copyright (C) 2018 ETH Zurich and University of Bologna
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

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
