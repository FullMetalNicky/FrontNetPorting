/*
 * Copyright (C) 2017 GreenWaves Technologies
 * All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD license.  See the LICENSE file for details.
 *
 */

#include <stdio.h>
#include <fcntl.h>
#include "ImgIO.h"
#include <stdlib.h> 
#include <unistd.h> 


#define PPM_HEADER 40

#define Max(a, b)               (((a)>(b))?(a):(b))
#define Min(a, b)               (((a)<(b))?(a):(b))

#define ALIGN(Value, Size)      (((Value)&((1<<(Size))-1))?((((Value)>>(Size))+1)<<(Size)):(Value))

#define CHUNK_SIZE 8192

static void progress_bar(char * OutString, int n, int tot){
	int tot_chars = 30;
	printf("%s",OutString);
	printf(" [");
	int chars = (n*tot_chars)/tot;

	for(int i=0;i<tot_chars;i++){
		if(i<=chars)
			printf("#");
		else printf(" ");
	}
	printf("]");
	printf("\n");

}


static unsigned int SkipComment(unsigned char *Img, unsigned int Ind)

{
	while (Img[Ind] == '#') {
		while (Img[Ind] != '\n') {printf("%c", Img[Ind]);Ind++;}
		Ind++;
	}
	return Ind;
}



static void WritePPMHeader(int FD, unsigned int W, unsigned int H)
{
  	unsigned int Ind = 0, x, i, L;
		unsigned char *Buffer = (unsigned char *) malloc(PPM_HEADER*sizeof(unsigned char));
	

  	/* P5<cr>* */
  	Buffer[Ind++] = 0x50; Buffer[Ind++] = 0x35; Buffer[Ind++] = 0xA;

  	/* W <space> */
  	x = W; L=0;
  	while (x>0) { x = x/10; L++; }
  	x = W; i = 1;
  	while (x>0) { Buffer[Ind+L-i] = 0x30 + (x%10); i++; x=x/10; }
  	Ind += L;
  	Buffer[Ind++] = 0x20;

  	/* H <cr> */
  	x = H; L=0;
  	while (x>0) { x = x/10; L++; }
  	x = H; i = 1;
  	while (x>0) { Buffer[Ind+L-i] = 0x30 + (x%10); i++; x=x/10; }
  	Ind += L;
  	Buffer[Ind++] = 0xA;

  	/* 255 <cr> */
  	Buffer[Ind++] = 0x32; Buffer[Ind++] = 0x35; Buffer[Ind++] = 0x35; Buffer[Ind++] = 0xA;
  	
		write(FD, Buffer, Ind*sizeof(unsigned char));

		free(Buffer);

  }


int WriteImageToFile(char *ImageName, unsigned int W, unsigned int H, unsigned char *OutBuffer)
{
	#define CHUNK_NUM 10
	int File = open(ImageName,O_WRONLY |O_CREAT| O_TRUNC, S_IRWXU);
	int ret = 0;
	WritePPMHeader(File,W,H);
	lseek(File, 0L, 2);

	
	ret += 	write(File, OutBuffer, (W*H)*sizeof(unsigned char));
	close(File);

	return ret;
}


unsigned char* ReadImageFromFifo(char *fifoName, unsigned int W, unsigned int H, unsigned char *InBuffer)
{
	int pipeid = open(fifoName, O_RDONLY);
	if (pipeid == -1) 
  {
		printf("Failed to open pipe %d \n", pipeid);
		return NULL;
  }

	unsigned int Size = (W)*(H);
	unsigned int AlignedSize = Size;
	unsigned char * ImagePtr = InBuffer;
	unsigned char *TargetImg = ImagePtr;
	unsigned int RemainSize = AlignedSize;
	unsigned int ReadSize = 0;

	while (RemainSize > 0) 
	{
		unsigned int Chunk = Min(4096, RemainSize);
		unsigned R = read(pipeid,TargetImg, Chunk);
		ReadSize+=R;
		if (R!=Chunk) break;
		TargetImg += Chunk; RemainSize -= Chunk;
	}
	if (AlignedSize!=ReadSize) 
	{
		printf("Error, expects %d bytes but got %d\n", AlignedSize, ReadSize); 
		return NULL;
	}
	return (ImagePtr);

}
