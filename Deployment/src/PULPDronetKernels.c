#include "PULPDronetKernelsInit.h"
#include "PULPDronetKernels.h"
void LargeParConv_5x5_S2_Max2x2_S2_H_1(
		short int * __restrict__ In,
		short int * __restrict__ Filter,
		short int * __restrict__ Out,
		unsigned int Norm,
		short int * __restrict__ Bias,
		Kernel_T *Ker)

{
	/* Local variables used by this kernel */
	rt_dma_copy_t DmaR_Evt1;
	rt_dma_copy_t DmaR_Evt2;
	rt_dma_copy_t DmaR_Evt3;
	rt_dma_copy_t DmaW_Evt1;
	int Iter;
	int Last, NextLast, NextNextLast;
	int N_Ti = 0;
	int Iter1;
	int Last1, NextLast1, NextNextLast1;
	int N_Ti1 = 0;
	int N_TiIp = 0, InPlane, OutPlane=0;
	KerParSetBias_fp_T S_KerArg0, *KerArg0 = &S_KerArg0;
	KerParConv_fp_T S_KerArg1, *KerArg1 = &S_KerArg1;
	KerParReLUMaxPool_fp_T S_KerArg2, *KerArg2 = &S_KerArg2;

	/* Initialize KerArg, Kernel invariant arguments */
	KerArg0->W = (unsigned short int) (54);
	KerArg0->OutFeatures = (unsigned short int) (32);
	KerArg1->W = (unsigned short int) (108);
	KerArg1->InFeatures = (unsigned short int) (1);
	KerArg1->OutFeatures = (unsigned short int) (32);
	KerArg1->Norm = (Norm);
	KerArg1->NTile = (unsigned short int) 4;
	KerArg1->Orientation = (unsigned short int) (1);
	KerArg1->Pad = (v4s) (33685504);
	KerArg1->TileSize = (unsigned short int) (19);
	KerArg1->TotalSize = (unsigned short int) (64);
	KerArg2->W = (unsigned short int) (54);
	KerArg2->OutFeatures = (unsigned short int) (32);
	KerArg2->Pad = (v4s) (16777216);
	KerArg2->DoReLU = (unsigned short int) (0);
	/* =======================Read First Tile=========================================== */
	/* Initial reads in L2, O_DB or O_BUFF */
	rt_dma_memcpy_2d((rt_pointerT) In+(0), (rt_pointerT) (PULP_Dronet_L1_Memory + 0)+0, (0?2592:3672), 
		12960, (0?2592:3672), RT_DMA_DIR_EXT2LOC, 0, &DmaR_Evt1);
	rt_dma_memcpy((rt_pointerT) Filter+(0), (rt_pointerT) (PULP_Dronet_L1_Memory + 49744)+0, 1600, RT_DMA_DIR_EXT2LOC, 0, &DmaR_Evt2);
	rt_dma_memcpy((rt_pointerT) Bias+0, (rt_pointerT) (PULP_Dronet_L1_Memory + 8208)+0, 64, RT_DMA_DIR_EXT2LOC, 0, &DmaR_Evt3);
	/* Wait for BUFF read in L2 */
	rt_dma_wait(&DmaR_Evt2);
	rt_dma_wait(&DmaR_Evt3);
	/* ===================End Read First Tile=========================================== */
	/* Kernel Iteration Loop on Iter space */
	for (Iter=0; Iter<4; Iter++) {
		/* Loop Iteration Body on Iter space */
		/* Elaborate Last, Next_Last, Next_Next_Last */
		Last = ((Iter+1) == 4); NextLast = ((Iter+2) == 4); NextNextLast = ((Iter+3) == 4);
		/* =======================Read Next Tile=========================================== */
		rt_dma_wait(&DmaR_Evt1);
		if (!Last) {
			rt_dma_memcpy_2d((rt_pointerT) In + ((Iter+1)*3456-432),
					(rt_pointerT) (PULP_Dronet_L1_Memory + 0) + (4104*((N_Ti+1) % 2)), (NextLast?3024:4104), 
					12960, (NextLast?3024:4104), RT_DMA_DIR_EXT2LOC, 0, &DmaR_Evt1);
		}
		/* ===================End Read Next Tile=========================================== */
		/* Call Kernel LOC_Unknown */
		KerArg0->Out = (short int * __restrict__) ((rt_pointerT) (PULP_Dronet_L1_Memory + 22096) + 0);
		KerArg0->H = (unsigned short int) (Last?6:8);
		KerArg0->Bias = (short int * __restrict__) ((rt_pointerT) (PULP_Dronet_L1_Memory + 8208) + 0);
		rt_team_fork(gap8_ncore(), (void *) KerParSetBias_fp, (void *) KerArg0);
		/* Kernel Iteration Loop on Iter1 space */
		Iter1=0; {
			/* Loop Iteration Body on Iter1 space */
			/* Elaborate Last, Next_Last, Next_Next_Last */
			Last1 = ((Iter1+1) == 1); NextLast1 = ((Iter1+2) == 1); NextNextLast1 = ((Iter1+3) == 1);
			/* Call Kernel LOC_Unknown */
			KerArg1->In = (short int * __restrict__) ((rt_pointerT) (PULP_Dronet_L1_Memory + 0) + 4104*((N_Ti) % 2));
			KerArg1->H = (unsigned short int) (Last?16:19);
			KerArg1->BaseOutFeature = (unsigned short int) (0)*32;
			KerArg1->Filter = (short int * __restrict__) ((rt_pointerT) (PULP_Dronet_L1_Memory + 49744) + 0 + (0)*1600);
			KerArg1->Out = (short int * __restrict__) ((rt_pointerT) (PULP_Dronet_L1_Memory + 22096) + 0);
			KerArg1->TileIndex = (unsigned short int) Iter;
			rt_team_fork(gap8_ncore(), (void *) KerParConv5x5Stride2_fp, (void *) KerArg1);
			N_Ti1++;
			/* End Kernel Iteration Loop on Iter1 space */
		}
		Iter1=1;
		/* Call Kernel LOC_Unknown */
		KerArg2->In = (short int * __restrict__) ((rt_pointerT) (PULP_Dronet_L1_Memory + 22096) + 0);
		KerArg2->H = (unsigned short int) (Last?6:8);
		KerArg2->Out = (short int * __restrict__) ((rt_pointerT) (PULP_Dronet_L1_Memory + 8272) + 6912*((N_Ti) % 2));
		rt_team_fork(gap8_ncore(), (void *) KerParMaxPool2x2Stride2_fp, (void *) KerArg2);
		/* =======================Write Tile=========================================== */
		if (Iter) {
			rt_dma_wait(&DmaW_Evt1);
		}
		rt_dma_memcpy_2d((rt_pointerT) Out + ((Iter)*216),
			(rt_pointerT) (PULP_Dronet_L1_Memory + 8272) + (6912*(N_Ti % 2)), Last?5184:6912, 
			810, Last?162:216, RT_DMA_DIR_LOC2EXT, 0, &DmaW_Evt1);
		/* ===================End Write Tile=========================================== */
		N_Ti++;
		/* End Kernel Iteration Loop on Iter space */
	}
	Iter=4;
	/* =======================Write Last Tile=========================================== */
	rt_dma_wait(&DmaW_Evt1);
	/* ===================End Write Last Tile=========================================== */
}

void ReLU_SW_1(
		short int * __restrict__ In,
		short int * __restrict__ Out,
		Kernel_T *Ker)

{
	/* Local variables used by this kernel */
	rt_dma_copy_t DmaR_Evt1;
	rt_dma_copy_t DmaR_Evt2;
	rt_dma_copy_t DmaW_Evt1;
	int Iter;
	int Last, NextLast, NextNextLast;
	int N_Ti = 0;
	int N_TiIp = 0, InPlane, OutPlane=0;
	KerReLUMaxPool2x2_fpT S_KerArg0, *KerArg0 = &S_KerArg0;

	/* Initialize KerArg, Kernel invariant arguments */
	KerArg0->W = (int) (27);
	KerArg0->H = (int) (15);
	/* =======================Read First Tile=========================================== */
	/* Initial reads in L2, O_DB or O_BUFF */
	rt_dma_memcpy((rt_pointerT) In+(0), (rt_pointerT) (PULP_Dronet_L1_Memory + 0)+0, 25920, RT_DMA_DIR_EXT2LOC, 0, &DmaR_Evt1);
	rt_dma_memcpy((rt_pointerT) Out+(0), (rt_pointerT) (PULP_Dronet_L1_Memory + 25920)+0, 25920, RT_DMA_DIR_EXT2LOC, 0, &DmaR_Evt2);
	/* Wait for BUFF read in L2 */
	rt_dma_wait(&DmaR_Evt1);
	rt_dma_wait(&DmaR_Evt2);
	/* ===================End Read First Tile=========================================== */
	/* Kernel Iteration Loop on Input+Output Planes */
	for (OutPlane=0; OutPlane<32; OutPlane++) {
		int LastOutPlane = ((OutPlane+1) == 32), NextLastOutPlane = ((OutPlane+2) == 32);
		/* Kernel Iteration Loop on Iter space */
		Iter=0; {
			/* Loop Iteration Body on Iter space */
			/* Elaborate Last, Next_Last, Next_Next_Last */
			Last = ((Iter+1) == 1); NextLast = ((Iter+2) == 1); NextNextLast = ((Iter+3) == 1);
			/* Call Kernel LOC_INNER_LOOP */
			KerArg0->In = (short int * __restrict__) ((rt_pointerT) (PULP_Dronet_L1_Memory + 0) + OutPlane*810 + (0)*810);
			KerArg0->Out = (short int * __restrict__) ((rt_pointerT) (PULP_Dronet_L1_Memory + 25920) + OutPlane*810 + (0)*810);
			rt_team_fork(gap8_ncore(), (void *) KerReLU_fp, (void *) KerArg0);
			N_Ti++;
			/* End Kernel Iteration Loop on Iter space */
		}
		Iter=1;
		/* End Kernel Iteration Loop on Input+Output Planes */
	}
	/* =======================Write Last Tile=========================================== */
	rt_dma_memcpy((rt_pointerT) Out + (0),
		(rt_pointerT) (PULP_Dronet_L1_Memory + 25920) + 0, 25920, RT_DMA_DIR_LOC2EXT, 0, &DmaW_Evt1);
	rt_dma_wait(&DmaW_Evt1);
	/* ===================End Write Last Tile=========================================== */
}

void MedParConv_3x3_S2_ReLU_2(
		short int * __restrict__ In,
		short int * __restrict__ Filter,
		short int * __restrict__ Out,
		unsigned int Norm,
		short int * __restrict__ Bias,
		Kernel_T *Ker)

{
	/* Local variables used by this kernel */
	rt_dma_copy_t DmaR_Evt1;
	rt_dma_copy_t DmaR_Evt2;
	rt_dma_copy_t DmaR_Evt3;
	rt_dma_copy_t DmaW_Evt1;
	int Iter;
	int Last, NextLast, NextNextLast;
	int N_Ti = 0;
	int Iter1;
	int Last1, NextLast1, NextNextLast1;
	int N_Ti1 = 0;
	int N_TiIp = 0, InPlane, OutPlane=0;
	KerParSetBias_fp_T S_KerArg0, *KerArg0 = &S_KerArg0;
	KerParConv_fp_T S_KerArg1, *KerArg1 = &S_KerArg1;
	KerParReLUMaxPool_fp_T S_KerArg2, *KerArg2 = &S_KerArg2;

	/* Initialize KerArg, Kernel invariant arguments */
	KerArg0->W = (unsigned short int) (14);
	KerArg0->H = (unsigned short int) (8);
	KerArg0->OutFeatures = (unsigned short int) (32);
	KerArg1->W = (unsigned short int) (27);
	KerArg1->H = (unsigned short int) (15);
	KerArg1->InFeatures = (unsigned short int) (32);
	KerArg1->OutFeatures = (unsigned short int) (32);
	KerArg1->Norm = (Norm);
	KerArg1->TileIndex = (unsigned short int) (32);
	KerArg1->NTile = (unsigned short int) (0);
	KerArg1->Pad = (v4s) (16843009);
	KerArg2->W = (unsigned short int) (14);
	KerArg2->H = (unsigned short int) (8);
	KerArg2->OutFeatures = (unsigned short int) (32);
	/* =======================Read First Tile=========================================== */
	/* Initial reads in L2, O_DB or O_BUFF */
	rt_dma_memcpy((rt_pointerT) In+(0), (rt_pointerT) (PULP_Dronet_L1_Memory + 25664)+0, 25920, RT_DMA_DIR_EXT2LOC, 0, &DmaR_Evt1);
	rt_dma_memcpy((rt_pointerT) Filter+(0), (rt_pointerT) (PULP_Dronet_L1_Memory + 0)+0, 18432, RT_DMA_DIR_EXT2LOC, 0, &DmaR_Evt2);
	rt_dma_memcpy((rt_pointerT) Bias+(0), (rt_pointerT) (PULP_Dronet_L1_Memory + 18432)+0, 64, RT_DMA_DIR_EXT2LOC, 0, &DmaR_Evt3);
	/* Wait for BUFF read in L2 */
	rt_dma_wait(&DmaR_Evt1);
	rt_dma_wait(&DmaR_Evt2);
	rt_dma_wait(&DmaR_Evt3);
	/* ===================End Read First Tile=========================================== */
	/* Kernel Iteration Loop on Iter space */
	Iter=0; {
		/* Loop Iteration Body on Iter space */
		/* Elaborate Last, Next_Last, Next_Next_Last */
		Last = ((Iter+1) == 1); NextLast = ((Iter+2) == 1); NextNextLast = ((Iter+3) == 1);
		/* Call Kernel LOC_Unknown */
		KerArg0->Out = (short int * __restrict__) ((rt_pointerT) (PULP_Dronet_L1_Memory + 18496) + 0 + (0)*7168);
		KerArg0->Bias = (short int * __restrict__) ((rt_pointerT) (PULP_Dronet_L1_Memory + 18432) + 0 + (0)*64);
		rt_team_fork(gap8_ncore(), (void *) KerParSetBias_fp, (void *) KerArg0);
		/* Kernel Iteration Loop on Iter1 space */
		Iter1=0; {
			/* Loop Iteration Body on Iter1 space */
			/* Elaborate Last, Next_Last, Next_Next_Last */
			Last1 = ((Iter1+1) == 1); NextLast1 = ((Iter1+2) == 1); NextNextLast1 = ((Iter1+3) == 1);
			/* Call Kernel LOC_Unknown */
			KerArg1->In = (short int * __restrict__) ((rt_pointerT) (PULP_Dronet_L1_Memory + 25664) + 0 + (0)*25920);
			KerArg1->BaseOutFeature = (unsigned short int) (0)*32;
			KerArg1->Filter = (short int * __restrict__) ((rt_pointerT) (PULP_Dronet_L1_Memory + 0) + 0 + (0)*18432);
			KerArg1->Out = (short int * __restrict__) ((rt_pointerT) (PULP_Dronet_L1_Memory + 18496) + 0 + (0)*7168);
			rt_team_fork(gap8_ncore(), (void *) KerParConv3x3Stride2_fp, (void *) KerArg1);
			N_Ti1++;
			/* End Kernel Iteration Loop on Iter1 space */
		}
		Iter1=1;
		/* Call Kernel LOC_Unknown */
		KerArg2->In = (short int * __restrict__) ((rt_pointerT) (PULP_Dronet_L1_Memory + 18496) + 0 + (0)*7168);
		KerArg2->Out = (short int * __restrict__) ((rt_pointerT) (PULP_Dronet_L1_Memory + 18496) + 0 + (0)*7168);
		rt_team_fork(gap8_ncore(), (void *) KerParReLU_fp, (void *) KerArg2);
		N_Ti++;
		/* End Kernel Iteration Loop on Iter space */
	}
	Iter=1;
	/* =======================Write Last Tile=========================================== */
	rt_dma_memcpy((rt_pointerT) Out + (0),
		(rt_pointerT) (PULP_Dronet_L1_Memory + 18496) + 0, 7168, RT_DMA_DIR_LOC2EXT, 0, &DmaW_Evt1);
	rt_dma_wait(&DmaW_Evt1);
	/* ===================End Write Last Tile=========================================== */
}

void MedParConv_3x3_S1_3(
		short int * __restrict__ In,
		short int * __restrict__ Filter,
		short int * __restrict__ Out,
		unsigned int Norm,
		short int * __restrict__ Bias,
		Kernel_T *Ker)

{
	/* Local variables used by this kernel */
	rt_dma_copy_t DmaR_Evt1;
	rt_dma_copy_t DmaR_Evt2;
	rt_dma_copy_t DmaR_Evt3;
	rt_dma_copy_t DmaW_Evt1;
	int Iter;
	int Last, NextLast, NextNextLast;
	int N_Ti = 0;
	int Iter1;
	int Last1, NextLast1, NextNextLast1;
	int N_Ti1 = 0;
	int N_TiIp = 0, InPlane, OutPlane=0;
	KerParSetBias_fp_T S_KerArg0, *KerArg0 = &S_KerArg0;
	KerParConv_fp_T S_KerArg1, *KerArg1 = &S_KerArg1;

	/* Initialize KerArg, Kernel invariant arguments */
	KerArg0->W = (unsigned short int) (14);
	KerArg0->H = (unsigned short int) (8);
	KerArg0->OutFeatures = (unsigned short int) (32);
	KerArg1->W = (unsigned short int) (14);
	KerArg1->H = (unsigned short int) (8);
	KerArg1->InFeatures = (unsigned short int) (32);
	KerArg1->OutFeatures = (unsigned short int) (32);
	KerArg1->Norm = (Norm);
	KerArg1->TileIndex = (unsigned short int) (32);
	KerArg1->NTile = (unsigned short int) (0);
	KerArg1->Pad = (v4s) (16843009);
	/* =======================Read First Tile=========================================== */
	/* Initial reads in L2, O_DB or O_BUFF */
	rt_dma_memcpy((rt_pointerT) In+(0), (rt_pointerT) (PULP_Dronet_L1_Memory + 25664)+0, 7168, RT_DMA_DIR_EXT2LOC, 0, &DmaR_Evt1);
	rt_dma_memcpy((rt_pointerT) Filter+(0), (rt_pointerT) (PULP_Dronet_L1_Memory + 0)+0, 18432, RT_DMA_DIR_EXT2LOC, 0, &DmaR_Evt2);
	rt_dma_memcpy((rt_pointerT) Bias+(0), (rt_pointerT) (PULP_Dronet_L1_Memory + 18432)+0, 64, RT_DMA_DIR_EXT2LOC, 0, &DmaR_Evt3);
	/* Wait for BUFF read in L2 */
	rt_dma_wait(&DmaR_Evt1);
	rt_dma_wait(&DmaR_Evt2);
	rt_dma_wait(&DmaR_Evt3);
	/* ===================End Read First Tile=========================================== */
	/* Kernel Iteration Loop on Iter space */
	Iter=0; {
		/* Loop Iteration Body on Iter space */
		/* Elaborate Last, Next_Last, Next_Next_Last */
		Last = ((Iter+1) == 1); NextLast = ((Iter+2) == 1); NextNextLast = ((Iter+3) == 1);
		/* Call Kernel LOC_Unknown */
		KerArg0->Out = (short int * __restrict__) ((rt_pointerT) (PULP_Dronet_L1_Memory + 18496) + 0 + (0)*7168);
		KerArg0->Bias = (short int * __restrict__) ((rt_pointerT) (PULP_Dronet_L1_Memory + 18432) + 0 + (0)*64);
		rt_team_fork(gap8_ncore(), (void *) KerParSetBias_fp, (void *) KerArg0);
		/* Kernel Iteration Loop on Iter1 space */
		Iter1=0; {
			/* Loop Iteration Body on Iter1 space */
			/* Elaborate Last, Next_Last, Next_Next_Last */
			Last1 = ((Iter1+1) == 1); NextLast1 = ((Iter1+2) == 1); NextNextLast1 = ((Iter1+3) == 1);
			/* Call Kernel LOC_Unknown */
			KerArg1->In = (short int * __restrict__) ((rt_pointerT) (PULP_Dronet_L1_Memory + 25664) + 0 + (0)*7168);
			KerArg1->BaseOutFeature = (unsigned short int) (0)*32;
			KerArg1->Filter = (short int * __restrict__) ((rt_pointerT) (PULP_Dronet_L1_Memory + 0) + 0 + (0)*18432);
			KerArg1->Out = (short int * __restrict__) ((rt_pointerT) (PULP_Dronet_L1_Memory + 18496) + 0 + (0)*7168);
			rt_team_fork(gap8_ncore(), (void *) KerParConv3x3Stride1_fp, (void *) KerArg1);
			N_Ti1++;
			/* End Kernel Iteration Loop on Iter1 space */
		}
		Iter1=1;
		N_Ti++;
		/* End Kernel Iteration Loop on Iter space */
	}
	Iter=1;
	/* =======================Write Last Tile=========================================== */
	rt_dma_memcpy((rt_pointerT) Out + (0),
		(rt_pointerT) (PULP_Dronet_L1_Memory + 18496) + 0, 7168, RT_DMA_DIR_LOC2EXT, 0, &DmaW_Evt1);
	rt_dma_wait(&DmaW_Evt1);
	/* ===================End Write Last Tile=========================================== */
}

void MedParConv_1x1_S2_4(
		short int * __restrict__ In,
		short int * __restrict__ Filter,
		short int * __restrict__ Out,
		unsigned int Norm,
		short int * __restrict__ Bias,
		Kernel_T *Ker)

{
	/* Local variables used by this kernel */
	rt_dma_copy_t DmaR_Evt1;
	rt_dma_copy_t DmaR_Evt2;
	rt_dma_copy_t DmaR_Evt3;
	rt_dma_copy_t DmaW_Evt1;
	int Iter;
	int Last, NextLast, NextNextLast;
	int N_Ti = 0;
	int Iter1;
	int Last1, NextLast1, NextNextLast1;
	int N_Ti1 = 0;
	int N_TiIp = 0, InPlane, OutPlane=0;
	KerParSetBias_fp_T S_KerArg0, *KerArg0 = &S_KerArg0;
	KerParConv_fp_T S_KerArg1, *KerArg1 = &S_KerArg1;

	/* Initialize KerArg, Kernel invariant arguments */
	KerArg0->W = (unsigned short int) (14);
	KerArg0->H = (unsigned short int) (8);
	KerArg0->OutFeatures = (unsigned short int) (32);
	KerArg1->W = (unsigned short int) (27);
	KerArg1->H = (unsigned short int) (15);
	KerArg1->InFeatures = (unsigned short int) (32);
	KerArg1->OutFeatures = (unsigned short int) (32);
	KerArg1->Norm = (Norm);
	KerArg1->TileIndex = (unsigned short int) (32);
	KerArg1->NTile = (unsigned short int) (0);
	KerArg1->Pad = (v4s) (0);
	/* =======================Read First Tile=========================================== */
	/* Initial reads in L2, O_DB or O_BUFF */
	rt_dma_memcpy((rt_pointerT) In+(0), (rt_pointerT) (PULP_Dronet_L1_Memory + 9280)+0, 25920, RT_DMA_DIR_EXT2LOC, 0, &DmaR_Evt1);
	rt_dma_memcpy((rt_pointerT) Filter+(0), (rt_pointerT) (PULP_Dronet_L1_Memory + 0)+0, 2048, RT_DMA_DIR_EXT2LOC, 0, &DmaR_Evt2);
	rt_dma_memcpy((rt_pointerT) Bias+(0), (rt_pointerT) (PULP_Dronet_L1_Memory + 2048)+0, 64, RT_DMA_DIR_EXT2LOC, 0, &DmaR_Evt3);
	/* Wait for BUFF read in L2 */
	rt_dma_wait(&DmaR_Evt1);
	rt_dma_wait(&DmaR_Evt2);
	rt_dma_wait(&DmaR_Evt3);
	/* ===================End Read First Tile=========================================== */
	/* Kernel Iteration Loop on Iter space */
	Iter=0; {
		/* Loop Iteration Body on Iter space */
		/* Elaborate Last, Next_Last, Next_Next_Last */
		Last = ((Iter+1) == 1); NextLast = ((Iter+2) == 1); NextNextLast = ((Iter+3) == 1);
		/* Call Kernel LOC_Unknown */
		KerArg0->Out = (short int * __restrict__) ((rt_pointerT) (PULP_Dronet_L1_Memory + 2112) + 0 + (0)*7168);
		KerArg0->Bias = (short int * __restrict__) ((rt_pointerT) (PULP_Dronet_L1_Memory + 2048) + 0 + (0)*64);
		rt_team_fork(gap8_ncore(), (void *) KerParSetBias_fp, (void *) KerArg0);
		/* Kernel Iteration Loop on Iter1 space */
		Iter1=0; {
			/* Loop Iteration Body on Iter1 space */
			/* Elaborate Last, Next_Last, Next_Next_Last */
			Last1 = ((Iter1+1) == 1); NextLast1 = ((Iter1+2) == 1); NextNextLast1 = ((Iter1+3) == 1);
			/* Call Kernel LOC_Unknown */
			KerArg1->In = (short int * __restrict__) ((rt_pointerT) (PULP_Dronet_L1_Memory + 9280) + 0 + (0)*25920);
			KerArg1->BaseOutFeature = (unsigned short int) (0)*32;
			KerArg1->Filter = (short int * __restrict__) ((rt_pointerT) (PULP_Dronet_L1_Memory + 0) + 0 + (0)*2048);
			KerArg1->Out = (short int * __restrict__) ((rt_pointerT) (PULP_Dronet_L1_Memory + 2112) + 0 + (0)*7168);
			rt_team_fork(gap8_ncore(), (void *) KerParConv1x1Stride2_fp, (void *) KerArg1);
			N_Ti1++;
			/* End Kernel Iteration Loop on Iter1 space */
		}
		Iter1=1;
		N_Ti++;
		/* End Kernel Iteration Loop on Iter space */
	}
	Iter=1;
	/* =======================Write Last Tile=========================================== */
	rt_dma_memcpy((rt_pointerT) Out + (0),
		(rt_pointerT) (PULP_Dronet_L1_Memory + 2112) + 0, 7168, RT_DMA_DIR_LOC2EXT, 0, &DmaW_Evt1);
	rt_dma_wait(&DmaW_Evt1);
	/* ===================End Write Last Tile=========================================== */
}

void AddFeatureMaps_SW_1(
		short int * __restrict__ In,
		short int * __restrict__ Out,
		Kernel_T *Ker)

{
	/* Local variables used by this kernel */
	rt_dma_copy_t DmaR_Evt1;
	rt_dma_copy_t DmaR_Evt2;
	rt_dma_copy_t DmaW_Evt1;
	int Iter;
	int Last, NextLast, NextNextLast;
	int N_Ti = 0;
	int N_TiIp = 0, InPlane, OutPlane=0;
	KerAddFM_fpT S_KerArg0, *KerArg0 = &S_KerArg0;

	/* Initialize KerArg, Kernel invariant arguments */
	KerArg0->W = (int) (14);
	KerArg0->H = (int) (8);
	/* =======================Read First Tile=========================================== */
	/* Initial reads in L2, O_DB or O_BUFF */
	rt_dma_memcpy((rt_pointerT) In+(0), (rt_pointerT) (PULP_Dronet_L1_Memory + 0)+0, 7168, RT_DMA_DIR_EXT2LOC, 0, &DmaR_Evt1);
	rt_dma_memcpy((rt_pointerT) Out+(0), (rt_pointerT) (PULP_Dronet_L1_Memory + 7168)+0, 7168, RT_DMA_DIR_EXT2LOC, 0, &DmaR_Evt2);
	/* Wait for BUFF read in L2 */
	rt_dma_wait(&DmaR_Evt1);
	rt_dma_wait(&DmaR_Evt2);
	/* ===================End Read First Tile=========================================== */
	/* Kernel Iteration Loop on Input+Output Planes */
	for (OutPlane=0; OutPlane<32; OutPlane++) {
		int LastOutPlane = ((OutPlane+1) == 32), NextLastOutPlane = ((OutPlane+2) == 32);
		/* Kernel Iteration Loop on Iter space */
		Iter=0; {
			/* Loop Iteration Body on Iter space */
			/* Elaborate Last, Next_Last, Next_Next_Last */
			Last = ((Iter+1) == 1); NextLast = ((Iter+2) == 1); NextNextLast = ((Iter+3) == 1);
			/* Call Kernel LOC_INNER_LOOP */
			KerArg0->In = (Word16 * __restrict__) ((rt_pointerT) (PULP_Dronet_L1_Memory + 0) + OutPlane*224 + (0)*224);
			KerArg0->Out = (Word16 * __restrict__) ((rt_pointerT) (PULP_Dronet_L1_Memory + 7168) + OutPlane*224 + (0)*224);
			rt_team_fork(gap8_ncore(), (void *) KerAddFM_fp, (void *) KerArg0);
			N_Ti++;
			/* End Kernel Iteration Loop on Iter space */
		}
		Iter=1;
		/* End Kernel Iteration Loop on Input+Output Planes */
	}
	/* =======================Write Last Tile=========================================== */
	rt_dma_memcpy((rt_pointerT) Out + (0),
		(rt_pointerT) (PULP_Dronet_L1_Memory + 7168) + 0, 7168, RT_DMA_DIR_LOC2EXT, 0, &DmaW_Evt1);
	rt_dma_wait(&DmaW_Evt1);
	/* ===================End Write Last Tile=========================================== */
}

void ReLU_SW_2(
		short int * __restrict__ In,
		short int * __restrict__ Out,
		Kernel_T *Ker)

{
	/* Local variables used by this kernel */
	rt_dma_copy_t DmaR_Evt1;
	rt_dma_copy_t DmaR_Evt2;
	rt_dma_copy_t DmaW_Evt1;
	int Iter;
	int Last, NextLast, NextNextLast;
	int N_Ti = 0;
	int N_TiIp = 0, InPlane, OutPlane=0;
	KerReLUMaxPool2x2_fpT S_KerArg0, *KerArg0 = &S_KerArg0;

	/* Initialize KerArg, Kernel invariant arguments */
	KerArg0->W = (int) (14);
	KerArg0->H = (int) (8);
	/* =======================Read First Tile=========================================== */
	/* Initial reads in L2, O_DB or O_BUFF */
	rt_dma_memcpy((rt_pointerT) In+(0), (rt_pointerT) (PULP_Dronet_L1_Memory + 0)+0, 7168, RT_DMA_DIR_EXT2LOC, 0, &DmaR_Evt1);
	rt_dma_memcpy((rt_pointerT) Out+(0), (rt_pointerT) (PULP_Dronet_L1_Memory + 7168)+0, 7168, RT_DMA_DIR_EXT2LOC, 0, &DmaR_Evt2);
	/* Wait for BUFF read in L2 */
	rt_dma_wait(&DmaR_Evt1);
	rt_dma_wait(&DmaR_Evt2);
	/* ===================End Read First Tile=========================================== */
	/* Kernel Iteration Loop on Input+Output Planes */
	for (OutPlane=0; OutPlane<32; OutPlane++) {
		int LastOutPlane = ((OutPlane+1) == 32), NextLastOutPlane = ((OutPlane+2) == 32);
		/* Kernel Iteration Loop on Iter space */
		Iter=0; {
			/* Loop Iteration Body on Iter space */
			/* Elaborate Last, Next_Last, Next_Next_Last */
			Last = ((Iter+1) == 1); NextLast = ((Iter+2) == 1); NextNextLast = ((Iter+3) == 1);
			/* Call Kernel LOC_INNER_LOOP */
			KerArg0->In = (short int * __restrict__) ((rt_pointerT) (PULP_Dronet_L1_Memory + 0) + OutPlane*224 + (0)*224);
			KerArg0->Out = (short int * __restrict__) ((rt_pointerT) (PULP_Dronet_L1_Memory + 7168) + OutPlane*224 + (0)*224);
			rt_team_fork(gap8_ncore(), (void *) KerReLU_fp, (void *) KerArg0);
			N_Ti++;
			/* End Kernel Iteration Loop on Iter space */
		}
		Iter=1;
		/* End Kernel Iteration Loop on Input+Output Planes */
	}
	/* =======================Write Last Tile=========================================== */
	rt_dma_memcpy((rt_pointerT) Out + (0),
		(rt_pointerT) (PULP_Dronet_L1_Memory + 7168) + 0, 7168, RT_DMA_DIR_LOC2EXT, 0, &DmaW_Evt1);
	rt_dma_wait(&DmaW_Evt1);
	/* ===================End Write Last Tile=========================================== */
}

void MedParConv_3x3_S2_ReLU_5(
		short int * __restrict__ In,
		short int * __restrict__ Filter,
		short int * __restrict__ Out,
		unsigned int Norm,
		short int * __restrict__ Bias,
		Kernel_T *Ker)

{
	/* Local variables used by this kernel */
	rt_dma_copy_t DmaR_Evt1;
	rt_dma_copy_t DmaR_Evt2;
	rt_dma_copy_t DmaR_Evt3;
	rt_dma_copy_t DmaW_Evt1;
	int Iter;
	int Last, NextLast, NextNextLast;
	int N_Ti = 0;
	int Iter1;
	int Last1, NextLast1, NextNextLast1;
	int N_Ti1 = 0;
	int N_TiIp = 0, InPlane, OutPlane=0;
	KerParSetBias_fp_T S_KerArg0, *KerArg0 = &S_KerArg0;
	KerParConv_fp_T S_KerArg1, *KerArg1 = &S_KerArg1;
	KerParReLUMaxPool_fp_T S_KerArg2, *KerArg2 = &S_KerArg2;

	/* Initialize KerArg, Kernel invariant arguments */
	KerArg0->W = (unsigned short int) (7);
	KerArg0->H = (unsigned short int) (4);
	KerArg0->OutFeatures = (unsigned short int) (64);
	KerArg1->W = (unsigned short int) (14);
	KerArg1->H = (unsigned short int) (8);
	KerArg1->InFeatures = (unsigned short int) (32);
	KerArg1->OutFeatures = (unsigned short int) (64);
	KerArg1->Norm = (Norm);
	KerArg1->TileIndex = (unsigned short int) (32);
	KerArg1->NTile = (unsigned short int) (0);
	KerArg1->Pad = (v4s) (16843009);
	KerArg2->W = (unsigned short int) (7);
	KerArg2->H = (unsigned short int) (4);
	KerArg2->OutFeatures = (unsigned short int) (64);
	/* =======================Read First Tile=========================================== */
	/* Initial reads in L2, O_DB or O_BUFF */
	rt_dma_memcpy((rt_pointerT) In+(0), (rt_pointerT) (PULP_Dronet_L1_Memory + 40576)+0, 7168, RT_DMA_DIR_EXT2LOC, 0, &DmaR_Evt1);
	rt_dma_memcpy((rt_pointerT) Filter+(0), (rt_pointerT) (PULP_Dronet_L1_Memory + 0)+0, 36864, RT_DMA_DIR_EXT2LOC, 0, &DmaR_Evt2);
	rt_dma_memcpy((rt_pointerT) Bias+(0), (rt_pointerT) (PULP_Dronet_L1_Memory + 36864)+0, 128, RT_DMA_DIR_EXT2LOC, 0, &DmaR_Evt3);
	/* Wait for BUFF read in L2 */
	rt_dma_wait(&DmaR_Evt1);
	rt_dma_wait(&DmaR_Evt2);
	rt_dma_wait(&DmaR_Evt3);
	/* ===================End Read First Tile=========================================== */
	/* Kernel Iteration Loop on Iter space */
	Iter=0; {
		/* Loop Iteration Body on Iter space */
		/* Elaborate Last, Next_Last, Next_Next_Last */
		Last = ((Iter+1) == 1); NextLast = ((Iter+2) == 1); NextNextLast = ((Iter+3) == 1);
		/* Call Kernel LOC_Unknown */
		KerArg0->Out = (short int * __restrict__) ((rt_pointerT) (PULP_Dronet_L1_Memory + 36992) + 0 + (0)*3584);
		KerArg0->Bias = (short int * __restrict__) ((rt_pointerT) (PULP_Dronet_L1_Memory + 36864) + 0 + (0)*128);
		rt_team_fork(gap8_ncore(), (void *) KerParSetBias_fp, (void *) KerArg0);
		/* Kernel Iteration Loop on Iter1 space */
		Iter1=0; {
			/* Loop Iteration Body on Iter1 space */
			/* Elaborate Last, Next_Last, Next_Next_Last */
			Last1 = ((Iter1+1) == 1); NextLast1 = ((Iter1+2) == 1); NextNextLast1 = ((Iter1+3) == 1);
			/* Call Kernel LOC_Unknown */
			KerArg1->In = (short int * __restrict__) ((rt_pointerT) (PULP_Dronet_L1_Memory + 40576) + 0 + (0)*7168);
			KerArg1->BaseOutFeature = (unsigned short int) (0)*32;
			KerArg1->Filter = (short int * __restrict__) ((rt_pointerT) (PULP_Dronet_L1_Memory + 0) + 0 + (0)*36864);
			KerArg1->Out = (short int * __restrict__) ((rt_pointerT) (PULP_Dronet_L1_Memory + 36992) + 0 + (0)*3584);
			rt_team_fork(gap8_ncore(), (void *) KerParConv3x3Stride2_fp, (void *) KerArg1);
			N_Ti1++;
			/* End Kernel Iteration Loop on Iter1 space */
		}
		Iter1=1;
		/* Call Kernel LOC_Unknown */
		KerArg2->In = (short int * __restrict__) ((rt_pointerT) (PULP_Dronet_L1_Memory + 36992) + 0 + (0)*3584);
		KerArg2->Out = (short int * __restrict__) ((rt_pointerT) (PULP_Dronet_L1_Memory + 36992) + 0 + (0)*3584);
		rt_team_fork(gap8_ncore(), (void *) KerParReLU_fp, (void *) KerArg2);
		N_Ti++;
		/* End Kernel Iteration Loop on Iter space */
	}
	Iter=1;
	/* =======================Write Last Tile=========================================== */
	rt_dma_memcpy((rt_pointerT) Out + (0),
		(rt_pointerT) (PULP_Dronet_L1_Memory + 36992) + 0, 3584, RT_DMA_DIR_LOC2EXT, 0, &DmaW_Evt1);
	rt_dma_wait(&DmaW_Evt1);
	/* ===================End Write Last Tile=========================================== */
}

void MedParConv_3x3_S1_6(
		short int * __restrict__ In,
		short int * __restrict__ Filter,
		short int * __restrict__ Out,
		unsigned int Norm,
		short int * __restrict__ Bias,
		Kernel_T *Ker)

{
	/* Local variables used by this kernel */
	rt_dma_copy_t DmaR_Evt1;
	rt_dma_copy_t DmaR_Evt2;
	rt_dma_copy_t DmaR_Evt3;
	rt_dma_copy_t DmaW_Evt1;
	int Iter;
	int Last, NextLast, NextNextLast;
	int N_Ti = 0;
	int Iter1;
	int Last1, NextLast1, NextNextLast1;
	int N_Ti1 = 0;
	int N_TiIp = 0, InPlane, OutPlane=0;
	KerParSetBias_fp_T S_KerArg0, *KerArg0 = &S_KerArg0;
	KerParConv_fp_T S_KerArg1, *KerArg1 = &S_KerArg1;

	/* Initialize KerArg, Kernel invariant arguments */
	KerArg0->W = (unsigned short int) (7);
	KerArg0->H = (unsigned short int) (4);
	KerArg0->OutFeatures = (unsigned short int) (16);
	KerArg1->W = (unsigned short int) (7);
	KerArg1->H = (unsigned short int) (4);
	KerArg1->InFeatures = (unsigned short int) (64);
	KerArg1->OutFeatures = (unsigned short int) (16);
	KerArg1->Norm = (Norm);
	KerArg1->TileIndex = (unsigned short int) (64);
	KerArg1->NTile = (unsigned short int) (0);
	KerArg1->Pad = (v4s) (16843009);
	/* =======================Read First Tile=========================================== */
	/* Initial reads in L2, O_DB or O_BUFF */
	rt_dma_memcpy((rt_pointerT) In+(0), (rt_pointerT) (PULP_Dronet_L1_Memory + 38784)+0, 3584, RT_DMA_DIR_EXT2LOC, 0, &DmaR_Evt1);
	rt_dma_memcpy((rt_pointerT) Filter+(0), (rt_pointerT) (PULP_Dronet_L1_Memory + 0)+0, 18432, RT_DMA_DIR_EXT2LOC, 0, &DmaR_Evt2);
	rt_dma_memcpy((rt_pointerT) Bias+(0), (rt_pointerT) (PULP_Dronet_L1_Memory + 36864)+0, 128, RT_DMA_DIR_EXT2LOC, 0, &DmaR_Evt3);
	/* Wait for BUFF read in L2 */
	rt_dma_wait(&DmaR_Evt1);
	rt_dma_wait(&DmaR_Evt3);
	/* ===================End Read First Tile=========================================== */
	/* Kernel Iteration Loop on Iter space */
	for (Iter=0; Iter<4; Iter++) {
		/* Loop Iteration Body on Iter space */
		/* Elaborate Last, Next_Last, Next_Next_Last */
		Last = ((Iter+1) == 4); NextLast = ((Iter+2) == 4); NextNextLast = ((Iter+3) == 4);
		/* =======================Read Next Tile=========================================== */
		rt_dma_wait(&DmaR_Evt2);
		if (!Last) {
			rt_dma_memcpy((rt_pointerT) Filter + ((Iter+1)*18432),
					(rt_pointerT) (PULP_Dronet_L1_Memory + 0) + (18432*((N_Ti+1) % 2)), 18432, RT_DMA_DIR_EXT2LOC, 0, &DmaR_Evt2);
		}
		/* ===================End Read Next Tile=========================================== */
		/* Call Kernel LOC_Unknown */
		KerArg0->Out = (short int * __restrict__) ((rt_pointerT) (PULP_Dronet_L1_Memory + 36992) + 896*((N_Ti) % 2));
		KerArg0->Bias = (short int * __restrict__) ((rt_pointerT) (PULP_Dronet_L1_Memory + 36864) + 0 + (Iter)*32);
		rt_team_fork(gap8_ncore(), (void *) KerParSetBias_fp, (void *) KerArg0);
		/* Kernel Iteration Loop on Iter1 space */
		Iter1=0; {
			/* Loop Iteration Body on Iter1 space */
			/* Elaborate Last, Next_Last, Next_Next_Last */
			Last1 = ((Iter1+1) == 1); NextLast1 = ((Iter1+2) == 1); NextNextLast1 = ((Iter1+3) == 1);
			/* Call Kernel LOC_Unknown */
			KerArg1->In = (short int * __restrict__) ((rt_pointerT) (PULP_Dronet_L1_Memory + 38784) + 0 + (0)*3584);
			KerArg1->BaseOutFeature = (unsigned short int) (0)*64;
			KerArg1->Filter = (short int * __restrict__) ((rt_pointerT) (PULP_Dronet_L1_Memory + 0) + 18432*((N_Ti) % 2));
			KerArg1->Out = (short int * __restrict__) ((rt_pointerT) (PULP_Dronet_L1_Memory + 36992) + 896*((N_Ti) % 2));
			rt_team_fork(gap8_ncore(), (void *) KerParConv3x3Stride1_fp, (void *) KerArg1);
			N_Ti1++;
			/* End Kernel Iteration Loop on Iter1 space */
		}
		Iter1=1;
		/* =======================Write Tile=========================================== */
		if (Iter) {
			rt_dma_wait(&DmaW_Evt1);
		}
		rt_dma_memcpy((rt_pointerT) Out + ((Iter)*896),
			(rt_pointerT) (PULP_Dronet_L1_Memory + 36992) + (896*(N_Ti % 2)), 896, RT_DMA_DIR_LOC2EXT, 0, &DmaW_Evt1);
		/* ===================End Write Tile=========================================== */
		N_Ti++;
		/* End Kernel Iteration Loop on Iter space */
	}
	Iter=4;
	/* =======================Write Last Tile=========================================== */
	rt_dma_wait(&DmaW_Evt1);
	/* ===================End Write Last Tile=========================================== */
}

void MedParConv_1x1_S2_7(
		short int * __restrict__ In,
		short int * __restrict__ Filter,
		short int * __restrict__ Out,
		unsigned int Norm,
		short int * __restrict__ Bias,
		Kernel_T *Ker)

{
	/* Local variables used by this kernel */
	rt_dma_copy_t DmaR_Evt1;
	rt_dma_copy_t DmaR_Evt2;
	rt_dma_copy_t DmaR_Evt3;
	rt_dma_copy_t DmaW_Evt1;
	int Iter;
	int Last, NextLast, NextNextLast;
	int N_Ti = 0;
	int Iter1;
	int Last1, NextLast1, NextNextLast1;
	int N_Ti1 = 0;
	int N_TiIp = 0, InPlane, OutPlane=0;
	KerParSetBias_fp_T S_KerArg0, *KerArg0 = &S_KerArg0;
	KerParConv_fp_T S_KerArg1, *KerArg1 = &S_KerArg1;

	/* Initialize KerArg, Kernel invariant arguments */
	KerArg0->W = (unsigned short int) (7);
	KerArg0->H = (unsigned short int) (4);
	KerArg0->OutFeatures = (unsigned short int) (64);
	KerArg1->W = (unsigned short int) (14);
	KerArg1->H = (unsigned short int) (8);
	KerArg1->InFeatures = (unsigned short int) (32);
	KerArg1->OutFeatures = (unsigned short int) (64);
	KerArg1->Norm = (Norm);
	KerArg1->TileIndex = (unsigned short int) (32);
	KerArg1->NTile = (unsigned short int) (0);
	KerArg1->Pad = (v4s) (0);
	/* =======================Read First Tile=========================================== */
	/* Initial reads in L2, O_DB or O_BUFF */
	rt_dma_memcpy((rt_pointerT) In+(0), (rt_pointerT) (PULP_Dronet_L1_Memory + 7808)+0, 7168, RT_DMA_DIR_EXT2LOC, 0, &DmaR_Evt1);
	rt_dma_memcpy((rt_pointerT) Filter+(0), (rt_pointerT) (PULP_Dronet_L1_Memory + 0)+0, 4096, RT_DMA_DIR_EXT2LOC, 0, &DmaR_Evt2);
	rt_dma_memcpy((rt_pointerT) Bias+(0), (rt_pointerT) (PULP_Dronet_L1_Memory + 4096)+0, 128, RT_DMA_DIR_EXT2LOC, 0, &DmaR_Evt3);
	/* Wait for BUFF read in L2 */
	rt_dma_wait(&DmaR_Evt1);
	rt_dma_wait(&DmaR_Evt2);
	rt_dma_wait(&DmaR_Evt3);
	/* ===================End Read First Tile=========================================== */
	/* Kernel Iteration Loop on Iter space */
	Iter=0; {
		/* Loop Iteration Body on Iter space */
		/* Elaborate Last, Next_Last, Next_Next_Last */
		Last = ((Iter+1) == 1); NextLast = ((Iter+2) == 1); NextNextLast = ((Iter+3) == 1);
		/* Call Kernel LOC_Unknown */
		KerArg0->Out = (short int * __restrict__) ((rt_pointerT) (PULP_Dronet_L1_Memory + 4224) + 0 + (0)*3584);
		KerArg0->Bias = (short int * __restrict__) ((rt_pointerT) (PULP_Dronet_L1_Memory + 4096) + 0 + (0)*128);
		rt_team_fork(gap8_ncore(), (void *) KerParSetBias_fp, (void *) KerArg0);
		/* Kernel Iteration Loop on Iter1 space */
		Iter1=0; {
			/* Loop Iteration Body on Iter1 space */
			/* Elaborate Last, Next_Last, Next_Next_Last */
			Last1 = ((Iter1+1) == 1); NextLast1 = ((Iter1+2) == 1); NextNextLast1 = ((Iter1+3) == 1);
			/* Call Kernel LOC_Unknown */
			KerArg1->In = (short int * __restrict__) ((rt_pointerT) (PULP_Dronet_L1_Memory + 7808) + 0 + (0)*7168);
			KerArg1->BaseOutFeature = (unsigned short int) (0)*32;
			KerArg1->Filter = (short int * __restrict__) ((rt_pointerT) (PULP_Dronet_L1_Memory + 0) + 0 + (0)*4096);
			KerArg1->Out = (short int * __restrict__) ((rt_pointerT) (PULP_Dronet_L1_Memory + 4224) + 0 + (0)*3584);
			rt_team_fork(gap8_ncore(), (void *) KerParConv1x1Stride2_fp, (void *) KerArg1);
			N_Ti1++;
			/* End Kernel Iteration Loop on Iter1 space */
		}
		Iter1=1;
		N_Ti++;
		/* End Kernel Iteration Loop on Iter space */
	}
	Iter=1;
	/* =======================Write Last Tile=========================================== */
	rt_dma_memcpy((rt_pointerT) Out + (0),
		(rt_pointerT) (PULP_Dronet_L1_Memory + 4224) + 0, 3584, RT_DMA_DIR_LOC2EXT, 0, &DmaW_Evt1);
	rt_dma_wait(&DmaW_Evt1);
	/* ===================End Write Last Tile=========================================== */
}

void AddFeatureMaps_SW_2(
		short int * __restrict__ In,
		short int * __restrict__ Out,
		Kernel_T *Ker)

{
	/* Local variables used by this kernel */
	rt_dma_copy_t DmaR_Evt1;
	rt_dma_copy_t DmaR_Evt2;
	rt_dma_copy_t DmaW_Evt1;
	int Iter;
	int Last, NextLast, NextNextLast;
	int N_Ti = 0;
	int N_TiIp = 0, InPlane, OutPlane=0;
	KerAddFM_fpT S_KerArg0, *KerArg0 = &S_KerArg0;

	/* Initialize KerArg, Kernel invariant arguments */
	KerArg0->W = (int) (7);
	KerArg0->H = (int) (4);
	/* =======================Read First Tile=========================================== */
	/* Initial reads in L2, O_DB or O_BUFF */
	rt_dma_memcpy((rt_pointerT) In+(0), (rt_pointerT) (PULP_Dronet_L1_Memory + 0)+0, 3584, RT_DMA_DIR_EXT2LOC, 0, &DmaR_Evt1);
	rt_dma_memcpy((rt_pointerT) Out+(0), (rt_pointerT) (PULP_Dronet_L1_Memory + 3584)+0, 3584, RT_DMA_DIR_EXT2LOC, 0, &DmaR_Evt2);
	/* Wait for BUFF read in L2 */
	rt_dma_wait(&DmaR_Evt1);
	rt_dma_wait(&DmaR_Evt2);
	/* ===================End Read First Tile=========================================== */
	/* Kernel Iteration Loop on Input+Output Planes */
	for (OutPlane=0; OutPlane<64; OutPlane++) {
		int LastOutPlane = ((OutPlane+1) == 64), NextLastOutPlane = ((OutPlane+2) == 64);
		/* Kernel Iteration Loop on Iter space */
		Iter=0; {
			/* Loop Iteration Body on Iter space */
			/* Elaborate Last, Next_Last, Next_Next_Last */
			Last = ((Iter+1) == 1); NextLast = ((Iter+2) == 1); NextNextLast = ((Iter+3) == 1);
			/* Call Kernel LOC_INNER_LOOP */
			KerArg0->In = (Word16 * __restrict__) ((rt_pointerT) (PULP_Dronet_L1_Memory + 0) + OutPlane*56 + (0)*56);
			KerArg0->Out = (Word16 * __restrict__) ((rt_pointerT) (PULP_Dronet_L1_Memory + 3584) + OutPlane*56 + (0)*56);
			rt_team_fork(gap8_ncore(), (void *) KerAddFM_fp, (void *) KerArg0);
			N_Ti++;
			/* End Kernel Iteration Loop on Iter space */
		}
		Iter=1;
		/* End Kernel Iteration Loop on Input+Output Planes */
	}
	/* =======================Write Last Tile=========================================== */
	rt_dma_memcpy((rt_pointerT) Out + (0),
		(rt_pointerT) (PULP_Dronet_L1_Memory + 3584) + 0, 3584, RT_DMA_DIR_LOC2EXT, 0, &DmaW_Evt1);
	rt_dma_wait(&DmaW_Evt1);
	/* ===================End Write Last Tile=========================================== */
}

void ReLU_SW_3(
		short int * __restrict__ In,
		short int * __restrict__ Out,
		Kernel_T *Ker)

{
	/* Local variables used by this kernel */
	rt_dma_copy_t DmaR_Evt1;
	rt_dma_copy_t DmaR_Evt2;
	rt_dma_copy_t DmaW_Evt1;
	int Iter;
	int Last, NextLast, NextNextLast;
	int N_Ti = 0;
	int N_TiIp = 0, InPlane, OutPlane=0;
	KerReLUMaxPool2x2_fpT S_KerArg0, *KerArg0 = &S_KerArg0;

	/* Initialize KerArg, Kernel invariant arguments */
	KerArg0->W = (int) (7);
	KerArg0->H = (int) (4);
	/* =======================Read First Tile=========================================== */
	/* Initial reads in L2, O_DB or O_BUFF */
	rt_dma_memcpy((rt_pointerT) In+(0), (rt_pointerT) (PULP_Dronet_L1_Memory + 0)+0, 3584, RT_DMA_DIR_EXT2LOC, 0, &DmaR_Evt1);
	rt_dma_memcpy((rt_pointerT) Out+(0), (rt_pointerT) (PULP_Dronet_L1_Memory + 3584)+0, 3584, RT_DMA_DIR_EXT2LOC, 0, &DmaR_Evt2);
	/* Wait for BUFF read in L2 */
	rt_dma_wait(&DmaR_Evt1);
	rt_dma_wait(&DmaR_Evt2);
	/* ===================End Read First Tile=========================================== */
	/* Kernel Iteration Loop on Input+Output Planes */
	for (OutPlane=0; OutPlane<64; OutPlane++) {
		int LastOutPlane = ((OutPlane+1) == 64), NextLastOutPlane = ((OutPlane+2) == 64);
		/* Kernel Iteration Loop on Iter space */
		Iter=0; {
			/* Loop Iteration Body on Iter space */
			/* Elaborate Last, Next_Last, Next_Next_Last */
			Last = ((Iter+1) == 1); NextLast = ((Iter+2) == 1); NextNextLast = ((Iter+3) == 1);
			/* Call Kernel LOC_INNER_LOOP */
			KerArg0->In = (short int * __restrict__) ((rt_pointerT) (PULP_Dronet_L1_Memory + 0) + OutPlane*56 + (0)*56);
			KerArg0->Out = (short int * __restrict__) ((rt_pointerT) (PULP_Dronet_L1_Memory + 3584) + OutPlane*56 + (0)*56);
			rt_team_fork(gap8_ncore(), (void *) KerReLU_fp, (void *) KerArg0);
			N_Ti++;
			/* End Kernel Iteration Loop on Iter space */
		}
		Iter=1;
		/* End Kernel Iteration Loop on Input+Output Planes */
	}
	/* =======================Write Last Tile=========================================== */
	rt_dma_memcpy((rt_pointerT) Out + (0),
		(rt_pointerT) (PULP_Dronet_L1_Memory + 3584) + 0, 3584, RT_DMA_DIR_LOC2EXT, 0, &DmaW_Evt1);
	rt_dma_wait(&DmaW_Evt1);
	/* ===================End Write Last Tile=========================================== */
}

void MedParConv_3x3_S2_ReLU_8(
		short int * __restrict__ In,
		short int * __restrict__ Filter,
		short int * __restrict__ Out,
		unsigned int Norm,
		short int * __restrict__ Bias,
		Kernel_T *Ker)

{
	/* Local variables used by this kernel */
	rt_dma_copy_t DmaR_Evt1;
	rt_dma_copy_t DmaR_Evt2;
	rt_dma_copy_t DmaR_Evt3;
	rt_dma_copy_t DmaW_Evt1;
	int Iter;
	int Last, NextLast, NextNextLast;
	int N_Ti = 0;
	int Iter1;
	int Last1, NextLast1, NextNextLast1;
	int N_Ti1 = 0;
	int N_TiIp = 0, InPlane, OutPlane=0;
	KerParSetBias_fp_T S_KerArg0, *KerArg0 = &S_KerArg0;
	KerParConv_fp_T S_KerArg1, *KerArg1 = &S_KerArg1;
	KerParReLUMaxPool_fp_T S_KerArg2, *KerArg2 = &S_KerArg2;

	/* Initialize KerArg, Kernel invariant arguments */
	KerArg0->W = (unsigned short int) (4);
	KerArg0->H = (unsigned short int) (2);
	KerArg0->OutFeatures = (unsigned short int) (16);
	KerArg1->W = (unsigned short int) (7);
	KerArg1->H = (unsigned short int) (4);
	KerArg1->InFeatures = (unsigned short int) (64);
	KerArg1->OutFeatures = (unsigned short int) (16);
	KerArg1->Norm = (Norm);
	KerArg1->TileIndex = (unsigned short int) (64);
	KerArg1->NTile = (unsigned short int) (0);
	KerArg1->Pad = (v4s) (16843009);
	KerArg2->W = (unsigned short int) (4);
	KerArg2->H = (unsigned short int) (2);
	KerArg2->OutFeatures = (unsigned short int) (16);
	/* =======================Read First Tile=========================================== */
	/* Initial reads in L2, O_DB or O_BUFF */
	rt_dma_memcpy((rt_pointerT) In+(0), (rt_pointerT) (PULP_Dronet_L1_Memory + 37632)+0, 3584, RT_DMA_DIR_EXT2LOC, 0, &DmaR_Evt1);
	rt_dma_memcpy((rt_pointerT) Filter+(0), (rt_pointerT) (PULP_Dronet_L1_Memory + 0)+0, 18432, RT_DMA_DIR_EXT2LOC, 0, &DmaR_Evt2);
	rt_dma_memcpy((rt_pointerT) Bias+(0), (rt_pointerT) (PULP_Dronet_L1_Memory + 36864)+0, 256, RT_DMA_DIR_EXT2LOC, 0, &DmaR_Evt3);
	/* Wait for BUFF read in L2 */
	rt_dma_wait(&DmaR_Evt1);
	rt_dma_wait(&DmaR_Evt3);
	/* ===================End Read First Tile=========================================== */
	/* Kernel Iteration Loop on Iter space */
	for (Iter=0; Iter<8; Iter++) {
		/* Loop Iteration Body on Iter space */
		/* Elaborate Last, Next_Last, Next_Next_Last */
		Last = ((Iter+1) == 8); NextLast = ((Iter+2) == 8); NextNextLast = ((Iter+3) == 8);
		/* =======================Read Next Tile=========================================== */
		rt_dma_wait(&DmaR_Evt2);
		if (!Last) {
			rt_dma_memcpy((rt_pointerT) Filter + ((Iter+1)*18432),
					(rt_pointerT) (PULP_Dronet_L1_Memory + 0) + (18432*((N_Ti+1) % 2)), 18432, RT_DMA_DIR_EXT2LOC, 0, &DmaR_Evt2);
		}
		/* ===================End Read Next Tile=========================================== */
		/* Call Kernel LOC_Unknown */
		KerArg0->Out = (short int * __restrict__) ((rt_pointerT) (PULP_Dronet_L1_Memory + 37120) + 256*((N_Ti) % 2));
		KerArg0->Bias = (short int * __restrict__) ((rt_pointerT) (PULP_Dronet_L1_Memory + 36864) + 0 + (Iter)*32);
		rt_team_fork(gap8_ncore(), (void *) KerParSetBias_fp, (void *) KerArg0);
		/* Kernel Iteration Loop on Iter1 space */
		Iter1=0; {
			/* Loop Iteration Body on Iter1 space */
			/* Elaborate Last, Next_Last, Next_Next_Last */
			Last1 = ((Iter1+1) == 1); NextLast1 = ((Iter1+2) == 1); NextNextLast1 = ((Iter1+3) == 1);
			/* Call Kernel LOC_Unknown */
			KerArg1->In = (short int * __restrict__) ((rt_pointerT) (PULP_Dronet_L1_Memory + 37632) + 0 + (0)*3584);
			KerArg1->BaseOutFeature = (unsigned short int) (0)*64;
			KerArg1->Filter = (short int * __restrict__) ((rt_pointerT) (PULP_Dronet_L1_Memory + 0) + 18432*((N_Ti) % 2));
			KerArg1->Out = (short int * __restrict__) ((rt_pointerT) (PULP_Dronet_L1_Memory + 37120) + 256*((N_Ti) % 2));
			rt_team_fork(gap8_ncore(), (void *) KerParConv3x3Stride2_fp, (void *) KerArg1);
			N_Ti1++;
			/* End Kernel Iteration Loop on Iter1 space */
		}
		Iter1=1;
		/* Call Kernel LOC_Unknown */
		KerArg2->In = (short int * __restrict__) ((rt_pointerT) (PULP_Dronet_L1_Memory + 37120) + 256*((N_Ti) % 2));
		KerArg2->Out = (short int * __restrict__) ((rt_pointerT) (PULP_Dronet_L1_Memory + 37120) + 256*((N_Ti) % 2));
		rt_team_fork(gap8_ncore(), (void *) KerParReLU_fp, (void *) KerArg2);
		/* =======================Write Tile=========================================== */
		if (Iter) {
			rt_dma_wait(&DmaW_Evt1);
		}
		rt_dma_memcpy((rt_pointerT) Out + ((Iter)*256),
			(rt_pointerT) (PULP_Dronet_L1_Memory + 37120) + (256*(N_Ti % 2)), 256, RT_DMA_DIR_LOC2EXT, 0, &DmaW_Evt1);
		/* ===================End Write Tile=========================================== */
		N_Ti++;
		/* End Kernel Iteration Loop on Iter space */
	}
	Iter=8;
	/* =======================Write Last Tile=========================================== */
	rt_dma_wait(&DmaW_Evt1);
	/* ===================End Write Last Tile=========================================== */
}

void MedParConv_3x3_S1_9(
		short int * __restrict__ In,
		short int * __restrict__ Filter,
		short int * __restrict__ Out,
		unsigned int Norm,
		short int * __restrict__ Bias,
		Kernel_T *Ker)

{
	/* Local variables used by this kernel */
	rt_dma_copy_t DmaR_Evt1;
	rt_dma_copy_t DmaR_Evt2;
	rt_dma_copy_t DmaR_Evt3;
	rt_dma_copy_t DmaW_Evt1;
	int Iter;
	int Last, NextLast, NextNextLast;
	int N_Ti = 0;
	int Iter1;
	int Last1, NextLast1, NextNextLast1;
	int N_Ti1 = 0;
	int N_TiIp = 0, InPlane, OutPlane=0;
	KerParSetBias_fp_T S_KerArg0, *KerArg0 = &S_KerArg0;
	KerParConv_fp_T S_KerArg1, *KerArg1 = &S_KerArg1;

	/* Initialize KerArg, Kernel invariant arguments */
	KerArg0->W = (unsigned short int) (4);
	KerArg0->H = (unsigned short int) (2);
	KerArg0->OutFeatures = (unsigned short int) (8);
	KerArg1->W = (unsigned short int) (4);
	KerArg1->H = (unsigned short int) (2);
	KerArg1->InFeatures = (unsigned short int) (128);
	KerArg1->OutFeatures = (unsigned short int) (8);
	KerArg1->Norm = (Norm);
	KerArg1->TileIndex = (unsigned short int) (128);
	KerArg1->NTile = (unsigned short int) (0);
	KerArg1->Pad = (v4s) (16843009);
	/* =======================Read First Tile=========================================== */
	/* Initial reads in L2, O_DB or O_BUFF */
	rt_dma_memcpy((rt_pointerT) In+(0), (rt_pointerT) (PULP_Dronet_L1_Memory + 37376)+0, 2048, RT_DMA_DIR_EXT2LOC, 0, &DmaR_Evt1);
	rt_dma_memcpy((rt_pointerT) Filter+(0), (rt_pointerT) (PULP_Dronet_L1_Memory + 0)+0, 18432, RT_DMA_DIR_EXT2LOC, 0, &DmaR_Evt2);
	rt_dma_memcpy((rt_pointerT) Bias+(0), (rt_pointerT) (PULP_Dronet_L1_Memory + 36864)+0, 256, RT_DMA_DIR_EXT2LOC, 0, &DmaR_Evt3);
	/* Wait for BUFF read in L2 */
	rt_dma_wait(&DmaR_Evt1);
	rt_dma_wait(&DmaR_Evt3);
	/* ===================End Read First Tile=========================================== */
	/* Kernel Iteration Loop on Iter space */
	for (Iter=0; Iter<16; Iter++) {
		/* Loop Iteration Body on Iter space */
		/* Elaborate Last, Next_Last, Next_Next_Last */
		Last = ((Iter+1) == 16); NextLast = ((Iter+2) == 16); NextNextLast = ((Iter+3) == 16);
		/* =======================Read Next Tile=========================================== */
		rt_dma_wait(&DmaR_Evt2);
		if (!Last) {
			rt_dma_memcpy((rt_pointerT) Filter + ((Iter+1)*18432),
					(rt_pointerT) (PULP_Dronet_L1_Memory + 0) + (18432*((N_Ti+1) % 2)), 18432, RT_DMA_DIR_EXT2LOC, 0, &DmaR_Evt2);
		}
		/* ===================End Read Next Tile=========================================== */
		/* Call Kernel LOC_Unknown */
		KerArg0->Out = (short int * __restrict__) ((rt_pointerT) (PULP_Dronet_L1_Memory + 37120) + 128*((N_Ti) % 2));
		KerArg0->Bias = (short int * __restrict__) ((rt_pointerT) (PULP_Dronet_L1_Memory + 36864) + 0 + (Iter)*16);
		rt_team_fork(gap8_ncore(), (void *) KerParSetBias_fp, (void *) KerArg0);
		/* Kernel Iteration Loop on Iter1 space */
		Iter1=0; {
			/* Loop Iteration Body on Iter1 space */
			/* Elaborate Last, Next_Last, Next_Next_Last */
			Last1 = ((Iter1+1) == 1); NextLast1 = ((Iter1+2) == 1); NextNextLast1 = ((Iter1+3) == 1);
			/* Call Kernel LOC_Unknown */
			KerArg1->In = (short int * __restrict__) ((rt_pointerT) (PULP_Dronet_L1_Memory + 37376) + 0 + (0)*2048);
			KerArg1->BaseOutFeature = (unsigned short int) (0)*128;
			KerArg1->Filter = (short int * __restrict__) ((rt_pointerT) (PULP_Dronet_L1_Memory + 0) + 18432*((N_Ti) % 2));
			KerArg1->Out = (short int * __restrict__) ((rt_pointerT) (PULP_Dronet_L1_Memory + 37120) + 128*((N_Ti) % 2));
			rt_team_fork(gap8_ncore(), (void *) KerParConv3x3Stride1_fp, (void *) KerArg1);
			N_Ti1++;
			/* End Kernel Iteration Loop on Iter1 space */
		}
		Iter1=1;
		/* =======================Write Tile=========================================== */
		if (Iter) {
			rt_dma_wait(&DmaW_Evt1);
		}
		rt_dma_memcpy((rt_pointerT) Out + ((Iter)*128),
			(rt_pointerT) (PULP_Dronet_L1_Memory + 37120) + (128*(N_Ti % 2)), 128, RT_DMA_DIR_LOC2EXT, 0, &DmaW_Evt1);
		/* ===================End Write Tile=========================================== */
		N_Ti++;
		/* End Kernel Iteration Loop on Iter space */
	}
	Iter=16;
	/* =======================Write Last Tile=========================================== */
	rt_dma_wait(&DmaW_Evt1);
	/* ===================End Write Last Tile=========================================== */
}

void MedParConv_1x1_S1_ReLU_10(
		short int * __restrict__ In,
		short int * __restrict__ Filter,
		short int * __restrict__ Out,
		unsigned int Norm,
		short int * __restrict__ Bias,
		Kernel_T *Ker)

{
	/* Local variables used by this kernel */
	rt_dma_copy_t DmaR_Evt1;
	rt_dma_copy_t DmaR_Evt2;
	rt_dma_copy_t DmaR_Evt3;
	rt_dma_copy_t DmaW_Evt1;
	int Iter;
	int Last, NextLast, NextNextLast;
	int N_Ti = 0;
	int Iter1;
	int Last1, NextLast1, NextNextLast1;
	int N_Ti1 = 0;
	int N_TiIp = 0, InPlane, OutPlane=0;
	KerParSetBias_fp_T S_KerArg0, *KerArg0 = &S_KerArg0;
	KerParConv_fp_T S_KerArg1, *KerArg1 = &S_KerArg1;

	/* Initialize KerArg, Kernel invariant arguments */
	KerArg0->W = (unsigned short int) (4);
	KerArg0->H = (unsigned short int) (2);
	KerArg0->OutFeatures = (unsigned short int) (128);
	KerArg1->W = (unsigned short int) (7);
	KerArg1->H = (unsigned short int) (4);
	KerArg1->InFeatures = (unsigned short int) (64);
	KerArg1->OutFeatures = (unsigned short int) (128);
	KerArg1->Norm = (Norm);
	KerArg1->TileIndex = (unsigned short int) (64);
	KerArg1->NTile = (unsigned short int) (0);
	KerArg1->Pad = (v4s) (0);
	/* =======================Read First Tile=========================================== */
	/* Initial reads in L2, O_DB or O_BUFF */
	rt_dma_memcpy((rt_pointerT) In+(0), (rt_pointerT) (PULP_Dronet_L1_Memory + 18688)+0, 3584, RT_DMA_DIR_EXT2LOC, 0, &DmaR_Evt1);
	rt_dma_memcpy((rt_pointerT) Filter+(0), (rt_pointerT) (PULP_Dronet_L1_Memory + 0)+0, 16384, RT_DMA_DIR_EXT2LOC, 0, &DmaR_Evt2);
	rt_dma_memcpy((rt_pointerT) Bias+(0), (rt_pointerT) (PULP_Dronet_L1_Memory + 16384)+0, 256, RT_DMA_DIR_EXT2LOC, 0, &DmaR_Evt3);
	/* Wait for BUFF read in L2 */
	rt_dma_wait(&DmaR_Evt1);
	rt_dma_wait(&DmaR_Evt2);
	rt_dma_wait(&DmaR_Evt3);
	/* ===================End Read First Tile=========================================== */
	/* Kernel Iteration Loop on Iter space */
	Iter=0; {
		/* Loop Iteration Body on Iter space */
		/* Elaborate Last, Next_Last, Next_Next_Last */
		Last = ((Iter+1) == 1); NextLast = ((Iter+2) == 1); NextNextLast = ((Iter+3) == 1);
		/* Call Kernel LOC_Unknown */
		KerArg0->Out = (short int * __restrict__) ((rt_pointerT) (PULP_Dronet_L1_Memory + 16640) + 0 + (0)*2048);
		KerArg0->Bias = (short int * __restrict__) ((rt_pointerT) (PULP_Dronet_L1_Memory + 16384) + 0 + (0)*256);
		rt_team_fork(gap8_ncore(), (void *) KerParSetBias_fp, (void *) KerArg0);
		/* Kernel Iteration Loop on Iter1 space */
		Iter1=0; {
			/* Loop Iteration Body on Iter1 space */
			/* Elaborate Last, Next_Last, Next_Next_Last */
			Last1 = ((Iter1+1) == 1); NextLast1 = ((Iter1+2) == 1); NextNextLast1 = ((Iter1+3) == 1);
			/* Call Kernel LOC_Unknown */
			KerArg1->In = (short int * __restrict__) ((rt_pointerT) (PULP_Dronet_L1_Memory + 18688) + 0 + (0)*3584);
			KerArg1->BaseOutFeature = (unsigned short int) (0)*64;
			KerArg1->Filter = (short int * __restrict__) ((rt_pointerT) (PULP_Dronet_L1_Memory + 0) + 0 + (0)*16384);
			KerArg1->Out = (short int * __restrict__) ((rt_pointerT) (PULP_Dronet_L1_Memory + 16640) + 0 + (0)*2048);
			rt_team_fork(gap8_ncore(), (void *) KerParConv1x1Stride2_fp, (void *) KerArg1);
			N_Ti1++;
			/* End Kernel Iteration Loop on Iter1 space */
		}
		Iter1=1;
		N_Ti++;
		/* End Kernel Iteration Loop on Iter space */
	}
	Iter=1;
	/* =======================Write Last Tile=========================================== */
	rt_dma_memcpy((rt_pointerT) Out + (0),
		(rt_pointerT) (PULP_Dronet_L1_Memory + 16640) + 0, 2048, RT_DMA_DIR_LOC2EXT, 0, &DmaW_Evt1);
	rt_dma_wait(&DmaW_Evt1);
	/* ===================End Write Last Tile=========================================== */
}

void AddFeatureMapsReLu_SW_3(
		short int * __restrict__ In,
		short int * __restrict__ Out,
		Kernel_T *Ker)

{
	/* Local variables used by this kernel */
	rt_dma_copy_t DmaR_Evt1;
	rt_dma_copy_t DmaR_Evt2;
	rt_dma_copy_t DmaW_Evt1;
	int Iter;
	int Last, NextLast, NextNextLast;
	int N_Ti = 0;
	int N_TiIp = 0, InPlane, OutPlane=0;
	KerAddFM_fpT S_KerArg0, *KerArg0 = &S_KerArg0;

	/* Initialize KerArg, Kernel invariant arguments */
	KerArg0->W = (int) (4);
	KerArg0->H = (int) (2);
	/* =======================Read First Tile=========================================== */
	/* Initial reads in L2, O_DB or O_BUFF */
	rt_dma_memcpy((rt_pointerT) In+(0), (rt_pointerT) (PULP_Dronet_L1_Memory + 0)+0, 2048, RT_DMA_DIR_EXT2LOC, 0, &DmaR_Evt1);
	rt_dma_memcpy((rt_pointerT) Out+(0), (rt_pointerT) (PULP_Dronet_L1_Memory + 2048)+0, 2048, RT_DMA_DIR_EXT2LOC, 0, &DmaR_Evt2);
	/* Wait for BUFF read in L2 */
	rt_dma_wait(&DmaR_Evt1);
	rt_dma_wait(&DmaR_Evt2);
	/* ===================End Read First Tile=========================================== */
	/* Kernel Iteration Loop on Input+Output Planes */
	for (OutPlane=0; OutPlane<128; OutPlane++) {
		int LastOutPlane = ((OutPlane+1) == 128), NextLastOutPlane = ((OutPlane+2) == 128);
		/* Kernel Iteration Loop on Iter space */
		Iter=0; {
			/* Loop Iteration Body on Iter space */
			/* Elaborate Last, Next_Last, Next_Next_Last */
			Last = ((Iter+1) == 1); NextLast = ((Iter+2) == 1); NextNextLast = ((Iter+3) == 1);
			/* Call Kernel LOC_INNER_LOOP */
			KerArg0->In = (Word16 * __restrict__) ((rt_pointerT) (PULP_Dronet_L1_Memory + 0) + OutPlane*16 + (0)*16);
			KerArg0->Out = (Word16 * __restrict__) ((rt_pointerT) (PULP_Dronet_L1_Memory + 2048) + OutPlane*16 + (0)*16);
			rt_team_fork(gap8_ncore(), (void *) KerAddFMReLu_fp, (void *) KerArg0);
			N_Ti++;
			/* End Kernel Iteration Loop on Iter space */
		}
		Iter=1;
		/* End Kernel Iteration Loop on Input+Output Planes */
	}
	/* =======================Write Last Tile=========================================== */
	rt_dma_memcpy((rt_pointerT) Out + (0),
		(rt_pointerT) (PULP_Dronet_L1_Memory + 2048) + 0, 2048, RT_DMA_DIR_LOC2EXT, 0, &DmaW_Evt1);
	rt_dma_wait(&DmaW_Evt1);
	/* ===================End Write Last Tile=========================================== */
}

void LinearLayer_SW_1(
		Word16 * __restrict__ In,
		Word16 * __restrict__ Filter,
		unsigned int NormFilter,
		Word16 * __restrict__ Bias,
		unsigned int NormBias,
		Word16 * __restrict__ Out,
		int OutSize,
		Kernel_T *Ker)

{
	/* Local variables used by this kernel */
	rt_dma_copy_t DmaR_Evt1;
	rt_dma_copy_t DmaR_Evt2;
	rt_dma_copy_t DmaR_Evt3;
	rt_dma_copy_t DmaW_Evt1;
	int Iter;
	int Last, NextLast, NextNextLast;
	int N_Ti = 0;
	int N_TiIp = 0, InPlane, OutPlane=0;
	KerLinearLayer_fpT S_KerArg0, *KerArg0 = &S_KerArg0;

	/* Initialize KerArg, Kernel invariant arguments */
	KerArg0->InSize = (int) (1024);
	KerArg0->NormFilter = (NormFilter);
	KerArg0->NormBias = (NormBias);
	KerArg0->OutSize = (int) (1);
	/* =======================Read First Tile=========================================== */
	/* Initial reads in L2, O_DB or O_BUFF */
	rt_dma_memcpy((rt_pointerT) In+0, (rt_pointerT) (PULP_Dronet_L1_Memory + 0)+0, 2048, RT_DMA_DIR_EXT2LOC, 0, &DmaR_Evt1);
	rt_dma_memcpy((rt_pointerT) Bias+(0), (rt_pointerT) (PULP_Dronet_L1_Memory + 2048)+0, 2, RT_DMA_DIR_EXT2LOC, 0, &DmaR_Evt2);
	rt_dma_memcpy((rt_pointerT) Filter+(0), (rt_pointerT) (PULP_Dronet_L1_Memory + 2052)+0, 2048, RT_DMA_DIR_EXT2LOC, 0, &DmaR_Evt3);
	/* Wait for BUFF read in L2 */
	rt_dma_wait(&DmaR_Evt1);
	rt_dma_wait(&DmaR_Evt2);
	rt_dma_wait(&DmaR_Evt3);
	/* ===================End Read First Tile=========================================== */
	/* Kernel Iteration Loop on Iter space */
	Iter=0; {
		/* Loop Iteration Body on Iter space */
		/* Elaborate Last, Next_Last, Next_Next_Last */
		Last = ((Iter+1) == 1); NextLast = ((Iter+2) == 1); NextNextLast = ((Iter+3) == 1);
		/* Call Kernel LOC_INNER_LOOP */
		KerArg0->In = (Word16 * __restrict__) ((rt_pointerT) (PULP_Dronet_L1_Memory + 0) + 0);
		KerArg0->Filter = (Word16 * __restrict__) ((rt_pointerT) (PULP_Dronet_L1_Memory + 2052) + 0 + (0)*2048);
		KerArg0->Bias = (Word16 *  __restrict__) ((rt_pointerT) (PULP_Dronet_L1_Memory + 2048) + 0 + (0)*2);
		KerArg0->Out = (Word16 *  __restrict__) ((rt_pointerT) (PULP_Dronet_L1_Memory + 4100) + 0 + (0)*2);
		rt_team_fork(gap8_ncore(), (void *) KerLinearLayer_fp, (void *) KerArg0);
		N_Ti++;
		/* End Kernel Iteration Loop on Iter space */
	}
	Iter=1;
	/* =======================Write Last Tile=========================================== */
	rt_dma_memcpy((rt_pointerT) Out + (0),
		(rt_pointerT) (PULP_Dronet_L1_Memory + 4100) + 0, 2, RT_DMA_DIR_LOC2EXT, 0, &DmaW_Evt1);
	rt_dma_wait(&DmaW_Evt1);
	/* ===================End Write Last Tile=========================================== */
}

void LinearLayer_SW_2(
		Word16 * __restrict__ In,
		Word16 * __restrict__ Filter,
		unsigned int NormFilter,
		Word16 * __restrict__ Bias,
		unsigned int NormBias,
		Word16 * __restrict__ Out,
		int OutSize,
		Kernel_T *Ker)

{
	/* Local variables used by this kernel */
	rt_dma_copy_t DmaR_Evt1;
	rt_dma_copy_t DmaR_Evt2;
	rt_dma_copy_t DmaR_Evt3;
	rt_dma_copy_t DmaW_Evt1;
	int Iter;
	int Last, NextLast, NextNextLast;
	int N_Ti = 0;
	int N_TiIp = 0, InPlane, OutPlane=0;
	KerLinearLayer_fpT S_KerArg0, *KerArg0 = &S_KerArg0;

	/* Initialize KerArg, Kernel invariant arguments */
	KerArg0->InSize = (int) (1024);
	KerArg0->NormFilter = (NormFilter);
	KerArg0->NormBias = (NormBias);
	KerArg0->OutSize = (int) (1);
	/* =======================Read First Tile=========================================== */
	/* Initial reads in L2, O_DB or O_BUFF */
	rt_dma_memcpy((rt_pointerT) In+0, (rt_pointerT) (PULP_Dronet_L1_Memory + 0)+0, 2048, RT_DMA_DIR_EXT2LOC, 0, &DmaR_Evt1);
	rt_dma_memcpy((rt_pointerT) Bias+(0), (rt_pointerT) (PULP_Dronet_L1_Memory + 2048)+0, 2, RT_DMA_DIR_EXT2LOC, 0, &DmaR_Evt2);
	rt_dma_memcpy((rt_pointerT) Filter+(0), (rt_pointerT) (PULP_Dronet_L1_Memory + 2052)+0, 2048, RT_DMA_DIR_EXT2LOC, 0, &DmaR_Evt3);
	/* Wait for BUFF read in L2 */
	rt_dma_wait(&DmaR_Evt1);
	rt_dma_wait(&DmaR_Evt2);
	rt_dma_wait(&DmaR_Evt3);
	/* ===================End Read First Tile=========================================== */
	/* Kernel Iteration Loop on Iter space */
	Iter=0; {
		/* Loop Iteration Body on Iter space */
		/* Elaborate Last, Next_Last, Next_Next_Last */
		Last = ((Iter+1) == 1); NextLast = ((Iter+2) == 1); NextNextLast = ((Iter+3) == 1);
		/* Call Kernel LOC_INNER_LOOP */
		KerArg0->In = (Word16 * __restrict__) ((rt_pointerT) (PULP_Dronet_L1_Memory + 0) + 0);
		KerArg0->Filter = (Word16 * __restrict__) ((rt_pointerT) (PULP_Dronet_L1_Memory + 2052) + 0 + (0)*2048);
		KerArg0->Bias = (Word16 *  __restrict__) ((rt_pointerT) (PULP_Dronet_L1_Memory + 2048) + 0 + (0)*2);
		KerArg0->Out = (Word16 *  __restrict__) ((rt_pointerT) (PULP_Dronet_L1_Memory + 4100) + 0 + (0)*2);
		rt_team_fork(gap8_ncore(), (void *) KerLinearLayer_fp, (void *) KerArg0);
		N_Ti++;
		/* End Kernel Iteration Loop on Iter space */
	}
	Iter=1;
	/* =======================Write Last Tile=========================================== */
	rt_dma_memcpy((rt_pointerT) Out + (0),
		(rt_pointerT) (PULP_Dronet_L1_Memory + 4100) + 0, 2, RT_DMA_DIR_LOC2EXT, 0, &DmaW_Evt1);
	rt_dma_wait(&DmaW_Evt1);
	/* ===================End Write Last Tile=========================================== */
}

void LinearLayer_SW_3(
		Word16 * __restrict__ In,
		Word16 * __restrict__ Filter,
		unsigned int NormFilter,
		Word16 * __restrict__ Bias,
		unsigned int NormBias,
		Word16 * __restrict__ Out,
		int OutSize,
		Kernel_T *Ker)

{
	/* Local variables used by this kernel */
	rt_dma_copy_t DmaR_Evt1;
	rt_dma_copy_t DmaR_Evt2;
	rt_dma_copy_t DmaR_Evt3;
	rt_dma_copy_t DmaW_Evt1;
	int Iter;
	int Last, NextLast, NextNextLast;
	int N_Ti = 0;
	int N_TiIp = 0, InPlane, OutPlane=0;
	KerLinearLayer_fpT S_KerArg0, *KerArg0 = &S_KerArg0;

	/* Initialize KerArg, Kernel invariant arguments */
	KerArg0->InSize = (int) (1024);
	KerArg0->NormFilter = (NormFilter);
	KerArg0->NormBias = (NormBias);
	KerArg0->OutSize = (int) (1);
	/* =======================Read First Tile=========================================== */
	/* Initial reads in L2, O_DB or O_BUFF */
	rt_dma_memcpy((rt_pointerT) In+0, (rt_pointerT) (PULP_Dronet_L1_Memory + 0)+0, 2048, RT_DMA_DIR_EXT2LOC, 0, &DmaR_Evt1);
	rt_dma_memcpy((rt_pointerT) Bias+(0), (rt_pointerT) (PULP_Dronet_L1_Memory + 2048)+0, 2, RT_DMA_DIR_EXT2LOC, 0, &DmaR_Evt2);
	rt_dma_memcpy((rt_pointerT) Filter+(0), (rt_pointerT) (PULP_Dronet_L1_Memory + 2052)+0, 2048, RT_DMA_DIR_EXT2LOC, 0, &DmaR_Evt3);
	/* Wait for BUFF read in L2 */
	rt_dma_wait(&DmaR_Evt1);
	rt_dma_wait(&DmaR_Evt2);
	rt_dma_wait(&DmaR_Evt3);
	/* ===================End Read First Tile=========================================== */
	/* Kernel Iteration Loop on Iter space */
	Iter=0; {
		/* Loop Iteration Body on Iter space */
		/* Elaborate Last, Next_Last, Next_Next_Last */
		Last = ((Iter+1) == 1); NextLast = ((Iter+2) == 1); NextNextLast = ((Iter+3) == 1);
		/* Call Kernel LOC_INNER_LOOP */
		KerArg0->In = (Word16 * __restrict__) ((rt_pointerT) (PULP_Dronet_L1_Memory + 0) + 0);
		KerArg0->Filter = (Word16 * __restrict__) ((rt_pointerT) (PULP_Dronet_L1_Memory + 2052) + 0 + (0)*2048);
		KerArg0->Bias = (Word16 *  __restrict__) ((rt_pointerT) (PULP_Dronet_L1_Memory + 2048) + 0 + (0)*2);
		KerArg0->Out = (Word16 *  __restrict__) ((rt_pointerT) (PULP_Dronet_L1_Memory + 4100) + 0 + (0)*2);
		rt_team_fork(gap8_ncore(), (void *) KerLinearLayer_fp, (void *) KerArg0);
		N_Ti++;
		/* End Kernel Iteration Loop on Iter space */
	}
	Iter=1;
	/* =======================Write Last Tile=========================================== */
	rt_dma_memcpy((rt_pointerT) Out + (0),
		(rt_pointerT) (PULP_Dronet_L1_Memory + 4100) + 0, 2, RT_DMA_DIR_LOC2EXT, 0, &DmaW_Evt1);
	rt_dma_wait(&DmaW_Evt1);
	/* ===================End Write Last Tile=========================================== */
}

void LinearLayer_SW_4(
		Word16 * __restrict__ In,
		Word16 * __restrict__ Filter,
		unsigned int NormFilter,
		Word16 * __restrict__ Bias,
		unsigned int NormBias,
		Word16 * __restrict__ Out,
		int OutSize,
		Kernel_T *Ker)

{
	/* Local variables used by this kernel */
	rt_dma_copy_t DmaR_Evt1;
	rt_dma_copy_t DmaR_Evt2;
	rt_dma_copy_t DmaR_Evt3;
	rt_dma_copy_t DmaW_Evt1;
	int Iter;
	int Last, NextLast, NextNextLast;
	int N_Ti = 0;
	int N_TiIp = 0, InPlane, OutPlane=0;
	KerLinearLayer_fpT S_KerArg0, *KerArg0 = &S_KerArg0;

	/* Initialize KerArg, Kernel invariant arguments */
	KerArg0->InSize = (int) (1024);
	KerArg0->NormFilter = (NormFilter);
	KerArg0->NormBias = (NormBias);
	KerArg0->OutSize = (int) (1);
	/* =======================Read First Tile=========================================== */
	/* Initial reads in L2, O_DB or O_BUFF */
	rt_dma_memcpy((rt_pointerT) In+0, (rt_pointerT) (PULP_Dronet_L1_Memory + 0)+0, 2048, RT_DMA_DIR_EXT2LOC, 0, &DmaR_Evt1);
	rt_dma_memcpy((rt_pointerT) Bias+(0), (rt_pointerT) (PULP_Dronet_L1_Memory + 2048)+0, 2, RT_DMA_DIR_EXT2LOC, 0, &DmaR_Evt2);
	rt_dma_memcpy((rt_pointerT) Filter+(0), (rt_pointerT) (PULP_Dronet_L1_Memory + 2052)+0, 2048, RT_DMA_DIR_EXT2LOC, 0, &DmaR_Evt3);
	/* Wait for BUFF read in L2 */
	rt_dma_wait(&DmaR_Evt1);
	rt_dma_wait(&DmaR_Evt2);
	rt_dma_wait(&DmaR_Evt3);
	/* ===================End Read First Tile=========================================== */
	/* Kernel Iteration Loop on Iter space */
	Iter=0; {
		/* Loop Iteration Body on Iter space */
		/* Elaborate Last, Next_Last, Next_Next_Last */
		Last = ((Iter+1) == 1); NextLast = ((Iter+2) == 1); NextNextLast = ((Iter+3) == 1);
		/* Call Kernel LOC_INNER_LOOP */
		KerArg0->In = (Word16 * __restrict__) ((rt_pointerT) (PULP_Dronet_L1_Memory + 0) + 0);
		KerArg0->Filter = (Word16 * __restrict__) ((rt_pointerT) (PULP_Dronet_L1_Memory + 2052) + 0 + (0)*2048);
		KerArg0->Bias = (Word16 *  __restrict__) ((rt_pointerT) (PULP_Dronet_L1_Memory + 2048) + 0 + (0)*2);
		KerArg0->Out = (Word16 *  __restrict__) ((rt_pointerT) (PULP_Dronet_L1_Memory + 4100) + 0 + (0)*2);
		rt_team_fork(gap8_ncore(), (void *) KerLinearLayer_fp, (void *) KerArg0);
		N_Ti++;
		/* End Kernel Iteration Loop on Iter space */
	}
	Iter=1;
	/* =======================Write Last Tile=========================================== */
	rt_dma_memcpy((rt_pointerT) Out + (0),
		(rt_pointerT) (PULP_Dronet_L1_Memory + 4100) + 0, 2, RT_DMA_DIR_LOC2EXT, 0, &DmaW_Evt1);
	rt_dma_wait(&DmaW_Evt1);
	/* ===================End Write Last Tile=========================================== */
}

