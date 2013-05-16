
#include <iostream>
#include <cuda.h>
#include<cuda_runtime.h>
#include <conio.h> 
#include<cutil.h>
#include<time.h>
#define DATAXSIZE 10
#define DATAYSIZE 10
#define DATAZSIZE 10
#define N1 10
#define N2 10
#define N3 10
//define the chunk sizes that each threadblock will work on
#define BLKXSIZE 32
#define BLKYSIZE 4
#define BLKZSIZE 4
 typedef float MATRIX[DATAYSIZE][DATAXSIZE];

float result=0,result1=0;
int i,j,k;
    const dim3 blockSize(BLKXSIZE, BLKYSIZE, BLKZSIZE);
    const dim3 gridSize(((DATAXSIZE+BLKXSIZE-1)/BLKXSIZE), ((DATAYSIZE+BLKYSIZE-1)/BLKYSIZE), ((DATAZSIZE+BLKZSIZE-1)/BLKZSIZE));

    const int nx = DATAXSIZE;
    const int ny = DATAYSIZE;
    const int nz = DATAZSIZE;
	MATRIX *vv;     // storage for result stored on host
    MATRIX *d_vv;   // storage for result computed on device
// for cuda error checking
#define cudaCheckErrors(msg) \
    do { \
        cudaError_t __err = cudaGetLastError(); \
        if (__err != cudaSuccess) { \
            fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n", \
                msg, cudaGetErrorString(__err), \
                __FILE__, __LINE__); \
            fprintf(stderr, "*** FAILED - ABORTING\n"); getch();\
            return 1; \
        } \
    } while (0)

// device function to set the 3D volume
__global__ void set(float gg[][DATAYSIZE][DATAXSIZE])
{
	unsigned idx = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned idy = blockIdx.y*blockDim.y + threadIdx.y;
    unsigned idz = blockIdx.z*blockDim.z + threadIdx.z;

	if ((idx < (DATAXSIZE)) && (idy < (DATAYSIZE)) && (idz < (DATAZSIZE)))
	{gg[idz][idy][idx]=idx*idy*idz;}
				__syncthreads();
	if ((idx < (DATAXSIZE)) && (idy < (DATAYSIZE)) && (idz < (DATAZSIZE)))
	{gg[idz][idy][idx]=gg[idz][idy][idx]+2.0;}

}

int main(int argc, char *argv[])
{ clock_t start, end,start1,end1;

// allocate storage for data set
	if ((vv = (MATRIX *)malloc((nx*ny*nz)*sizeof(float))) == 0) {fprintf(stderr,"malloc1 Fail \n"); return 1;}

	float result,result1;
	cudaMalloc((void **) &d_vv, (nx*ny*nz)*sizeof(float));
    cudaCheckErrors("Failed to allocate device buffer");
	//start= clock();
	cudaMemcpy(d_vv, vv, ((nx*ny*nz)*sizeof(float)), cudaMemcpyHostToDevice);
	cudaCheckErrors("CUDA memcpy failure");
// compute result
	start= clock();
    set<<<gridSize,blockSize>>>(d_vv);
	end=clock();
    cudaCheckErrors("Kernel launch failure");
// copy output data back to host
	
	cudaMemcpy(vv, d_vv, ((nx*ny*nz)*sizeof(float)), cudaMemcpyDeviceToHost);
	cudaCheckErrors("CUDA memcpy failure");

	//end=clock();
		 result=((float(end-start))/CLOCKS_PER_SEC);
	 printf("\n GPU Time= %lg\n",result);

	 printf("gridsize Device = %d\n", gridSize.x*gridSize.y*gridSize.z);
    printf("blockSize Device = %d\n", blockSize.x*blockSize.y*blockSize.z);
// and check for accuracy
	start1= clock();	
    for ( k=0; k<nz; k++)
	    for ( j=0; j<ny; j++)
	        for ( i=0; i<nx; i++)
		         if (vv[k][j][i] != float(i*j*k+2) )
				 {
                     printf("Mismatch at x= %d, y= %d, z= %d  Host= %d, Device = %lf\n", i, j, k, (i*j*k+2.0), vv[k][j][i]);
			         getch();
                     return 1;
				 }
end1=clock();	  
	
	 printf("CLOCKS_PER_SEC=%d\n",CLOCKS_PER_SEC);
	  result1=((float(end1-start1))/CLOCKS_PER_SEC);
	 printf("\n CPU Time= %lg",result1);
    free(vv);
    cudaFree(d_vv);
    cudaCheckErrors("cudaFree fail");
	getch();
    return 0;
}