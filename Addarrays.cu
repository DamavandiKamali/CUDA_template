
/*
 * Driver APIC code that calls a runtime kernel
 * Vector addition: C = A + B.
 */

// Includes
#include <stdio.h>
#include <cuda.h>
#include <cutil_inline.h>

// Variables
CUdevice cuDevice;
CUcontext cuContext;
CUmodule cuModule;
CUfunction vecAdd;
float* h_A;
float* h_B;
float* h_C;
CUdeviceptr d_A;
CUdeviceptr d_B;
CUdeviceptr d_C;

// Functions

__global__ void kernel(float* d_a, float* d_b, float* d_c, int n)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  
  if(idx < n)
    d_c[idx] = d_a[idx] + d_b[idx];
}

// Allocates an array with random float entries.
void RandomInit(float* data, int n)
{
  for (int i = 0; i < n; ++i)
    data[i] = rand() / (float)RAND_MAX;
}

void errorExit()
{
  printf("Error exit!\n");
  exit(1);
}

// Host code
int main(int argc, char** argv)
{
  int N = 50000;
  unsigned int size = N * sizeof(float);
  CUresult error;
  
  printf("Vector Addition (Driver API)\n");
  // Initialize
  error = cuInit(0);
  if (error != CUDA_SUCCESS) errorExit();
  
  // Get number of devices supporting CUDA
  int deviceCount = 0;
  error = cuDeviceGetCount(&deviceCount);
  if (error != CUDA_SUCCESS) errorExit();
  if (deviceCount == 0) {
    printf("There is no device supporting CUDA.\n");
    exit(1);
  }
  
  // Get handle for device 0
  error = cuDeviceGet(&cuDevice, 0);
  if (error != CUDA_SUCCESS) errorExit();
  
  // Create context
  error = cuCtxCreate(&cuContext, 0, cuDevice);
  if (error != CUDA_SUCCESS) errorExit();
  
  // Allocate input vectors h_A and h_B in host memory
  h_A = (float*)malloc(size);
  if (h_A == 0) errorExit();
  h_B = (float*)malloc(size);
  if (h_B == 0) errorExit();
  h_C = (float*)malloc(size);
  if (h_C == 0) errorExit();
  
  // Initialize input vectors
  RandomInit(h_A, N);
  RandomInit(h_B, N);
  
  // Allocate vectors in device memory
  error = cuMemAlloc(&d_A, size);
  if (error != CUDA_SUCCESS) errorExit();
  error = cuMemAlloc(&d_B, size);
  if (error != CUDA_SUCCESS) errorExit();
  error = cuMemAlloc(&d_C, size);
  if (error != CUDA_SUCCESS) errorExit();
  
  // Copy vectors from host memory to device memory
  error = cuMemcpyHtoD(d_A, h_A, size);
  if (error != CUDA_SUCCESS) errorExit();
  error = cuMemcpyHtoD(d_B, h_B, size);
  if (error != CUDA_SUCCESS) errorExit();
  
  // Invoke kernel (Runtime API)
  int nThreadsPerBlk=128;
  int nBlks = (N/nThreadsPerBlk) + (((N%nThreadsPerBlk)>0)?1:0);
  kernel<<<nBlks,nThreadsPerBlk>>>((float*)d_A,(float*) d_B,(float*) d_C, N);
  
  // Copy result from device memory to host memory
  // h_C contains the result in host memory
  error = cuMemcpyDtoH(h_C, d_C, size);
  if (error != CUDA_SUCCESS) errorExit();
  
  // Verify result
  int i;
  for (i = 0; i < N; ++i) {
    float sum = h_A[i] + h_B[i];
    if (fabs(h_C[i] - sum) > 1e-7f) {
      printf("Mistake index %d %g %g\n",i,h_C[i],sum);
      break;
    }
  }
  printf("Test %s \n", (i == N) ? "PASSED" : "FAILED");
  return(0);
}
