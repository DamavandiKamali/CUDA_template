#include <stdio.h>
#define row1 300
#define col2 300
#define col1_row2 300
#define block_size 20
/////////////////////////////////////////////////////////
/// mul_matrix function ////////////////////////////
/////////////////////////////////////////////////////////
__global__ void mul_matrix(int *a,int *b,int *c){
int col = blockIdx.x*blockDim.x+threadIdx.x;
int row = blockIdx.y*blockDim.y+threadIdx.y;
int P_val = 0;
for (int k = 0; k <col1_row2; k++)
P_val +=a[row + row1*k]*b[k + col*col1_row2 ] ;
c[row+col*row1] = P_val;
}
//////////////////////////////////////////////////////////
/// print_matrix function ///////////////////////////
////////////////////////////////////////////////////////
void print_matrix(int *c){
for (int i = 0; i < row1; ++i){
for (int j = 0; j < col2; ++j)
printf("%4d",c[i+row1*j]);
printf("\n\n");
}
}
////////////////////////////////////////////////////////
/// main function ////////////////////////////////
////////////////////////////////////////////////////////
int main(){
cudaEvent_t start,stop;
int a[row1*col1_row2];
int b[col1_row2*col2];
int c[row1*col2];
int size1 = row1*col1_row2 * sizeof(int);
int size2 = col1_row2*col2 * sizeof(int);
int size3 = row1*col2 * sizeof(int);
for(int i=0;i<row1;i++)
for(int j=0;j<col1_row2;j++)
a[i+row1*j]=1;
for(int i=0;i<col1_row2;i++)
for(int j=0;j<col2;j++)
b[i+col1_row2*j]=1;
int *d_a, *d_b, *d_c; // device copies of a, b, c
// Alloc space for device copies of a, b, c
cudaMalloc((void **)&d_a, size1);
cudaMalloc((void **)&d_b, size2);
cudaMalloc((void **)&d_c, size3);
// Copy inputs to device
cudaMemcpy(d_a, a, size1, cudaMemcpyHostToDevice);
cudaMemcpy(d_b, b, size2, cudaMemcpyHostToDevice);
cudaEventCreate(&start);
cudaEventCreate(&stop);
cudaEventRecord(start,0);
dim3 dimGrid(col2/block_size,row1/block_size);
dim3 dimBlock(block_size,block_size);
// Launch add() kernel on GPU with N blocks
mul_matrix<<<dimGrid, dimBlock>>>(d_a, d_b, d_c);
cudaEventRecord(stop,0);
cudaEventSynchronize(stop);
float elapsedTime;
cudaEventElapsedTime(&elapsedTime,start,stop);
printf("time to generate:%3.5f ms\n",elapsedTime);
// Copy result back to host
cudaMemcpy(c, d_c, size3, cudaMemcpyDeviceToHost);
// Cleanup
cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
//disply answer
//print_matrix(c);
printf("c[%d]=%d\n",100,c[100]);
return(0);
}