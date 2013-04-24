#include <stdio.h>
typedef struct {
int width;
int height;
float *elements;
} Matrix;
// Thread block size
#define BLOCK_SIZE 20
__global__ void MatMulKernel(Matrix A, Matrix B, Matrix C)
{
// Each thread computes one element of C
// by accumulating results into Cvalue
	float Cvalue = 0;
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	for (int e = 0; e < A.width; ++e)
		Cvalue += A.elements[row * A.width + e]* B.elements[e * B.width + col];
	C.elements[row * C.width + col] = Cvalue;
	}
// Forward declaration of the matrix multiplication kernel
//__global__ void MatMulKernel(const Matrix, const Matrix, Matrix);
// Matrix multiplication - Host code
// Matrix dimensions are assumed to be multiples of BLOCK_SIZE
void MatMul( Matrix A, Matrix B, Matrix C)
{
// Load A and B to device memory
	Matrix d_A;
	d_A.width = A.width; d_A.height = A.height;
	size_t size = A.width * A.height * sizeof(float);
	cudaMalloc((void **)&d_A.elements, size);
	cudaMemcpy(d_A.elements, A.elements, size,cudaMemcpyHostToDevice);
	Matrix d_B;
	d_B.width = B.width; d_B.height = B.height;
	size = B.width * B.height * sizeof(float);
	cudaMalloc((void **)&d_B.elements, size);
	cudaMemcpy(d_B.elements, B.elements, size,cudaMemcpyHostToDevice);
	// Allocate C in device memory
	Matrix d_C;
	d_C.width = C.width; d_C.height = C.height;
	size = C.width * C.height * sizeof(float);
	cudaMalloc((void **)&d_C.elements, size);
	// Invoke kernel
	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
	dim3 dimGrid(B.width / dimBlock.x, A.height / dimBlock.y);
	MatMulKernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C);
	// Read C from device memory
	cudaMemcpy(C.elements, d_C.elements, size,cudaMemcpyDeviceToHost);
	// Free device memory
	cudaFree(d_A.elements);
	cudaFree(d_B.elements);
	cudaFree(d_C.elements);
	}
// Matrix multiplication kernel called by MatMul()
//////////////////////////////////////////////////////////
/// print_matrix function ///////////////////////////
////////////////////////////////////////////////////////
void print_matrix(float *c,int row,int col){
	for (int i = 0; i < row; ++i){
		for (int j = 0; j < col; ++j)
			printf("%4.2f ",c[i+row*j]);
		printf("\n\n");
	}
}
//////////////////////////////////////////////////////////
/// random_init function ///////////////////////////
////////////////////////////////////////////////////////
void random_init(float *a,int size){
	for(int i=0;i<size;i++)
	a[i]=rand()%10;
}
////////////////////////////////////////////////////////
int main(void){
///////////////////////////////////////////////////////////////////////////////
cudaEvent_t start,stop;
///////////////////////////////////////////////////////////////////////////////
	Matrix A,B;
	Matrix C;
	A.width=400;
	A.height=400;
	B.width=400;
	B.height=400;
	C.width=B.width;
	C.height=A.height;
	size_t size = A.width * A.height * sizeof(float);
	A.elements = (float *)malloc(size);
	random_init(A.elements,A.width * A.height );
	size = B.width * B.height * sizeof(float);
	B.elements= (float *)malloc(size);
	random_init(B.elements,B.width * B.height);
	size = C.width * C.height * sizeof(float);
	C.elements= (float *)malloc(size);
	//for(int i=0;i<A.width*A.height;i++)
	//A.elements[i]=1;
	//for(int i=0;i<B.width*B.height;i++)
	//B.elements[i]=1;
	printf("matrix A(%d,%d) & matrix B(%d,%d) & matrix C(%d,%d)\n",A.width,A.height,B.width,
	B.height,C.width,C.height);
	///////////////////////////////////////////////////////////////////////////////
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start,0);
	///////////////////////////////////////////////////////////////////////////////
	MatMul(A,B,C);
	///////////////////////////////////////////////////////////////////////////////
	cudaEventRecord(stop,0);
	cudaEventSynchronize(stop);
	float elapsedTime;
	cudaEventElapsedTime(&elapsedTime,start,stop);
	printf("Time to genreat : %3.5f ms\n",elapsedTime);
	///////////////////////////////////////////////////////////////////////////////
	// print_matrix(C.elements,C.height,C.width);
	printf("c[%d]=%f\n",1,C.elements[100]);
	return(0);
}