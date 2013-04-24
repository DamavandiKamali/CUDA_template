#include <stdio.h>
#include <time.h>
#define row1 200
#define col2 200
#define col1_row2 200
int a[row1*col1_row2];
int b[col1_row2*col2];
int c[row1*col2];
int main(){
	/////////////////////////////////////////////
	clock_t start;
	clock_t end;
	double function_time;
	//////////////////////////////////////////////
	for(int i=0;i<row1;i++)
		for(int j=0;j<col1_row2;j++)
			a[i+row1*j]=1;
	for(int i=0;i<col1_row2;i++)
		for(int j=0;j<col2;j++)
			b[i+col1_row2*j]=1;
	start = clock();
	for (int i = 0; i < row1; ++i)
		for (int j = 0; j < col2; ++j)
			for (int k = 0; k < col1_row2; ++k)
				c[i+row1*j] += a[i+row1*k] * b[k+col1_row2*j];
	end = clock();
	function_time = (double)(end - start) / (CLOCKS_PER_SEC / 1000.0);
	printf("Difference is %2.5f ms \n",(float) function_time);
	/*for (int i = 0; i < row1; ++i){
		for (int j = 0; j < col2; ++j)
			printf("%4d",c[i+row1*j]);
		printf("\n\n");
	}*/
	printf("%4d\n",c[1]);
	return(0);
	}