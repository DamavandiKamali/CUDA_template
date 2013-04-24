#include <stdio.h>
#define row1 50
#define col2 50
#define col1_row2 5
int a[row1][col1_row2],b[col1_row2][col2],c[row1][col2];
int main(){

	for(int i=0;i<row1;i++)
		for(int j=0;j<col1_row2;j++)
			a[i][j]=1;
			
	for(int i=0;i<col1_row2;i++)
		for(int j=0;j<col2;j++)
			b[i][j]+=1;
			
	for(int i=0;i<row1;i++)
		for(int j=0;j<col2;j++)
			for(int k=0;k<col1_row2;k++)
				c[i][j]+=a[i][k]*b[k][j];
				
	for(int i=0;i<row1;i++){
		for(int j=0;j<col2;j++)
			printf("%2d",c[i][j]);
		printf("\n\n");
		}
	return(0);
	}