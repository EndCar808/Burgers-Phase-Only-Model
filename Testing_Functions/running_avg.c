#include <stdlib.h>
#include <stdio.h>
#include <math.h>



int main(int argc, char** argv) {

	int N = 50;

	int dw = 5;

	double runsum = 0.0;

	double* data   = (double* )malloc(sizeof(double) * N);

	double* cumsum = (double*)malloc(sizeof(double) * N);
	double* avg    = (double* )malloc(sizeof(double) * N);

	for (int i = 0; i < N; ++i) {
		data[i] = (double) i;
		// printf("d[%d]: %lf\n", i, data[i]);
	}
	printf("\n\n");


	// do average
	int count = 1;
	for (int i = 0; i < N; ++i)	{

		runsum += data[i];

		cumsum[i] = runsum;
		
		if (i >= dw) {
			avg[i] = (cumsum[i] - cumsum[i - dw]) / dw;
			printf("Avg: %5.16lf\n", avg[i]);
		}		
		count++;
	}
	


	return 0;
}