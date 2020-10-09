// Enda Carroll
// May 2020
// Programme to test the convolution term

// ---------------------------------------------------------------------
//  Standard Libraries and Headers
// ---------------------------------------------------------------------
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <complex.h>
#include <lapacke.h>
#include <gsl/gsl_cblas.h>



double my_dot(int n, double* a, int incra, double* b, int incrb);


// ---------------------------------------------------------------------
//  Main function
// ---------------------------------------------------------------------
int main(int argc, char** argv) {


	int N = 8;
	int M = N;

	double* A = (double* )malloc(sizeof(double) * N * M);
	double* res = (double* )malloc(sizeof(double) * N * M);


	// Fill array
	for (int i = 0; i < N; ++i)
	{
		for (int j = 0; j < M; ++j)
		{
			A[i * M + j] = (double) ((i + 1) * (j + 1));

			printf("A[%d]: \t%lf\t", i * M + j, A[i * M + j]);
		}
		printf("\n");
	}
	printf("\n\n");


	double tmp = my_dot(N, &A[0], M, &A[2], M);
	printf("My Dot: %5.16lf\n", tmp);
	printf("\n\n");
	tmp = cblas_ddot(N, &A[0], M, &A[2], M);
	printf("CBLAS Dot: %5.16lf\n", tmp);


	for (int i = 0; i < N; ++i)
	{
		for (int j = 0; j < i; ++j)
		{
			// tmp = cblas_ddot(N, A[i], M, A[j], M);

		}
		// printf("\n");
	}


	return 0;
}



double my_dot(int n, double* a, int incra, double* b, int incrb) {

	double res;

	for (int i = 0; i < n; ++i)
	{
		res += a[i * incra] * b[i * incrb];
		printf("a[i]: %lf | a[j]: %lf\n", a[i * incra], b[i * incrb]);
	}
	printf("\n\n");

	return res;
}