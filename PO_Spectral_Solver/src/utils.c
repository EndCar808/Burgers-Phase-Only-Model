// Enda Carroll
// Sept 2019
// File containing utility functions for solver

// ---------------------------------------------------------------------
//  Standard Libraries and Headers
// ---------------------------------------------------------------------
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include <complex.h>
#include <fftw3.h>
#include <hdf5.h>
#include <hdf5_hl.h>
#include <omp.h>
#include <gsl/gsl_cblas.h>
#include <sys/types.h>
#include <sys/stat.h>

// ---------------------------------------------------------------------
//  User Libraries and Headers
// ---------------------------------------------------------------------
#include "data_types.h"
#include "solver.h"


// ---------------------------------------------------------------------
//  Function Definitions
// ---------------------------------------------------------------------
void mem_chk (void *arr_ptr, char *name) {
  if (arr_ptr == NULL ) {
    fprintf(stderr, "ERROR!! |in file \"%s\"-line:%d | Unable to malloc required memory for %s, now exiting!\n", __FILE__, __LINE__, name);
    exit(1);
  }
}


int max_indx_d(double* array, int n) {

	double max = 0.0;

	int indx = 1;

	for (int i = 0; i < n; ++i) {
		if (fabs(array[i]) > max) {
			indx = i;
			max  = fabs(array[i]);
		}
	}

	return indx;
}

int sgn(int x) {

	int val = 0;

	if (x > 0) {
		val = 1;
	}
	else if(x < 0) {
		val = -1;
	}
	else if (x == 0) {
		val = 0;
	}

	return val;
}

void write_array(double *A, int n, char *filename) {

	int i;

	FILE *ft = fopen( filename, "w");

	for ( i = 0; i < n; i++ ) {
		fprintf(ft, "%f ",  A[i]);
	}

	fclose(ft);
}

void write_array_fftwreal(fftw_complex *A, int n, char *filename) {

	int i;

	FILE *ft = fopen( filename, "w");

	for ( i = 0; i < n; i++ ) {
		fprintf(ft, "%lf ",  creal(A[i]));
	}

	fclose(ft);
}

void write_array_fftwimag(fftw_complex *A, int n, char *filename) {

	int i;

	FILE *ft = fopen( filename, "w");

	for ( i = 0; i < n; i++ ) {
		fprintf(ft, "%lf ",  cimag(A[i]));
	}

	fclose(ft);
}

void print_array_1d_d(double* arr, char* arr_name, int n) {

	for (int i = 0; i < n; ++i)	{
		printf("%s[%d]: %5.16lf \n", arr_name, i, arr[i]);
	}
	printf("\n\n");
}

void print_array_2d_d(double* arr, char* arr_name, int r, int c) {

	int tmp;
	int indx;
	for (int i = 0; i < r; ++i)	{
		tmp = i * c;
		for (int j = 0; j < c; ++j) {
			indx = tmp + j;
			printf("%s[%d]: %5.16lf \t", arr_name, indx, arr[indx]);
		}
		printf("\n");
	}
	printf("\n\n");
}

void print_array_1d_z(fftw_complex* arr, char* arr_name, int n) {

	for (int i = 0; i < n; ++i)	{
		printf("%s[%d]: %5.16lf %5.16lf I\n", arr_name, i, creal(arr[i]), cimag(arr[i]));
	}
	printf("\n\n");
}

void print_array_2d_z(fftw_complex* arr, char* arr_name, int r, int c) {

	int tmp;
	int indx;
	for (int i = 0; i < r; ++i)	{
		tmp = i * c;
		for (int j = 0; j < c; ++j) {
			indx = tmp + j;
			printf("%s[%d]: %5.16lf %5.16lf I\t", arr_name, indx, creal(arr[indx]), cimag(arr[indx]));
		}
		printf("\n");
	}
	printf("\n\n");
}

