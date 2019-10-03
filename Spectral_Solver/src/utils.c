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
#include <complex.h>
#include <fftw3.h>


// ---------------------------------------------------------------------
//  User Libraries and Headers
// ---------------------------------------------------------------------
// #include "forcing.h"




// ---------------------------------------------------------------------
//  Function Definitions
// ---------------------------------------------------------------------
void initial_condition(double* u, double complex* u_z, fftw_plan real2compl, double dx, int N) {

	double x;

	// fill real solution array
	for(int i = 0; i < N; ++i) {
		x = 0 + (double)i*dx;

		u[i] = -sin(x); 
	}

	fftw_execute(real2compl);
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


void deriv(double complex* u_z, double complex* dudt_z, int* k, fftw_plan real2compl, fftw_plan compl2real, int n) {

	double norm_fact = 1 / (double) N;

	// Allocate temporary memory
	double complex* dudx_z_tmp;
	dudx_z_tmp = (double complex* )fftw_malloc((N/2 + 1)*sizeof(double complex));
	double* dudx_tmp;
	dudx_tmp   = (double* )malloc(N*sizeof(double));
	double* u_tmp;
	u_tmp      = (double* )malloc(N*sizeof(double));
	

	// find the derivative of dudx in complex space
	for (int i = 0; i < N/2 + 1; ++i) {
		dudx_z_tmp[i] = I*k[i]*u_z[i];
	}

	// transform this derivative back using "new array execute" functionality of FFTW
	fftw_execute_dft_c2r(compl2real, dudx_z_tmp, dudx_tmp);
	fftw_execute_dft_c2r(compl2real, u_z, u_tmp);

	// do the multiplicatoin in real space
	for (int i = 0; i < N; ++i)	{
		dudx_tmp[i] = -u_tmp[i]*dudx_tmp[i];
	}


	// now transform back to Fourier space
	fftw_execute_dft_r2c(real2compl, dudx_tmp, dudt_z);


	// normalize the transform
	for (int i = 0; i < N; ++i)	{
		dudt_z[i] *= norm_fact;
	}

}