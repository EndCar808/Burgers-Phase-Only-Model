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
#include <hdf5.h>
#include <hdf5_hl.h>


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


void write_array(double *A, int n, char *filename) {

	int i;

	FILE *ft = fopen( filename, "w");

	for ( i = 0; i < n; i++ ) {
		fprintf(ft, "%f ",  A[i]);
	}

	fclose(ft);
}


void deriv(double complex* u_z, double complex* dudt_z, double nu, int* k, fftw_plan *real2compl_ptr, fftw_plan *compl2real_ptr, int N) {

	double norm_fact = 1 / (double) N;

	// Allocate temporary memory
	double complex* dudx_z_tmp;
	dudx_z_tmp = (double complex* )fftw_malloc((N/2 + 1)*sizeof(double complex));
	double* dudx_tmp;
	dudx_tmp   = (double* )malloc(N*sizeof(double));
	double* u_tmp;
	u_tmp      = (double* )malloc(N*sizeof(double));
	

	// find the derivative - dudx - in complex space
	for (int i = 0; i < N/2 + 1; ++i) {
		dudx_z_tmp[i] = I*k[i]*u_z[i];
	}

	// transform this derivative back using "new array execute" functionality of FFTW
	fftw_execute_dft_c2r(*compl2real_ptr, dudx_z_tmp, dudx_tmp);
	fftw_execute_dft_c2r(*compl2real_ptr, u_z, u_tmp);

	// do the multiplicatoin in real space
	for (int i = 0; i < N; ++i)	{
		dudx_tmp[i] = -u_tmp[i]*dudx_tmp[i];
	}


	// now transform back to Fourier space
	fftw_execute_dft_r2c(*real2compl_ptr, dudx_tmp, dudt_z);


	// add the viscous term and normalize the transform
	for (int i = 0; i < N/2 + 1; ++i)	{
		dudt_z[i] -= nu*k[i]*k[i]*u_z[i];
		dudt_z[i] *= norm_fact;
	}


	//free memory
	free(dudx_tmp);
	free(u_tmp);
	fftw_free(dudx_z_tmp);
}



double system_energy(double complex* u_z, int N) {

	double sys_energy;

	// initialize the energy sum
	sys_energy = u_z[0]*conj(u_z[0]); // 0th mode does not have a conj by definion

	// loop over modes and find the total energy
	for (int i = 1; i < N/2 + 1; ++i)	{
		sys_energy += 2.0*u_z[i]*conj(u_z[i]);   // account for the Reality condition u_{-k} = - u_{k}
	}

	return sys_energy;
}


double system_enstrophy(double complex* u_z, int* k, int N) {

	double sys_enstrophy;

	// initialize enstrophy sum
	sys_enstrophy = cabs(I*k[0]*u_z[0])*cabs(I*k[0]*u_z[0]);

	// loop over modes
	for (int i = 1; i < N/2 + 1; ++i) {
		sys_enstrophy += cabs(I*k[0]*u_z[0])*cabs(I*k[0]*u_z[0]);
	}

	return sys_enstrophy;
}