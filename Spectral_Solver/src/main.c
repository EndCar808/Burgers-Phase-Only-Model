// Enda Carroll
// Sept 2019
// Main file for calling the pseudospectral solver for the 1D Burgers equation

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
#include "utils.h"
// #include "forcing.h"





// ---------------------------------------------------------------------
//  Main function
// ---------------------------------------------------------------------
int main(int argc, char** argv) {

	// Domain variables
	int N = atoi(argv[1]);
	
	// spatial variables
	double leftBoundary  = 0.0;
	double rightBoundary = 2.0*M_PI;
	double dx            = (leftBoundary - rightBoundary) / (double)N;

	
	// time varibales
	double t0 = 0.0;
	double T  = 1.0;
	double dt = 1e-3;


	// wavenumbers
	int* kx;
	kx = (int* )malloc(N*sizeof(int));


	// soln vectors
	double* u;
	u = (double* )malloc(N*sizeof(double));

	fftw_complex* u_z;
	u_z	= (fftw_complex* )fftw_malloc((N/2 + 1)*sizeof(fftw_complex));



	// ------------------------------
	//  FFTW plans
	// ------------------------------
	
	// create fftw3 plans objects
	fftw_plan fftw_plan_r2c, fftw_plan_c2r;

	// create plans - ensure no overwriting - fill arrays after
	fftw_plan_r2c = fftw_plan_dft_r2c_1d(N, u, u_z, FFTW_PRESERVE_INPUT); 
	fftw_plan_c2r = fftw_plan_dft_c2r_1d(N, u_z, u, FFTW_PRESERVE_INPUT);




	// ------------------------------
	//  Generate initial conditions
	// ------------------------------
	initial_condition(u, u_z, fftw_plan_r2c, dx, N);


	return 0;
}