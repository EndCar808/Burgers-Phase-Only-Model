// Enda Carroll
// May 2020
// Main file for calling the pseudospectral solver for the 1D Burgers equation

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
#include <lapacke.h>


// ---------------------------------------------------------------------
//  User Libraries and Headers
// ---------------------------------------------------------------------
#include "lce_spectrum.h"
#include "utils.h"
#include "solver.h"



// ---------------------------------------------------------------------
//  Main function
// ---------------------------------------------------------------------
int main(int argc, char** argv) {


	// ------------------------------
	//  Setup 
	// ------------------------------
	// Start timer
	clock_t begin = clock();


	// Collocation points
	int N = atoi(argv[1]);
	

	// Spectrum variables
	int k0 = 0;
	double alpha = 1.0;
	double beta  = 1.0;


	// Integration parameters
	int m_end  = 8000;
	int m_iter = 10;

	// Perturbation
	double pert = 10;



	// ------------------------------
	//  Compute Lyapunov Spectrum
	// ------------------------------
	compute_spectrum(N, k0, alpha, beta, m_end, m_iter, pert);
	// ------------------------------
	//  Compute Lyapunov Spectrum
	// ------------------------------
	
	
	// Finish timing
	clock_t end = clock();

	// calculate execution time
	double time_spent = (double)(end - begin) / CLOCKS_PER_SEC;

	printf("\n\tExecution Time: %20.16lf\n", time_spent);
	printf("\n\n");

	return 0;
}