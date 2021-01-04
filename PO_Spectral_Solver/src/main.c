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
#include <gsl/gsl_histogram.h>

// ---------------------------------------------------------------------
//  User Libraries and Headers
// ---------------------------------------------------------------------
#include "data_types.h"
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

	int k0 = atoi(argv[2]);


	double alpha = atof(argv[3]);
	double beta  = atof(argv[4]);

	// Initial Condition
	char u0[128];
	strcpy(u0, argv[5]);

	// Time steps
	int tsteps     = atoi(argv[6]);
	int save_steps = SAVE_DATA_STEP;

	int Nmax = N * 2;


	// Get the number of threads 
	int n_threads = 1;

	// set number of threads
	omp_set_num_threads(n_threads);
	
	printf("\n\tNumber of OpenMP Threads running = %d\n\n" , omp_get_max_threads());
	
	// Initialize and set threads for fftw plans
	fftw_init_threads();
	fftw_plan_with_nthreads((int)omp_get_max_threads());



	#ifdef __FXD_PT_SEARCH__
	// ------------------------------
	//  Call Searching Algo
	// ------------------------------
	fixed_point_search(N, Nmax, k0, alpha, beta, tsteps, save_steps, u0);
	// ------------------------------
	//  Call Searching Algo
	// ------------------------------
	#else
	// ------------------------------
	//  Call Solver Here
	// ------------------------------
	int flag = solver(N, k0, alpha, beta, tsteps, save_steps, u0);
	// ------------------------------
	//  Call Solver Here
	// ------------------------------
	#endif


	// Clean up FFTW
	fftw_cleanup_threads();
	fftw_cleanup();
	
	// Finish timing
	clock_t end = clock();

	// calculate execution time
	double time_spent = (double)(end - begin) / CLOCKS_PER_SEC;

	printf("\n\n\tTotal Execution Time: %20.16lf\n\n", time_spent);

	return 0;
}
// ---------------------------------------------------------------------
//  End of File
// ---------------------------------------------------------------------