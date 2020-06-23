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


// ---------------------------------------------------------------------
//  User Libraries and Headers
// ---------------------------------------------------------------------
// #include "data_types_solver.h"
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

	int k0 = 1;

	double alpha = 1.0;
	double beta  = 1.0;

	int tsteps     = 1e5;
	int save_steps = 1;


	// // Get the number of threads 
	int n_threads = atoi(argv[2]);


	// set number of threads
	omp_set_num_threads(n_threads);
	
	printf("\n\tNumber of OpenMP Threads running = %d\n\n" , omp_get_max_threads());
	
	// Initialize and set threads for fftw plans
	fftw_init_threads();
	fftw_plan_with_nthreads((int)omp_get_max_threads());


	// ------------------------------
	//  Call Solver Here
	// ------------------------------
	solver(N, k0, alpha, beta, tsteps, save_steps, "ALIGNED");
	// ------------------------------
	//  Call Solver Here
	// ------------------------------
	

	
	// Finish timing
	clock_t end = clock();

	// calculate execution time
	double time_spent = (double)(end - begin) / CLOCKS_PER_SEC;

	printf("\n\tExecution Time: %20.16lf\n", time_spent);
	printf("\n\n");

	return 0;
}