// Enda Carroll
// May 2020
// Main file for calling the pseudospectral solver for the 1D Burgers equation

// ---------------------------------------------------------------------
//  Standard Libraries and Headers
// ---------------------------------------------------------------------
#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <string.h>
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
#include "stats.h"



// ---------------------------------------------------------------------
//  Main function
// ---------------------------------------------------------------------
int main(int argc, char** argv) {


	// ------------------------------
	//  Setup 
	// ------------------------------
	// Start timer
	clock_t begin = clock();

	// Initialize Default Variables
	int c;
	// Space Variables
	int N    = 64;
	int k0   = 1;
	int Nmax = N * 2;
	// Slope variables
	double alpha = 1.0;
	double beta  = 0.0;
	// Initial Condition
	char u0[128];
	strcpy(u0, "RANDOM");
	// Time steps
	int tsteps     = 4e5;
	int save_steps = 1e2;
	int compute_steps = 1e2;

	// Read in Command line arguements
	while ((c = getopt(argc, argv, "n:k:a:b:u:t:s:c:")) != -1) {
		switch (c) {
			case 'n':
				// Get the number of collocation points
				N = atoi(optarg);
				if (N < 0) {
					fprintf(stderr, "The number of collocation points N must be positive, value given %d\n-->> Now Exiting!\n\n", N);
					exit(1);
				}
				break;
			case 'k':
				// Get the value of k0 -> the number of the first k0 wavenumbers to set to 0
				k0 = atoi(optarg);
				if (k0 < 0) {
					fprintf(stderr, "k0 must be positive, value given %d\n-->> Now Exiting!\n\n", N);
					exit(1);	
				}
				break;
			case 'a':
				// Get spectrum slope alpha
				alpha = atof(optarg);
				if (alpha < 0.0) {
					fprintf(stderr, "Spectrum Slope Alpha must be positive, value given %lf\n-->> Now Exiting!\n\n", alpha);
					exit(1);
				}
				break;
			case 'b':
				// Get spectrum slope alpha
				beta = atof(optarg);
				if (beta < 0.0) {
					fprintf(stderr, "Spectrum Fall off Beta must be positive, value given %lf\n-->> Now Exiting!\n\n", beta);
					exit(1);
				}
				break;
			case 'u':
				// Get the initial condition
				strncpy(u0, optarg, 128);
				break;
			case 't':
				tsteps = atoi(optarg);
				if (tsteps < 0.0) {
					fprintf(stderr, "The number of integration steps must be positive, value given %d\n-->> Now Exiting!\n\n", tsteps);
					exit(1);
				}
				break;
			case 's':
				save_steps = atoi(optarg);
				if (save_steps < 0.0) {
					fprintf(stderr, "The number of integration steps to perform before saving must be positive, value given %d\n-->> Now Exiting!\n\n", save_steps);
					exit(1);
				}
				break;
			case 'c':
				compute_steps = atoi(optarg);
				if (compute_steps < 0.0) {
					fprintf(stderr, "The number of integration steps to perform before saving must be positive, value given %d\n-->> Now Exiting!\n\n", compute_steps);
					exit(1);
				}
				break;
			default:
				fprintf(stderr, "\n[ERROR] Incorrect command line flag encountered\n");		
				fprintf(stderr, "Use"" -n"" to specify the number of collocation points\n");
				fprintf(stderr, "Use"" -k"" to specify k0\n");
				fprintf(stderr, "Use"" -a"" to specify the Spectrum slope alpha\n");
				fprintf(stderr, "Use"" -b"" to specify the spectrum fall off beta\n");
				fprintf(stderr, "Use"" -t"" to specify the number of integration steps\n");
				fprintf(stderr, "Use"" -u"" to specify the intial condition\n");
				fprintf(stderr, "Use"" -s"" to specify the number of integration steps before saving to file\n");
				fprintf(stderr, "Use"" -c"" to specify the number of integration steps before performing computation of runtime data\n");
				fprintf(stderr, "\nExample usage:\n\t./bin/main -n 64 -k 1 -a 1.0 -b 0.0 -u RANDOM -t 4000000 -s 100 -c 100\n");
				fprintf(stderr, "\n-->> Now Exiting!\n\n");
				exit(1);
		}
	}


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
	int flag = solver(N, k0, alpha, beta, tsteps, save_steps, compute_steps, u0);
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