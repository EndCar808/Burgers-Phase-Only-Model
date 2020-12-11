// Enda Carroll
// May 2020
// Main function file for calling the Benettin et al., algorithm
// for computing the Lyapunov spectrum of the Phase Only 1D Burgers equation

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
#include "data_types.h"
#include "utils.h"
#include "lce_spectrum.h"



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

	// Alpha value
	double alpha = atof(argv[3]);
	double beta  = atof(argv[4]);
	
	// Kill first k0 modes
	int k0 = atoi(argv[2]);;

	// Specify initial condition
	char* u0[128];
	strcpy(u0, argv[5]);

	// Integration parameters
	int m_end  = atoi(argv[6]);
	int m_iter = atoi(argv[7]);

	// Number of LEs
	int numLEs = (int) N / 2 - k0;
	if (argc == 9) {
		numLEs = atoi(argv[8]);
		if (numLEs > (int) N / 2 - k0) {
			printf("\nNo. of LEs [%d] is too big...setting to max [%d]\n\n", numLEs, (int) N / 2 - k0);
			numLEs = (int) N / 2 - k0;
		}
	}


	// ------------------------------
	//  Compute Spectrum & CLVs
	// ------------------------------
	compute_lce_spectrum_clvs(N, alpha, beta, u0, k0, m_end, m_iter, numLEs);
	// ------------------------------
	//  Compute Spectrum & CLVs
	// ------------------------------


	// Finish timing
	clock_t end = clock();

	// calculate execution time
	double time_spent = (double)(end - begin) / CLOCKS_PER_SEC;

	printf("\n\tExecution Time: %20.16lf\n", time_spent);
	printf("\n\n");


	return 0;
}
// ---------------------------------------------------------------------
//  End of File
// ---------------------------------------------------------------------