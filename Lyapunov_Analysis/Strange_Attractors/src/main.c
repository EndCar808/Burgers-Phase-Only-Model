// Enda Carroll
// Sept 2020
// Main function file for calling the Benettin et al. and Ginelli et al., algorithms
// for computing the Lyapunov spectrum and vectors of the Lorenz system

// ---------------------------------------------------------------------
//  Standard Libraries and Headers
// ---------------------------------------------------------------------
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include <hdf5.h>
#include <hdf5_hl.h>
#include <gsl/gsl_cblas.h>
#include <lapacke.h>


// ---------------------------------------------------------------------
//  User Libraries and Headers
// ---------------------------------------------------------------------
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


	// System dimension
	int N = 3;
	int numLEs = N;

	// Integration parameters
	int m_trans = 1000;
	int m_rev_trans = m_trans;
	int m_end  = 60000000;
	int m_iter = 1;


	// ------------------------------
	//  Compute Spectrum
	// ------------------------------
	compute_lce_spectrum(N, numLEs, m_trans, m_rev_trans, m_end, m_iter);
	// ------------------------------
	//  Compute Spectrum
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