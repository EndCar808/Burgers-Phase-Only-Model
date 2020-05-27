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


	// // Get the number of threads 
	int n_threads = atoi(argv[2]);


	// set number of threads
	omp_set_num_threads(n_threads);
	
	printf("\n\tNumber of OpenMP Threads running = %d\n\n" , omp_get_max_threads());
	
	// Initialize and set threads for fftw plans
	fftw_init_threads();
	fftw_plan_with_nthreads((int)omp_get_max_threads());


	// Create the HDF5 file handle
	hid_t HDF_Outputfile_handle;



	// ------------------------------
	//  Call Solver Here
	// ------------------------------
	solver(&HDF_Outputfile_handle, N, 1, 1.0, 0.0, 1e2, 1, "ALIGNED");
	// ------------------------------
	//  Call Solver Here
	// ------------------------------
	

	
	// ------------------------------
	//  Close and exit
	// ------------------------------
	// Close pipeline to output file
	H5Fclose(HDF_Outputfile_handle);
	
	// Finish timing
	clock_t end = clock();

	// calculate execution time
	double time_spent = (double)(end - begin) / CLOCKS_PER_SEC;

	printf("\n\tExecution Time: %20.16lf\n", time_spent);
	printf("\n\n");

	return 0;
}