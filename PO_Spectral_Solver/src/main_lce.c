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
#include "utils.h"
#include "solver.h"
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

	int k0 = 1;

	double a = 1.0;
	double b = 1.0;

	char* u0 = "ALIGNED";

	int m_end  = 4000;
	int m_iter = 50;


	// parallel thread variables
	int i;
	int tid;
	int nthreads; 
	int chunk = 1;

	// alpha vars
	double a_end   = 2.5;
	double a_start = 0.0;
	double a_step  = 0.05;
	int a_len      = (int)((a_end - a_start) / a_step) + 1; 

	// array of alpha variables
	// double* alpha = (double* )malloc(sizeof(double) * a_len);
	// for (int i = 0; i < a_len; ++i)	{
	// 	alpha[i] = i * a_step;
	// }
	// a_len = 2;
	// double alpha[a_len];
	// alpha[0] = 0.0;
	// alpha[1] = 1.0; 



	// set number of threads
	// omp_set_num_threads(a_len);
	
	// printf("\n\tNumber of OpenMP Threads running = %d\n\n" , omp_get_max_threads());
	

	// #pragma omp parallel shared(N, a, b, u0, k0, m_end, m_iter, alpha, tid, chunk) private(i) 
	// {

	// 	// Obtain and print thread id 
	// 	tid = omp_get_thread_num();
		

	// 	#pragma omp for schedule(dynamic, chunk) nowait 
			// for (i = 0; i < a_len; ++i) {

				// print update
				// printf("Thread[%d] working on Alpha[%d]: %5.5lf\n", tid, i, alpha[i]);

				// ------------------------------
				//  Compute Spectrum
				// ------------------------------
				compute_lce_spectrum(N, a, b, u0, k0, m_end, m_iter);
				// ------------------------------
				//  Compute Spectrum
				// ------------------------------
			// }
		
	// }  

	


	// Finish timing
	clock_t end = clock();

	// calculate execution time
	double time_spent = (double)(end - begin) / CLOCKS_PER_SEC;

	printf("\n\tExecution Time: %20.16lf\n", time_spent);
	printf("\n\n");


	return 0;
}