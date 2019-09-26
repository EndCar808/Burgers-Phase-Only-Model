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


// ---------------------------------------------------------------------
//  User Libraries and Headers
// ---------------------------------------------------------------------
// #include "forcing.h"



// ---------------------------------------------------------------------
//  Function Definitions
// ---------------------------------------------------------------------
void initial_condition(double* u, fftw_complex* u_z, fftw_plan real2compl, double dx, int N) {

	double x;

	// fill real solution array
	for(int i = 0; i < N; ++i) {
		x = 0 + (double)i*dx;

		u[i] = -sin(x); 
	}

	fftw_execute(real2compl);

}