// Enda Carroll
// Sept 2019
// Programme to run through some examples using FFTW

// ---------------------------------------------------------------------
//  Standard Libraries and Headers
// ---------------------------------------------------------------------
#include <stdio.h>
#include <stdlib.h>
#include <math.h>


// ---------------------------------------------------------------------
//  User Libraries and Headers
// ---------------------------------------------------------------------
#include <fftw3.h> // include the latest FFTW library


 
// ---------------------------------
//  Main Programme
// ---------------------------------
int main(int argc, char** argv) {

	// size of array
	int N; 

	// pointers of type fftw_complex
	fftw_complex* in;
	fftw_complex* out;

	// create fftw plan environment
	fftw_plan my_plan;


	// Allocate memory
	in  = fftw_malloc(N*sizeof(fftw_complex));
	out = fftw_malloc(N*sizeof(fftw_complex));


	// Create plan 
	my_plan = fftw_plan_dft_1d(N, in, out, FFTW_FORWARD, FFTW_ESTIMATE);


	// execute plan
	fftw_execute(my_plan);


	// Destroy plan
	fftw_destroy_plan(my_plan);

	fftw_free(in);
	fftw_free(out);

	return 0;
}