// Enda Carroll
// Sept 2019
// Programme to run through some examples using FFTW

// ---------------------------------------------------------------------
//  Standard Libraries and Headers
// ---------------------------------------------------------------------
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <complex.h> // having this before <fftw3.h> means fftw_complex is the native double-precision complex
					 // type and you can manipulate it with ordinary arithmetic. Otherwise FFTW defines its own complex type


// ---------------------------------------------------------------------
//  User Libraries and Headers
// ---------------------------------------------------------------------
#include <fftw3.h> // include the latest FFTW library


 
// ---------------------------------
//  Main Programme
// ---------------------------------
int main(int argc, char** argv) {

	// size of array
	int N = 10; 

	// pointers of type fftw_complex
	fftw_complex* in_arr;
	fftw_complex* out_arr;

	// create fftw plan to store the type of FFT we want to perform
	fftw_plan my_plan;


	// Allocate memory
	in_arr  = (fftw_complex* )fftw_malloc(N*sizeof(fftw_complex)); // must cast as (ffftw_complex* )
	out_arr = (fftw_complex* )fftw_malloc(N*sizeof(fftw_complex));


	// Specify the type of plan we want to perform
	/*
	Function inputs:
	N - size of the input/output arrays
	in_arr - input array
	out_arr - input array
			  These arrays can be equal. If so this indicates a 'transform in place' and the input array is overwritten with the output
	FFTW_FORWARD - determines to sign of the exponent of the DFT - FFTW_FORWARD = -1; FFTW_BACKWARD = +1;
	FFTW_ESTIMATE - flag to determine optimization - is typically FFTW_ESTIMATE or FFTW_MEASURE(can be others); FFTW_MEASURE performs a check
	by running different dft's to find the fastest out for the given N - this is takes a few seconds but is useful if you need to 
	perform many dft's throughout the programme. FFTW_ESTIMATE does not perform any check
	NOTE: You should always create plans before initializing arrays - FFTW_MEASURE will overwrite these arrays 
	 */
	my_plan = fftw_plan_dft_1d(N, in_arr, out_arr, FFTW_FORWARD, FFTW_ESTIMATE);


	// execute plan - essentially performs the dft outlined in the plan
	fftw_execute(my_plan); // can perform this multiple times


	// Destroy plan
	fftw_destroy_plan(my_plan);

	fftw_free(in);
	fftw_free(out);

	return 0;
}