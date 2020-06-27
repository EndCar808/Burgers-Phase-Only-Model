// Enda Carroll
// Sept 2019
// Programme to run through some examples using FFTW

// ---------------------------------------------------------------------
//  Standard Libraries and Headers
// ---------------------------------------------------------------------
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <complex.h>


// ---------------------------------------------------------------------
//  User Libraries and Headers
// ---------------------------------------------------------------------
#include <fftw3.h> // include the latest FFTW library



int main( int argc, char** argv) {

	// size of array
	int N = 10;

	// allocate memory
	fftw_complex* in;
	fftw_complex* out;
	in  = (fftw_complex* )fftw_malloc( N*sizeof(fftw_complex));
	out = (fftw_complex* )fftw_malloc( N*sizeof(fftw_complex));

	// create plan
	fftw_plan my_plan;

	// determin dft and store in plan - arrays do not need to be initialized to determine plan
	my_plan = fftw_plan_dft_1d(N, in, out, FFTW_FORWARD, FFTW_ESTIMATE);


	//fill arrays
	printf("\nx_k:\n\n");
	for(int i = 0; i < N; ++i) {
		in[i] = (i + 1.0) + (3.0*i - 1.0)*I;
		printf("%d %11.7f %11.7f\n", i, creal(in[i]), cimag(in[i]));
	}


	// perform fft
	fftw_execute(my_plan);

	// print output
	printf("\nxhat_k:\n\n");
	for(int i = 0; i < N; ++i) {
		printf("%d %11.7f %11.7f\n", i, creal(out[i]), cimag(out[i]));
	}



	printf("\n\n");



	//fill arrays
	printf("\nx_k:\n\n");
	for(int i = 0; i < N; ++i) {
		in[i] = exp(-i);
		printf("%d %11.7f %11.7f\n", i, creal(in[i]), cimag(in[i]));
	}


	// perform fft again but on the new data
	fftw_execute(my_plan);

	// print output
	printf("\nxhat_k:\n\n");
	for(int i = 0; i < N; ++i) {
		printf("%d %11.7f %11.7f\n", i, creal(out[i]), cimag(out[i]));
	}	



	// destroy plan and clean up memory
	fftw_destroy_plan(my_plan);

	fftw_free(in);
	fftw_free(out);

	return 0;
}

