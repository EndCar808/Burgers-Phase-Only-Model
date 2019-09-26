// Enda Carroll
// Sept 2019
// Programme to run through some examples using FFTW on real data - transfrom from real to complex and back

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

	int N = 100;

	// Allocate memory
	double* u;
	u = (double* )malloc(N*sizeof(double));

	fftw_complex* u_z;
	u_z = (fftw_complex* )fftw_malloc((N/2 + 1)*sizeof(fftw_complex)); // only  N/2 + 1 values returned due to Hermittian property

	// Alternate way - with padding
	double* u_g;
	u_g    = fftw_alloc_real(N);
	double complex* u_z_g;
	u_z_g = fftw_alloc_complex(N); // these are wrapper functions that preform exactly the same mem allocation as above
								   // these ensure 16 byte aligned memory to take advantage of SIMD operations in the fftw lib
								   // both ways work fine see doc for more detail

	// create two plans - one for transforming forward, one for backward
	fftw_plan fftw_plan_r2c, fftw_plan_c2r;

	fftw_plan_r2c = fftw_plan_dft_r2c_1d(N, u, u_z, FFTW_PRESERVE_INPUT); // FFTW_PRESERVE_INPUT forces fftw to not overwrtie input array
	fftw_plan_c2r = fftw_plan_dft_c2r_1d(N, u_z, u, FFTW_PRESERVE_INPUT);


	fftw_plan fftw_plan_r2c_g, fftw_plan_c2r_g;

	fftw_plan_r2c_g = fftw_plan_dft_r2c_1d(N, u_g, u_z_g, FFTW_PRESERVE_INPUT);
	fftw_plan_c2r_g = fftw_plan_dft_c2r_1d(N, u_z_g, u_g, FFTW_PRESERVE_INPUT);


	// fill real array
	double x;
	for(int i = 0; i < N; ++i) {
		x = (double) i/N;
		u[i] = -sin(x);
		u_g[i] = -sin(x);
		printf("u[%d] = %.16lf\n", i, u[i]);
	}



	//////////////////////
	/// Simple r2c and c2r
	//////////////////////

	// execute r2c transform
	fftw_execute(fftw_plan_r2c);

	for(int i = 0; i < N/2 + 1; ++i) {
		printf("u[%d] = %.lf %lfI\n", i, creal(u_z[i]), cimag(u_z[i]));
	}


	// execute backwards from c2r
	fftw_execute(fftw_plan_c2r);

	for(int i = 0; i < N; ++i) {
		printf("u[%d] = %.16lf \n", i, u[i] / N); // need to renormalize (dividing by N) after transforming back
	}


	///////////////////////////////////
	/// Simple r2c and c2r with padding
	///////////////////////////////////
	
	fftw_execute(fftw_plan_r2c_g);

	for(int i = 0; i < N; ++i) {
		printf("u_z_g[%d] = %.lf %lfI\n", i, creal(u_z[i]), cimag(u_z[i]));
	}


	// execute backwards from c2r
	fftw_execute(fftw_plan_c2r);

	for(int i = 0; i < N; ++i) {
		printf("u_z[%d] = %.16lf \n", i, u[i] / N); // need to renormalize (dividing by N) after transforming back
	}
 


	fftw_destroy_plan(fftw_plan_r2c);
	fftw_destroy_plan(fftw_plan_c2r);

	fftw_free(u);
	fftw_free(u_z);

	return 0;
}