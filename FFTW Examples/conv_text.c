// Enda Carroll
// Sept 2019
// Programme to test the convolution term

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

	int N = 32;

	// Allocate memory
	double* u;
	u = (double* )malloc(N*sizeof(double));

	double complex* conv;
	conv = (double complex*)malloc((N/2 + 1)*sizeof(double));

	fftw_complex* u_z;
	u_z = (fftw_complex* )fftw_malloc((N + 1)*sizeof(fftw_complex)); // only  N/2 + 1 values returned due to Hermittian property

	// create two plans - one for transforming forward, one for backward
	fftw_plan fftw_plan_r2c, fftw_plan_c2r;

	fftw_plan_r2c = fftw_plan_dft_r2c_1d(N, u, u_z, FFTW_PRESERVE_INPUT); // FFTW_PRESERVE_INPUT forces fftw to not overwrtie input array
	fftw_plan_c2r = fftw_plan_dft_c2r_1d(N, u_z, u, FFTW_PRESERVE_INPUT);



	// fill the complex array
	// u_z[0] = 0.0 + 0.0*I;
	// for (int i = 1; i < N/2 + 1; ++i)
	// {
	// 	u_z[i] = pow((double)i, -1) * (cos(M_PI/2) + I*sin(M_PI/2));
	// 	u_z[N - i + 1] = pow((double)i, -1) * (cos(M_PI/2) - I*sin(M_PI/2));
	// }
	// for (int i = 0; i < N + 1; ++i)
	// {
	// 	printf("u_z[%d]: %lf  %lfi\n", i, creal(u_z[i]), cimag(u_z[i]));
	// }


	// // Perform convolution directly
	// conv[0] = 0.0;
	// for (int k = 1; k < N/2 + 1; ++k)
	// {	
	// 	conv[k] = 0.0 + 0.0*I;
	// 	for (int k1 = 0; k1 < N; ++k1)
	// 	{
	// 		if ( abs(k - k1) < N/2 ) {
	// 			conv[k] += u_z[k1]*u_z[k - k1] + conj(u_z[k1])*u_z[k + k1];
	// 		} else {
	// 			conv[k] += 0.0 + 0.0*I;
	// 		}
	// 	}
	// }
	// for (int i = 0; i < N/2 + 1; ++i)
	// {
	// 	printf("cov[%d]: %lf  %lfi\n", i, creal(conv[i]), cimag(conv[i]));
	// }



	////// Compare fftw with MATLAB
	double* uhat;
	uhat = (double* )malloc(N*sizeof(double));
	fftw_complex* uhat_z;
	uhat_z = (fftw_complex* )fftw_malloc((N/2 + 1)*sizeof(fftw_complex)); // only  N/2 + 1 values returned due to Hermittian property

	fftw_plan_r2c = fftw_plan_dft_r2c_1d(N, uhat, uhat_z, FFTW_PRESERVE_INPUT); // FFTW_PRESERVE_INPUT forces fftw to not overwrtie input array
	fftw_plan_c2r = fftw_plan_dft_c2r_1d(N, uhat_z, uhat, FFTW_PRESERVE_INPUT);
		

	// fill modes
	printf("\nFill modes\n");
	for (int i = 0; i < N/2 +1; ++i)
	{
		if(i == 0|| i == N/2) {
			uhat_z[i] = 0.0 + 0.0*I;
		} else {
			uhat_z[i] = pow((double)i, -1) * (cos(M_PI/2) + I*sin(M_PI/2));
		}
		printf("uhat_z[%d]: %lf  %lfi\n", i, creal(uhat_z[i]), cimag(uhat_z[i]));
	}

	// Transform back to real
	fftw_execute(fftw_plan_c2r);

	// check real
	printf("\n\nReal Space\n");
	for (int i = 0; i < N; ++i)
	{
		printf("uhat[%d]: %lf\n",i, uhat[i]/32.0 );
		uhat[i] *= uhat[i];

	}
	printf("\n\nReal Space ^2\n");
	for (int i = 0; i < N; ++i)
	{
		printf("uhat[%d]: %lf\n",i, uhat[i]);
	}
	

	//Transform forward
	fftw_execute(fftw_plan_r2c);

	// normalize and print
	printf("\n\nComplete\n");
	for (int i = 0; i < N/2 + 1; ++i)
	{
		uhat_z[i] *= 1.0/((double)N);
		printf("uhat_z[%d]: %lf  %lfi\n", i, creal(uhat_z[i]), cimag(uhat_z[i]));
	}


	// Free memory and destroy plans
	fftw_destroy_plan(fftw_plan_r2c);
	fftw_destroy_plan(fftw_plan_c2r);

	fftw_free(u);
	fftw_free(u_z);

	fftw_free(uhat);
	fftw_free(uhat_z);

	return 0;
}