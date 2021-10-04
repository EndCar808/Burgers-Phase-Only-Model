#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <complex.h>
#include <fftw3.h>


int main(int argc, char **argv)
{
	
	// vars
	int N       = 16;
	int num_osc = N / 2 + 1;


	double* phi = (double*) malloc(sizeof(double) * num_osc);
	double* amp = (double*) malloc(sizeof(double) * num_osc);

	fftw_complex* u_z = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * num_osc);

	double* u = (double* ) malloc(sizeof(double) * N);
	double* u1 = (double* ) malloc(sizeof(double) * N);

	fftw_plan fftw_r2c, fftw_c2r;
	fftw_r2c = fftw_plan_dft_r2c_1d(N, u, u_z, FFTW_PRESERVE_INPUT); 
	fftw_c2r = fftw_plan_dft_c2r_1d(N, u_z, u, FFTW_PRESERVE_INPUT);

	for (int i = 0; i < N; ++i) {
		u[i] = 0.0;
		u1[i] = 0.0;
		if( i < num_osc) {
			if (i > 0) {
				phi[i] = M_PI / 4.0;
				amp[i] = 1.0 / pow(i, 1.0);
				u_z[i] = amp[i] * cexp(I * phi[i]);
			}
			else{
				phi[i] = 0.0;
				amp[i] = 0.0;
				u_z[i] = 0.0 + I * 0.0;
			}
		}		
	}

	for (int i = 0; i < num_osc; ++i)	{
		printf("u_z[%d]: %6.16lf %+6.16lf\n", i, creal(u_z[i]), cimag(u_z[i]));
	}
	printf("\n\n");


	
	fftw_execute_dft_c2r(fftw_c2r, u_z, u);

	for (int i = 0; i < N; ++i) {
		printf("u[%d]: %6.15lf \n", i, u[i]);
	}
	printf("\n\n");

	// Set last mode to 0 due to reality of u
	u_z[num_osc - 1] = 0.0 + 0.0 * I;

	fftw_execute_dft_c2r(fftw_c2r, u_z, u1);

	for (int i = 0; i < N; ++i) {
		printf("u[%d]: %6.15lf \n", i, u1[i]);
	}
	printf("\n\n");


	for (int i = 0; i < N; ++i) {
		printf("err[%d]: %6.16lf\n", i, fabs(u[i] - u1[i]));
	}
	printf("\n\n");

	for (int i = 0; i < count; ++i)
	{
		/* code */
	}

	free(phi);
	free(amp);
	free(u);
	fftw_free(u_z);

	return 0;
}