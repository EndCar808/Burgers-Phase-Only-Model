#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <complex.h>
#include <fftw3.h>



int sgn(int x) {

	int val = 0;

	if (x > 0) {
		val = 1;
	}
	else if(x < 0) {
		val = -1;
	}
	else if (x == 0) {
		val = 0;
	}

	return val;
}


void conv_2N_pad(fftw_complex* convo, fftw_complex* uz, fftw_plan *fftw_plan_r2c_ptr, fftw_plan *fftw_plan_c2r_ptr, int n, int num_osc, int k0) {

	// Padded resolution
	int m = 2*n;

	// Normalization factor
	double norm_fact = 1.0 / (double) m;

	// Allocate temporary arrays
	double* u_tmp = (double* )malloc(m*sizeof(double));
	// mem_chk(u_tmp, "u_tmp");
	fftw_complex* u_z_tmp = (fftw_complex* )fftw_malloc((2*num_osc - 1)*sizeof(fftw_complex));
	// mem_chk(u_z_tmp, "u_z_tmp");

	// write input data to padded array
	for (int i = 0; i < (2*num_osc - 1); ++i)	{
		if(i < num_osc){
			u_z_tmp[i] = uz[i];
		} else {
			u_z_tmp[i] = 0.0 + 0.0*I;
		}
	}

	// // transform back to Real Space
	fftw_execute_dft_c2r((*fftw_plan_c2r_ptr), u_z_tmp, u_tmp);

	// // square
	for (int i = 0; i < m; ++i)	{
		u_tmp[i] = pow(u_tmp[i], 2);
	}

	// here we move the derivative from real to spectral space so that the derivatives can be returned in spectral space
	fftw_execute_dft_r2c((*fftw_plan_r2c_ptr), u_tmp, u_z_tmp);

	// // normalize
	for (int i = 0; i < num_osc; ++i)	{
		if (i <= k0) {
			convo[i] = 0.0 + 0.0*I;
		} else {
			convo[i] = u_z_tmp[i]*(norm_fact);
		}		
	}

	///---------------
	/// Free temp memory
	///---------------
	free(u_tmp);
	fftw_free(u_z_tmp);
}






int main(int argc, char** argv) {

	// Parameter defs
	int N = atoi(argv[1]);
	int M = 2 * N;
	int num_osc = (int) N / 2 + 1;

	int k0 = atoi(argv[2]); 
	int kmin = k0 + 1;
	int kmax = num_osc - 1;

	double alpha = 1.5;
	// double beta = 0.0;
	

	// Mem alloc
	double* phi  = (double* )malloc(sizeof(double) * num_osc);
	double* amps = (double* )malloc(sizeof(double) * num_osc);
	fftw_complex* u_z            = (fftw_complex* )fftw_malloc(sizeof(fftw_complex) * num_osc);
	fftw_complex* conv           = (fftw_complex* )fftw_malloc(sizeof(fftw_complex) * num_osc);
	fftw_complex* phase_sync_ser = (fftw_complex* )fftw_malloc(sizeof(fftw_complex) * num_osc);
	fftw_complex* phase_sync_par = (fftw_complex* )fftw_malloc(sizeof(fftw_complex) * num_osc);
	fftw_complex* solver_sync    = (fftw_complex* )fftw_malloc(sizeof(fftw_complex) * num_osc);


	// padded solution arrays
	double* u_pad = (double* ) malloc(M * sizeof(double));
	fftw_complex* u_z_pad = (fftw_complex* ) fftw_malloc((2 * num_osc - 1) * sizeof(fftw_complex));
	// FFTW Plans
	fftw_plan fftw_plan_r2c_pad, fftw_plan_c2r_pad;
	fftw_plan_r2c_pad = fftw_plan_dft_r2c_1d(M, u_pad, u_z_pad, FFTW_PRESERVE_INPUT); 
	fftw_plan_c2r_pad = fftw_plan_dft_c2r_1d(M, u_z_pad, u_pad, FFTW_PRESERVE_INPUT);


	// Fill arrays
	for (int i = 0; i < num_osc; ++i) {
	 	if (i <= k0) {
	 		phi[i]  = 0.0;
	 		amps[i] = 0.0;
	 		u_z[i]  = 0.0 + 0.0 * I;
	 		conv[i] = 0.0 + 0.0 * I;
	 	}
	 	else {
	 		phi[i]  = M_PI / 4;
	 		amps[i] = 1 / (pow(i, alpha));	
	 		u_z[i]  = amps[i] * cexp(I * phi[i]);
	 		conv[i] = 0.0 + 0.0 * I;
	 	}
	 	phase_sync[i] = 0.0 + 0.0 * I;

	 	printf("a[%d]: %6.10lf\tphi[%d]: %6.10lf\tsync[%d]: %6.10lf + %6.10lf I\n", i, amps[i], i, phi[i], i, creal(phase_sync[i]), cimag(phase_sync[i]) );
	} 
	printf("\n\n");


	// Get the convolution
	conv_2N_pad(conv, u_z, &fftw_plan_r2c_pad, &fftw_plan_c2r_pad, N, num_osc, k0);


	// Compute solver phase sync
	for (int k = kmin; k < num_osc; ++k) {
		solver_sync[k] = I * (conv[k] * cexp(-I * phi[k])); 
		// printf("Solv[%d]:  %6.10lf + %6.10lf I\n", k, creal(solver_sync[k]), cimag(solver_sync[k]));
	}
	// printf("\n");

	// // Compute the ORDERED phase order parameter
	int k1;
	for (int k = kmin; k <= kmax; ++k) {
		// Loop over shitfed k1 domain
		
		for (int kk1 = 0; kk1 <= 2 * kmax - k; ++kk1) {
			// Readjust k1 to correct value
			k1 = kk1 - kmax + k;

			// Consider valid k1 values
			if( (abs(k1) >= kmin) && (abs(k - k1) >= kmin)) {
				// printf("(%d, %d, %d), ", k1, k-k1, -k);
				// printf("(%d, %d, %d), ", sgn(k - k1) * abs(k1), sgn(k1) * abs(k - k1),  -sgn(k1 * (k - k1)) * abs(k));
				phase_sync[k] +=  cexp(I * (sgn(k - k1) * phi[abs(k1)] + sgn(k1) * phi[abs(k - k1)] - sgn(k1 * (k - k1)) * phi[k])); //amps[abs(k1)] * amps[abs(k - k1)] * 
				// printf("p[%d]: %6.10lf + %6.10lf I\n", k, creal(phase_sync[k]), cimag(phase_sync[k]));
			}			
		}
		// printf("\n");
	}
	// printf("\n\n");

	for (int i = 0; i < num_osc; ++i) {
		printf("or[%d]: %6.10lf + %6.10lf I\n", i, creal(phase_sync[i]), cimag(phase_sync[i]));
	}





	/*// // Compute the UNORDERED phase order parameter
	int k1;
	for (int k = kmin; k <= kmax; ++k) {
		// Loop over shitfed k1 domain
		
		for (int kk1 = 0; kk1 <= 2 * kmax - k; ++kk1) {
			// Readjust k1 to correct value
			k1 = kk1 - kmax + k;

			// Consider valid k1 values
			if( (abs(k1) >= kmin) && (abs(k - k1) >= kmin)) {
				// printf("(%d, %d, %d), ", k1, k-k1, k);
				// printf("(%d, %d, %d), ", sgn(k - k1) * abs(k1), sgn(k1) * abs(k - k1),  -sgn(k1 * (k - k1)) * abs(k));
				phase_sync[k] += amps[abs(k1)] * amps[abs(k - k1)] * cexp(I * (sgn(k1) * phi[abs(k1)] + sgn(k - k1) * phi[abs(k - k1)] - phi[k]));
				printf("p[%d]: %6.10lf + %6.10lf I\n", k, creal(phase_sync[k]), cimag(phase_sync[k]));
			}			
		}
		phase_sync[k] *= I;
		// printf("\n");
	}
	printf("\n\n");




	for (int i = 0; i < num_osc; ++i) {
		printf("or[%d]: %6.10lf + %6.10lf I\n", i, creal(phase_sync[i]), cimag(phase_sync[i]));
	}

*/





	fftw_destroy_plan(fftw_plan_r2c_pad);
	fftw_destroy_plan(fftw_plan_c2r_pad);


	free(phi);
	free(amps);
	fftw_free(conv);
	fftw_free(u_z);
	fftw_free(u_pad);
	fftw_free(u_z_pad);
	fftw_free(phase_sync);
	fftw_free(solver_sync);

	return 0;
}