// Enda Carroll
// May 2020
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



int find_index(int* k_array, int val, int n);
void conv_3N2_pad(double complex* convo, double complex* uz, fftw_plan *fftw_plan_r2c_ptr, fftw_plan *fftw_plan_c2r_ptr, int n, int num_osc, int k0);
void conv_23(double complex* convo, double complex* uz, fftw_plan *fftw_plan_r2c_ptr, fftw_plan *fftw_plan_c2r_ptr, int n, int kmax, int k0);
void conv_2N_pad(double complex* convo, double complex* uz, fftw_plan *fftw_plan_r2c_ptr_2, fftw_plan *fftw_plan_c2r_ptr_2, int n, int num_osc, int k0);




int main( int argc, char** argv) {

	///-----------------
	///	Initialize vars
	///-----------------
	// Real space collocation points 
	int N       = 32;

	// Number of oscillators
	int num_osc = (N / 2) + 1;

	// Forcing wavenumber
	int k0 = 5;
	
	// Spectrum exonent
	double a = 2.23;

	// Aliasing vars
	int M    = 2 * N;
	int M3   = 3 * N / 2;
	int kmax = floor(N / 3) + 1;
	
	
	///-----------------
	///	Allocate memory
	///-----------------
	// Oscillator arrays
	double* amp;
	amp = (double*) malloc(num_osc*sizeof(double));
	double* phi;
	phi = (double*) malloc(num_osc*sizeof(double));

	// Wavenumbers
	int* k;
	k = (int*) malloc(N*sizeof(int));
	
	// Real space arrays
	double* u_3N2;
	u_3N2  = (double*) malloc(M3*sizeof(double));
	double* u_2N;
	u_2N   = (double*) malloc(M*sizeof(double));
	double* u_23;
	u_23   = (double*) malloc(N*sizeof(double));
	
	// Fourier space arrays
	fftw_complex* u_z;
	u_z     = (fftw_complex*) fftw_malloc(num_osc*sizeof(fftw_complex));
	fftw_complex* u_z_3N2;
	u_z_3N2 = (fftw_complex*) fftw_malloc((3*N/4 + 1)*sizeof(fftw_complex));
	fftw_complex* u_z_23;
	u_z_23  = (fftw_complex*) fftw_malloc(num_osc*sizeof(fftw_complex));
	fftw_complex* u_z_2N;
	u_z_2N  = (fftw_complex*) fftw_malloc(2*num_osc*sizeof(fftw_complex));
 
	// Convolution arrats
	fftw_complex* conv;
	conv  = (fftw_complex*) fftw_malloc(num_osc*sizeof(fftw_complex));
	fftw_complex* convo;
	convo = (fftw_complex*) fftw_malloc(num_osc*sizeof(fftw_complex));



	///-----------------
	///	Create DFT plans
	///-----------------
	fftw_plan fftw_plan_r2c_2N;
	fftw_plan fftw_plan_c2r_2N;

	fftw_plan fftw_plan_r2c_3N2;
	fftw_plan fftw_plan_c2r_3N2;
	
	fftw_plan fftw_plan_r2c_23;
	fftw_plan fftw_plan_c2r_23;
				
	fftw_plan_r2c_2N = fftw_plan_dft_r2c_1d(M, u_2N, u_z_2N, FFTW_PRESERVE_INPUT);  
	fftw_plan_c2r_2N = fftw_plan_dft_c2r_1d(M, u_z_2N, u_2N, FFTW_PRESERVE_INPUT);

	fftw_plan_r2c_3N2 = fftw_plan_dft_r2c_1d(M3, u_3N2, u_z_3N2, FFTW_PRESERVE_INPUT);  
	fftw_plan_c2r_3N2 = fftw_plan_dft_c2r_1d(M3, u_z_3N2, u_3N2, FFTW_PRESERVE_INPUT);

	fftw_plan_r2c_23 = fftw_plan_dft_r2c_1d(N, u_23, u_z_23, FFTW_PRESERVE_INPUT);  
	fftw_plan_c2r_23 = fftw_plan_dft_c2r_1d(N, u_z_23, u_23, FFTW_PRESERVE_INPUT);



	///-----------------
	///	Create oscillators
	///-----------------
	phi[0] = 0;
	amp[0] = 0;

	srand(123456789);

	for (int i = 0; i < num_osc; ++i)	{
		k[i] = i;
		if(i > 0) {
			amp[i] = pow((double)i, -a);
			// phi[i] = M_PI/2.0;	
			phi[i] = M_PI*( (double) rand() / (double) RAND_MAX);	
		}
		u_z[i] = amp[i]*(cos(phi[i]) + I*sin(phi[i]));
		
		// Set forcing here
		if(i <= k0) { /*|| i >= kmax) {*/
			u_z[i] = 0.0 + 0.0*I;
			amp[i] = 0.0;
			phi[i] = 0.0;
		}
		printf("k[%d]: %d | a: %5.15lf   p: %5.16lf   | u_z[%d]: %5.16lf  %5.16lfI \n", i, k[i], amp[i], phi[i], i, creal(u_z[i]), cimag(u_z[i]));
	}
	printf("\n\n");

	// Write data to file for comparison
	FILE* f;
	f = fopen("phases.dat", "w");
	for (int i = 0; i < num_osc; ++i) {
		fprintf(f, "%5.16lf ", phi[i]);
	}
	fclose(f);
	

	///------------------
	/// Test Convolution
	///------------------
	int k1;
	conv[0] = 0.0 + 0.0*I;
	for (int kk = k0 + 1; kk < num_osc; ++kk)	{
		printf("k: %d | ", kk);
		for (int k_1 = 1 + kk; k_1 < 2*num_osc; ++k_1)	{
			// Get correct k1 value
			if(k_1 < num_osc) {
				k1 = -num_osc + k_1;
			} else {
				k1 = k_1 - num_osc;
			}
			// Check we have the right wavenumber pairs
			// printf("{%d, %d}", k1, kk - k1);

			if (k1 < 0) {
				// printf("{%d, %d}",abs(k1), kk - k1);
				conv[kk] += conj(u_z[abs(k1)])*u_z[kk - k1]; 	
			} else if (kk - k1 < 0) {
				// printf("{%d, %d}", k1, abs(kk - k1));
				conv[kk] += u_z[k1]*conj(u_z[abs(kk - k1)]); 
			} else {
				// printf("{%d, %d}", k1, kk - k1);
				conv[kk] += u_z[k1]*u_z[kk - k1];
			}			
		}
		// if(kk >= kmax) {
		// 	conv[kk] = 0.0 + 0.0*I;
		// }
		// printf("\n");
		printf("conv[%d]:  %5.16lf  %5.16lfI \n", kk, creal(conv[kk]), cimag(conv[kk]) );
	}
	printf("\n\n");



	///------------------
	/// Test Conv funcs
	///------------------
	// conv_3N2_pad(convo, u_z, &fftw_plan_r2c_3N2, &fftw_plan_c2r_3N2, N, num_osc, k0);
	// for (int i = 0; i < num_osc; ++i)	{
	// 	printf("conv_3N2[%d]:  %5.16lf  %5.16lfI \n", i, creal(convo[i]), cimag(convo[i]) );
	// }
	// printf("\n\n");

	conv_2N_pad(convo, u_z, &fftw_plan_r2c_2N, &fftw_plan_c2r_2N, N, num_osc, k0);
	for (int i = 0; i < num_osc; ++i)	{
		if(i <= k0){
			convo[i] = 0.0 + 0.0*I;
		}
		printf("conv_2N[%d]:  %5.16lf  %5.16lfI \n", i, creal(convo[i]), cimag(convo[i]) );
	}
	printf("\n\n");
	
	// conv_23(convo, u_z, &fftw_plan_r2c_23, &fftw_plan_c2r_23, N, kmax, k0);
	// for (int i = 0; i < N; ++i)	{
	// 	printf("conv_23[%d]:  %5.16lf  %5.16lfI \n", i, creal(convo[i]), cimag(convo[i]) );
	// }
	


	///------------------
	/// Free memory
	///------------------
	fftw_destroy_plan(fftw_plan_c2r_23);
	fftw_destroy_plan(fftw_plan_r2c_23);
	fftw_destroy_plan(fftw_plan_c2r_2N);
	fftw_destroy_plan(fftw_plan_r2c_2N);
	fftw_destroy_plan(fftw_plan_c2r_3N2);
	fftw_destroy_plan(fftw_plan_r2c_3N2);

	free(amp);
	free(phi);
	free(k);
	free(u_23);
	free(u_2N);
	free(u_3N2);
	fftw_free(u_z);
	fftw_free(u_z_23);
	fftw_free(u_z_3N2);
	fftw_free(u_z_2N);
	fftw_free(convo);
	fftw_free(conv);


	
	return 0;
}



int find_index(int* k_array, int val, int n) {

	int indx = -1;

	for (int i = 0; i < n; ++i)	{
		if (k_array[i] == val) {
			indx = i;
			break;
		}
	}

	return indx;
}

void convolution_direct(fftw_complex* convo, fftw_complex* u_z, int n, int k0) {
	
	// Set the 0 to k0 modes to 0;
	for (int i = 0; i <= k0; ++i) {
		convo[0] = 0.0 + 0.0*I;
	}
	
	// Compute the convolution on the remaining wavenumbers
	int k1;
	for (int kk = k0 + 1; kk < n; ++kk)	{
		for (int k_1 = 1 + kk; k_1 < 2*n; ++k_1)	{
			// Get correct k1 value
			if(k_1 < n) {
				k1 = -n + k_1;
			} else {
				k1 = k_1 - n;
			}
			if (k1 < 0) {
				convo[kk] += conj(u_z[abs(k1)])*u_z[kk - k1]; 	
			} else if (kk - k1 < 0) {
				convo[kk] += u_z[k1]*conj(u_z[abs(kk - k1)]); 
			} else {
				convo[kk] += u_z[k1]*u_z[kk - k1];
			}			
		}
	}
}


void conv_3N2_pad(double complex* convo, double complex* uz, fftw_plan *fftw_plan_r2c_ptr, fftw_plan *fftw_plan_c2r_ptr, int n, int num_osc, int k0) {

	// Padded resolution
	int m = 3*n / 2;

	// Normalization factor
	double norm_fact = 1.0 / (double) m;

	// Allocate temporary arrays
	double* u_tmp;
	u_tmp = (double*)malloc(m*sizeof(double));
	double complex* uz_pad;
	uz_pad = (double complex*)malloc((3*num_osc/2)*sizeof(double complex));

	// write input data to padded array
	for (int i = 0; i < 3*n/4 + 1; ++i)	{
		if(i < num_osc){
			uz_pad[i] = uz[i];
		} else {
			uz_pad[i] = 0.0 + 0.0*I;
		}
	}

	// transform back to Real Space
	fftw_execute_dft_c2r((*fftw_plan_c2r_ptr), uz_pad, u_tmp);

	// square
	for (int i = 0; i < (2*m - 1); ++i)	{
		u_tmp[i] = pow(u_tmp[i], 2);
	}

	// here we move the derivative from real to spectral space so that the derivatives can be returned in spectral space
	fftw_execute_dft_r2c((*fftw_plan_r2c_ptr), u_tmp, uz_pad);

	// normalize
	for (int i = 0; i < num_osc; ++i)	{
		if (i <= k0) {
			convo[i] = 0.0 + 0.0*I;
		} else {
			convo[i] = uz_pad[i]*(norm_fact);
		}		
	}
}


void conv_2N_pad(fftw_complex* convo, fftw_complex* uz, fftw_plan *fftw_plan_r2c_ptr, fftw_plan *fftw_plan_c2r_ptr, int n, int num_osc, int k0) {

	// Padded resolution
	int m = 2*n;

	// Normalization factor
	double norm_fact = 1.0 / (double) m;

	// Allocate temporary arrays
	double* u_tmp;
	u_tmp = (double*)malloc(m*sizeof(double));
	fftw_complex* uz_pad;
	uz_pad = (fftw_complex*)malloc(2*num_osc*sizeof(fftw_complex));
	
	// write input data to padded array
	for (int i = 0; i < 2*num_osc; ++i)	{
		if(i < num_osc){
			uz_pad[i] = uz[i];
		} else {
			uz_pad[i] = 0.0 + 0.0*I;
		}
	}

	// // transform back to Real Space
	fftw_execute_dft_c2r((*fftw_plan_c2r_ptr), uz_pad, u_tmp);

	// // square
	for (int i = 0; i < m; ++i)	{
		u_tmp[i] = pow(u_tmp[i], 2);
	}

	// here we move the derivative from real to spectral space so that the derivatives can be returned in spectral space
	fftw_execute_dft_r2c((*fftw_plan_r2c_ptr), u_tmp, uz_pad);

	// // normalize
	for (int i = 0; i < num_osc; ++i)	{
		if (i <= k0) {
			convo[i] = 0.0 + 0.0*I;
		} else {
			convo[i] = uz_pad[i]*(norm_fact);
		}		
	}
}


void conv_23(double complex* convo, double complex* uz, fftw_plan *fftw_plan_r2c_ptr_23, fftw_plan *fftw_plan_c2r_ptr_23, int n, int kmax, int k0) {

	double norm_fact = 1/((double) n);

	double* u_tmp;
	u_tmp = (double* )malloc(n*sizeof(double));

	// transform back to Real Space
	fftw_execute_dft_c2r((*fftw_plan_c2r_ptr_23), uz, u_tmp);

	for (int i = 0; i < n; ++i)	{
		u_tmp[i] = pow(u_tmp[i], 2);
	}

	// here we move the derivative from real to spectral space so that the derivatives can be returned in spectral space
	fftw_execute_dft_r2c((*fftw_plan_r2c_ptr_23), u_tmp, convo);

	// normalize
	for (int i = 0; i < kmax; ++i)	{
		if(i <= k0)	{
			convo[i] = 0.0 + 0.0*I;
		} else {
			convo[i] *= norm_fact;	
		}
	}

	// apply filter mask
	for (int i = kmax; i < n; ++i)	{
		convo[i] = 0.0 + 0.0*I; 
	}
}