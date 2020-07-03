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



void convolution_direct(fftw_complex* convo, fftw_complex* u_z, int num_osc, int k0);
void jacobian(double* jac, fftw_complex* u_z, int n, int num_osc, int k0);
double trace(fftw_complex* u_z, int n, int num_osc, int k0);


int main( int argc, char** argv) {

	///-----------------
	///	Initialize vars
	///-----------------
	// Real space collocation points 
	int N       = 16;

	// Number of oscillators
	int num_osc = (N / 2) + 1;

	// Forcing wavenumber
	int k0 = 2;
	
	// Spectrum vars
	double a = 1.0;
	double b = 0.0;
	double cutoff = ((double) num_osc - 1.0) / 2.0;


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

	// Modes
	fftw_complex* u_z;
	u_z = (fftw_complex* ) fftw_malloc(num_osc*sizeof(fftw_complex));

	// Convolution
	fftw_complex* conv;
	conv = (fftw_complex* ) fftw_malloc(num_osc*sizeof(fftw_complex));

	// Jacobian
	double* jac;
	jac = (double* ) malloc((num_osc - 1)*(num_osc - 1)*sizeof(double));



	///-----------------
	///	Create Data
	///-----------------
	phi[0] = 0;
	amp[0] = 0;

	srand(123456789);

	for (int i = 0; i < num_osc; ++i)	{
		k[i] = i;
		
		// Set forcing here
		if(i <= k0) { /*|| i >= kmax) {*/
			u_z[i] = 0.0 + 0.0*I;
			amp[i] = 0.0;
			phi[i] = 0.0;
		} else if (i % 3 == 0) {
		// } else {
			amp[i] = pow((double)i, -a) * exp(-b * pow((double)k[i]/cutoff, 2) );
			phi[i] = M_PI/2.0;	
			// phi[i] = M_PI*( (double) rand() / (double) RAND_MAX);	
		} else if ( i % 3 == 1) {
			amp[i] = pow((double)i, -a) * exp(-b * pow((double)k[i]/cutoff, 2) );
			phi[i] = M_PI/6.0;	
		} else if ( i % 3 == 2) {
			amp[i] = pow((double)i, -a) * exp(-b * pow((double)k[i]/cutoff, 2) );
			phi[i] = -M_PI/6.0;	
		}

		u_z[i] = amp[i]*(cos(phi[i]) + I*sin(phi[i]));

		printf("k[%d]: %d | a: %5.15lf   p: %5.16lf   | u_z[%d]: %5.16lf  %5.16lfI \n", i, k[i], amp[i], phi[i], i, creal(u_z[i]), cimag(u_z[i]));
	}
	printf("\n\n");


	// Get convolution for diagonals
	convolution_direct(conv, u_z, num_osc, k0);
	for (int i = 0; i < num_osc; ++i) {
		printf("conv[%d]: %5.16lf %5.16lfI\n", i, creal(conv[i]), cimag(conv[i]));
	}
	printf("\n\n");


	///-----------------
	///	Construct Jacobian
	///-----------------
	jacobian(jac, u_z, N, num_osc, k0);

	int temp;
	int indx;
	for (int i = k0 + 1; i < num_osc; ++i)	{
		temp = (i - (k0 + 1)) * (num_osc - (k0 + 1));
		for (int j = k0 + 1; j < num_osc; ++j)	{
			indx = temp + (j - (k0 + 1));
			printf("%5.16lf ", jac[indx]);
		}
		printf("\n");
	}
	printf("\n\n");

	double tra;
	tra = trace(u_z, N, num_osc, k0);
	printf("Trace: %5.16lf\n", tra);




	///------------------
	/// Free memory
	///------------------
	// fftw_destroy_plan(fftw_plan_c2r_23);
	// fftw_destroy_plan(fftw_plan_r2c_23);
	// fftw_destroy_plan(fftw_plan_c2r_2N);
	// fftw_destroy_plan(fftw_plan_r2c_2N);
	// fftw_destroy_plan(fftw_plan_c2r_3N2);
	// fftw_destroy_plan(fftw_plan_r2c_3N2);

	free(amp);
	free(phi);
	free(k);
	fftw_free(u_z);
	free(jac);
	fftw_free(conv);

	
	return 0;
}

void jacobian(double* jac, fftw_complex* u_z, int n, int num_osc, int k0) {

	// Initialize temp vars
	int temp;
	int index;

	// initialize array for the convolution
	fftw_complex* conv;
	conv = (fftw_complex* ) fftw_malloc(num_osc*sizeof(fftw_complex));
	
	// Call convolution for diagonal elements
	convolution_direct(conv, u_z, num_osc, k0);

	// Loop through k and k'
	for (int kk = k0 + 1; kk < num_osc; ++kk) {
		temp = (kk - (k0 + 1)) * (num_osc - (k0 + 1));
		for (int kp = k0 + 1; kp < num_osc; ++kp) {
			index = temp + (kp - (k0 + 1));
			
			if(kk == kp) { // Diagonal elements
				if(kk + kp <= num_osc - 1) {
					jac[index] = ((double) kk) * cimag( (u_z[kp] * conj( u_z[kk + kp] )) / conj( u_z[kk] ) );
					jac[index] -= ((double) kk / 2.0) * cimag( conv[kp] / u_z[kk] );
				} else {
					jac[index] = 0.0;
					jac[index] -= ((double) kk / 2.0) * cimag( conv[kp] / u_z[kk] );
				} 
			} else { // Off diagonal elements
				if (kk + kp > num_osc - 1)	{
					if (kk - kp < -k0) {
						jac[index] = ((double) kk) * cimag( (u_z[kp] * conj( u_z[abs(kk - kp)] )) / u_z[kk] );
					} else if (kk - kp > k0) {
						jac[index] = ((double) kk) * cimag( (u_z[kp] * u_z[kk - kp]) / u_z[kk] );
					} else {
						jac[index] = 0.0;
					}					
				} else {
					if (kk - kp < -k0) {
						jac[index] = ((double) kk) * cimag( (u_z[kp] * conj( u_z[abs(kk - kp)] )) / u_z[kk] ) + ((double) kk) * cimag( (u_z[kp] * conj( u_z[kk + kp] )) / conj( u_z[kk] ) );
					} else if (kk - kp > k0) {
						jac[index] = ((double) kk) * cimag( (u_z[kp] * u_z[kk - kp]) / u_z[kk] ) + ((double) kk) * cimag( (u_z[kp] * conj( u_z[kk + kp] )) / conj( u_z[kk] ) );
					} else {
						jac[index] = ((double) kk) * cimag( (u_z[kp] * conj( u_z[kk + kp] )) / conj( u_z[kk] ) );
					}
				}
			}
		}
	}
}

double trace(fftw_complex* u_z, int n, int num_osc, int k0) {

	// init temp var
	int indx;

	double tra;

	// initialize array for the convolution
	fftw_complex* conv;
	conv = (fftw_complex* ) fftw_malloc(num_osc*sizeof(fftw_complex));
	
	// Call convolution for diagonal elements
	convolution_direct(conv, u_z, num_osc, k0);

	for (int i = k0 + 1; i < num_osc; ++i)	{
		if(2*i <= num_osc - 1) {
			tra +=  ((double) i) * cimag( (u_z[i] * conj( u_z[2*i] )) / conj( u_z[i] ) );
			tra -= ((double) i / 2.0) * cimag( conv[i] / u_z[i] );
		} else {
			tra += 0.0;
			tra -= ((double) i / 2.0) * cimag( conv[i] / u_z[i] );
		} 
	}
	return tra;
}




void convolution_direct(fftw_complex* convo, fftw_complex* u_z, int num_osc, int k0) {
	
	// Set the 0 to k0 modes to 0;
	for (int i = 0; i <= k0; ++i) {
		convo[0] = 0.0 + 0.0*I;
	}
	
	// Compute the convolution on the remaining wavenumbers
	int k1;
	for (int kk = k0 + 1; kk < num_osc; ++kk)	{
		for (int k_1 = 1 + kk; k_1 < 2*num_osc; ++k_1)	{
			// Get correct k1 value
			if(k_1 < num_osc) {
				k1 = -num_osc + k_1;
			} else {
				k1 = k_1 - num_osc;
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