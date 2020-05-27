// Enda Carroll
// May 2020
// Programme to test the convolution term

// ---------------------------------------------------------------------
//  Standard Libraries and Headers
// ---------------------------------------------------------------------
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <complex.h>


// ---------------------------------------------------------------------
//  User Libraries and Headers
// ---------------------------------------------------------------------
#include <fftw3.h> // include the latest FFTW library
#include <omp.h>
#include <gsl/gsl_cblas.h>
// #include <f2c.h>
// #include <cblas.h>
#include <lapacke.h>
// #include <clapack.h>
// #include <gsl/gsl_linalg.h>



void conv_2N_pad(fftw_complex* convo, fftw_complex* uz, fftw_plan *fftw_plan_r2c_ptr, fftw_plan *fftw_plan_c2r_ptr, int n, int num_osc, int k0);
void po_rhs(double* rhs, fftw_complex* u_z, fftw_plan *plan_c2r_pad, fftw_plan *plan_r2c_pad, int* kx, int n, int num_osc, int k0);
void po_rhs_extended(double* rhs, double* rhs_ext, fftw_complex* u_z, double* pert, fftw_plan *plan_c2r_pad, fftw_plan *plan_r2c_pad, int* kx, int n, int num_osc, int k0);
void jacobian(double* jac, fftw_complex* u_z, int n, int num_osc, int k0);
void convolution_direct(fftw_complex* convo, fftw_complex* u_z, int n, int k0);
void orthonormalize(double* rhs_pert, double* pert, double* znorm, int num_osc, int kmin);


int main( int argc, char** argv) {

	///-----------------
	///	Initialize vars
	///-----------------
	// Real space collocation points 
	int N = 8;

	// padded array size
	int M = 2 * N;

	// Number of oscillators
	int num_osc = (N / 2) + 1;

	// Forcing wavenumber
	int k0 = 0;
	int kmin = k0 + 1;
	
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

	double* rhs;
	rhs = (double* ) fftw_malloc(num_osc*sizeof(double));

	// padded arrays for dft
	double* u_pad;
	u_pad = (double*) malloc(M*sizeof(double));
	fftw_complex* u_z_pad;
	u_z_pad = (fftw_complex* ) fftw_malloc(2*num_osc*sizeof(fftw_complex));


	///-----------------
	///	Init OpenMP Threads
	///-----------------
	// int n_threads = atoi(argv[1]);
	// omp_set_num_threads(n_threads);
	// printf("\n\tNumber of OpenMP Threads running = %d\n\n" , omp_get_max_threads());
	// fftw_init_threads();
	// fftw_plan_with_nthreads((int)omp_get_max_threads());
	
	

	///-----------------
	///	DFT Plans
	///-----------------
	fftw_plan fftw_plan_c2r_pad;
	fftw_plan fftw_plan_r2c_pad;

	fftw_plan_c2r_pad = fftw_plan_dft_c2r_1d(M, u_z_pad, u_pad, FFTW_PRESERVE_INPUT);
	fftw_plan_r2c_pad = fftw_plan_dft_r2c_1d(M, u_pad, u_z_pad, FFTW_PRESERVE_INPUT);


	///-----------------
	///	Create Data
	///-----------------
	phi[0] = 0;
	amp[0] = 0;

	srand(123456789);

	for (int i = 0; i < num_osc; ++i)	{
		k[i] = i;
		if(i > 0) {
			amp[i] = pow((double)i, -a) * exp(-b * pow((double)k[i]/cutoff, 2) );
			phi[i] = M_PI/2.0;	
			// phi[i] = M_PI*( (double) rand() / (double) RAND_MAX);	
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



	///-----------------
	///	Evaluate RHS
	///-----------------
	// clock_t begin = clock();

	// po_rhs(rhs, u_z, &fftw_plan_c2r_pad, &fftw_plan_r2c_pad, k, N, num_osc, k0);
	
	// clock_t end = clock();

	// // calculate execution time
	// double time_spent = (double)(end - begin) / CLOCKS_PER_SEC;

	// printf("Time: %5.8lf\n", time_spent);
	// printf("\n\n");

	// printf("RHS:\n");
	// for (int i = 0; i < num_osc; ++i) {
	// 	printf("rhs[%d]: %5.15lf \n", i, rhs[i]);
	// }
	// printf("\n\n");
	


	// ///-----------------
	// /// Test calling of cblas_dgemm
	// ///-----------------
	// // Allocate memory and fill with known values
	// double* A, *B, *C;
	// A = (double* )malloc((num_osc - 1)*(num_osc - 1)*sizeof(double));
	// B = (double* )malloc((num_osc - 1)*(num_osc - 1)*sizeof(double));
	// C = (double* )malloc((num_osc - 1)*(num_osc - 1)*sizeof(double));

	// int tmp;
	// int indx;
	// printf("A: before\n");
	// for (int i = 0; i < num_osc - 1; ++i) {
	// 	tmp = i * (num_osc - 1);
	// 	for (int j = 0; j < num_osc -1; ++j) {
	// 		indx = tmp + j;
	// 		// if (i == j) {
	// 		// 	B[indx] = 1.0;	
	// 		// } else {
	// 		// 	B[indx] = 0.0;	
	// 		// }
	// 		B[indx] = 8.5*indx;
	// 		A[indx] = indx;
	// 		C[indx] = 0.0;
	// 		printf("%5.5lf \t", A[indx]);
	// 	}
	// 	printf("\n");
	// }
	// printf("B: before\n");
	// for (int i = 0; i < num_osc - 1; ++i) {
	// 	tmp = i * (num_osc - 1);
	// 	for (int j = 0; j < num_osc -1; ++j) {
	// 		indx = tmp + j;
	// 		printf("%5.5lf \t", B[indx]);
	// 	}
	// 	printf("\n");
	// }
	// printf("C: before\n");
	// for (int i = 0; i < num_osc - 1; ++i) {
	// 	tmp = i * (num_osc - 1);
	// 	for (int j = 0; j < num_osc -1; ++j) {
	// 		indx = tmp + j;
	// 		printf("%5.5lf \t", C[indx]);
	// 	}
	// 	printf("\n");
	// }
	// printf("\n\n");

	// // dgemm variables
	// double alpha = 1.0;     // prefactor of A*B
	// double beta  = 0.0;     // prefactor of C
	// int MM = num_osc - 1;    // no. of rows of A
	// int NN = num_osc - 1;    // no. of cols of B
	// int K = num_osc - 1;    // no. of cols of A / rows of B
	// int lda = num_osc - 1;  // leading dim of A - length of elements between consecutive rows
	// int ldb = num_osc - 1;  // leading dim of B
	// int ldc = num_osc - 1;  // leading dim of C

	// cblas_dgemm (CblasRowMajor, CblasNoTrans, CblasNoTrans, MM, NN, K, 1.0, A, lda, B, ldb, 0.0, C, ldc);

	// printf("C After:\n");
	// for (int i = 0; i < num_osc - 1; ++i) {
	// 	tmp = i * (num_osc - 1);
	// 	for (int j = 0; j < num_osc -1; ++j) {
	// 		indx = tmp + j;			
	// 		printf("%5.16lf \t", C[indx]);
	// 	}
	// 	printf("\n");
	// }
	// printf("\n\n");




	///-----------------
	///	Extended RHS
	///-----------------
	// Allocate memory for pertubed system
	double* pert;
	pert     = (double* ) malloc((num_osc - (k0 + 1))*(num_osc - (k0 + 1))*sizeof(double));
	double* rhs_pert;
	rhs_pert = (double* ) malloc((num_osc - (k0 + 1))*(num_osc - (k0 + 1))*sizeof(double));
	

	// fill pert arrays
	int tmp;
	int indx;
	printf("Before:\n");
	for (int i = 0; i < num_osc - (k0 + 1); ++i) {
		tmp = i * (num_osc - (k0 + 1));
		for (int j = 0; j < num_osc - (k0 + 1); ++j) {
			indx = tmp + j;
			if (i == j) {
				pert[indx] = 1.0;	
			} else {
				pert[indx] = 0.0;	
			}
			rhs_pert[indx] = 0.0;
			printf("%5.5lf \t", pert[indx]);
		}
		printf("\n");
	}
	printf("\n\n");


	po_rhs_extended(rhs, rhs_pert, u_z, pert, &fftw_plan_c2r_pad, &fftw_plan_r2c_pad, k, N, num_osc, k0);
	

	printf("After:\n");
	for (int i = 0; i < num_osc - (k0 + 1); ++i) {
		tmp = i * (num_osc - (k0 + 1));
		for (int j = 0; j < num_osc - (k0 + 1); ++j) {
			indx = tmp + j;
			
			printf("%5.16lf \t", rhs_pert[indx]);
		}
		printf("\n");
	}
	printf("\n\n");



	///-----------------
	///	Orthonormalize
	///-----------------
	int kdim = num_osc - kmin;
	double* r;
	r = (double* )malloc(sizeof(double) * (kdim));
	double* Q;
	Q = (double* )malloc(sizeof(double) * (kdim) * (kdim));

	// // norm of first col
	// r[0] = 0.0;
	// for (int i = 0; i < (kdim); ++i) {
	// 	r[0] += pow(rhs_pert[i * (kdim)], 2);
	// }
	// r[0] = sqrt(r[0]);
	// // write first col of Q;
	// for (int i = 0; i < kdim; ++i) {
	// 	Q[i * (kdim)] = rhs_pert[i * (kdim)] / r[0];
	// }


	// for (int kk = 1; kk < (kdim); ++kk) {
	// 	for (int j = 0; j < kk - 1; ++j) {
	// 		// get the dot prod of col q_j with col k of input
	// 		int rr = 0.0;
	// 		for (int i = 0; i < (kdim); ++i) {
	// 			rr += rhs_pert[i * (kdim) + kk] * Q[i * (kdim) + j];
	// 		}

	// 		// update
	// 		for (int i = 0; i < (kdim); ++i) {
	// 			rhs_pert[i * (kdim) + kk] = rhs_pert[i * (kdim) + kk]  - rr * Q[i * (kdim) + j];
	// 		}
	// 	}

	// 	// norm of 
	// 	r[kk] = 0.0;
	// 	for (int i = 0; i < (kdim); ++i) {
	// 		r[kk] += pow(rhs_pert[i * (kdim) + kk], 2);
	// 	}
	// 	r[kk] = sqrt(r[kk]);

	// 	// write first col of Q;
	// 	for (int i = 0; i < kdim; ++i) {
	// 		Q[i * (kdim) + kk] = rhs_pert[i * (kdim) + kk] / r[kk];
	// 	}
	// }

	/////////////////////////////////////////
	// int dim  = kdim;
	// double* r_tmp;
	// r_tmp = (double* )malloc(sizeof(double) * dim * dim);

	// for (int i = 0; i < dim; ++i) {
	// 	for (int j = 0; j < dim; ++j) {
	// 		for (int kk = 0; kk < dim; ++kk) {
	// 			r_tmp[i * dim + j] = r_tmp[i * dim + j] + rhs_pert[kk * dim + i]*rhs_pert[kk * dim + j]; 
	// 		}
	// 	}
	// 	r[i] = sqrt(r_tmp[i * dim + i]);

	// 	for (int kk = 0; kk < dim; ++kk) {
	// 		rhs_pert[kk * dim + i] = rhs_pert[kk * dim + i] / r[i]; 
	// 	}
	// 	for (int j = i + 1; j < dim; ++j) {
	// 		r_tmp[i * dim + j] /= r[i];
	// 	}
	// 	for (int j = i + 1; j < dim; ++j) {
	// 		for (int kk = 0; kk < dim; ++kk) {
	// 			rhs_pert[kk * dim + j] = rhs_pert[kk * dim + j] - rhs_pert[kk * dim + i] * r_tmp[i * dim + j]; 
	// 		}
	// 	}
	// }



	// printf("Norm:\n");
	// for (int i = 0; i < kdim; ++i) {
	// 	printf("r[%d]: %20.15lf\n", i, r[i]);
	// }
	// printf("\n\n");

	// for (int i = 0; i < kdim; ++i) {
	// 	tmp = i * (kdim);
	// 	for (int j = 0; j < kdim; ++j) {
	// 		indx = tmp + j;
	// 		printf("Q[%d]: %20.16lf\t", indx, rhs_pert[indx]);
	// 	}
	// 	printf("\n");
	// }
	// printf("\n\n");

	double* znorm;
	znorm = (double* )malloc(sizeof(double) * (num_osc - kmin));

	orthonormalize(rhs_pert, pert, znorm, num_osc, kmin);

	///------------------
	/// Free memory
	///------------------
	fftw_destroy_plan(fftw_plan_c2r_pad);
	fftw_destroy_plan(fftw_plan_r2c_pad);

	free(amp);
	free(phi);
	free(k);
	free(rhs);
	free(u_pad);
	free(pert);
	free(rhs_pert);
	fftw_free(u_z);
	fftw_free(u_z_pad);

	// Clean up threads
	fftw_cleanup_threads();

	return 0;
}

void po_rhs(double* rhs, fftw_complex* u_z, fftw_plan *plan_c2r_pad, fftw_plan *plan_r2c_pad, int* kx, int n, int num_osc, int k0) {

	// Initialize variables
	int m = 2 * n;
	double norm_fac = 1.0 / (double) m;

	fftw_complex pre_fac;

	// Allocate temporary arrays
	double* u_tmp;
	u_tmp = (double* ) malloc(m*sizeof(double));

	fftw_complex* u_z_tmp;
	u_z_tmp = (fftw_complex* ) fftw_malloc(2*num_osc*sizeof(fftw_complex));

	///---------------
	/// Convolution
	///---------------
	// Write data to padded array
	for (int i = 0; i < 2*num_osc; ++i)	{
		if(i < num_osc){
			u_z_tmp[i] = u_z[i];
		} else {
			u_z_tmp[i] = 0.0 + 0.0*I;
		}
	}

	// transform back to Real Space
	fftw_execute_dft_c2r((*plan_c2r_pad), u_z_tmp, u_tmp);

	// multiplication in real space
	for (int i = 0; i < m; ++i)	{
		u_tmp[i] = pow(u_tmp[i], 2);
	}

	// transform forward to Fourier space
	fftw_execute_dft_r2c((*plan_r2c_pad), u_tmp, u_z_tmp);

	///---------------
	/// RHS
	///---------------
	for (int k = 0; k < num_osc; ++k) {
		if (k <= k0) {
			rhs[k] = 0.0;
		} else {
			pre_fac = (-I * kx[k]) / (2.0 * u_z[k]);
			rhs[k]  = cimag( pre_fac* (u_z_tmp[k] * norm_fac) );
		}		
	}

	free(u_tmp);
	fftw_free(u_z_tmp);
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


void jacobian(double* jac, fftw_complex* u_z, int n, int num_osc, int k0) {

	// Initialize temp vars
	int temp;
	int index;

	// initialize array for the convolution
	fftw_complex* conv;
	conv = (fftw_complex* ) fftw_malloc(num_osc*sizeof(fftw_complex));
	
	// Call convolution for diagonal elements
	convolution_direct(conv, u_z, n, k0);

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
						jac[index] = ((double) kk) * cimag( (u_z[kp] * conj( u_z[abs(kk - kp)] )) / u_z[kk] ) + ((double) kk) * cimag( (u_z[kp] * conj( u_z[kk + kp] )) / conj( u_z[kk] ) );;
					} else if (kk - kp > k0) {
						jac[index] = ((double) kk) * cimag( (u_z[kp] * u_z[kk - kp]) / u_z[kk] ) + ((double) kk) * cimag( (u_z[kp] * conj( u_z[kk + kp] )) / conj( u_z[kk] ) );;
					} else {
						jac[index] = ((double) kk) * cimag( (u_z[kp] * conj( u_z[kk + kp] )) / conj( u_z[kk] ) );;
					}
				}
			}
		}
	}

	fftw_free(conv);
}

void orthonormalize(double* rhs_pert, double* pert, double* znorm, int num_osc, int kmin) {

	// Initialize vars
	int tmp;
	int indx;
	int kdim = num_osc - kmin;

	// Initialize lapack vars
	lapack_int info;
	lapack_int m   = kdim;
	lapack_int n   = kdim;
	lapack_int lda = kdim;

	// Initialize the blas variables for dgemm
	double alpha = 1.0;     // prefactor of A*B
	double beta  = 0.0;     // prefactor of C
	int K = kdim;    // no. of cols of A / rows of B
	int ldb = kdim;  // leading dim of B
	int ldc = kdim;  // leading dim of C
	
	// Allocate temporary memory
	double* tau;
	tau = (double* )malloc(sizeof(double) * kdim);	
	double* col_change;
	col_change =  (double* )malloc(sizeof(double) * (kdim) * (kdim));

	// Initialize col_change matrix
	for (int i = 0; i < kdim; ++i) {
		for (int j = 0; j < kdim; ++j) {
			col_change[i * (kdim) + j] = 0.0; 
		}
	}

	///---------------
	/// Perform QR Fac
	///---------------
	info = LAPACKE_dgeqrf(LAPACK_ROW_MAJOR, m, n, rhs_pert, lda, tau);


	// extract the diagonals of R
	for (int i = 0; i < kdim; ++i) {
		if (rhs_pert[i * (kdim) + i] < 0) {
			col_change[i * (kdim) + i] = -1.0;
		} else {
			col_change[i * (kdim) + i] = 1.0;
		}
		znorm[i] = rhs_pert[i * (kdim) + i] * col_change[i * (kdim) + i];
		printf("r[%d]: %20.15lf\n", i, znorm[i]);
		
	}
	printf("\n\n");

	for (int i = 0; i < kdim; ++i) {
		tmp = i * (kdim);
		for (int j = 0; j < kdim; ++j) {
			indx = tmp + j;
			printf("col[%d]: %20.16lf\t", indx, col_change[indx]);
		}
		printf("\n");
	}
	printf("\n\n");


	///---------------
	/// Form the Q matrix
	///---------------
    info = LAPACKE_dorgqr(LAPACK_ROW_MAJOR, m, m, n, rhs_pert, lda, tau);

   
   	// Correct the orientation of the Q matrix columns 
	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, K, 1.0, rhs_pert, lda, col_change, ldb, 0.0, pert, ldc);


    for (int i = 0; i < kdim; ++i) {
		tmp = i * (kdim);
		for (int j = 0; j < kdim; ++j) {
			indx = tmp + j;
			printf("Q[%d]: %20.16lf\t", indx, pert[indx]);
		}
		printf("\n");
	}

	free(tau);
	free(col_change);

}


void po_rhs_extended(double* rhs, double* rhs_ext, fftw_complex* u_z, double* pert, fftw_plan *plan_c2r_pad, fftw_plan *plan_r2c_pad, int* kx, int n, int num_osc, int k0) {


	///---------------
	/// RHS
	///---------------
	// Initialize variables
	int m = 2 * n;
	double norm_fac = 1.0 / (double) m;

	fftw_complex pre_fac;

	// Allocate temporary arrays
	double* u_tmp;
	u_tmp = (double* ) malloc(m*sizeof(double));

	double* jac_tmp;
	jac_tmp = (double* ) malloc((num_osc - (k0 + 1))*(num_osc - (k0 + 1))*sizeof(double));

	fftw_complex* u_z_tmp;
	u_z_tmp = (fftw_complex* ) fftw_malloc(2*num_osc*sizeof(fftw_complex));

	fftw_complex* conv;
	conv = (fftw_complex* ) fftw_malloc(num_osc*sizeof(fftw_complex));

	// Write data to padded array
	for (int i = 0; i < 2*num_osc; ++i)	{
		if(i < num_osc){
			u_z_tmp[i] = u_z[i];
		} else {
			u_z_tmp[i] = 0.0 + 0.0*I;
		}
	}

	// transform back to Real Space
	fftw_execute_dft_c2r((*plan_c2r_pad), u_z_tmp, u_tmp);

	// multiplication in real space
	for (int i = 0; i < m; ++i)	{
		u_tmp[i] = pow(u_tmp[i], 2);
	}

	// transform forward to Fourier space
	fftw_execute_dft_r2c((*plan_r2c_pad), u_tmp, u_z_tmp);
	
	for (int k = 0; k < num_osc; ++k) {
		if (k <= k0) {
			rhs[k]  = 0.0;
			conv[k] = 0.0 + 0.0*I; 
		} else {
			pre_fac = (-I * kx[k]) / (2.0 * u_z[k]);
			conv[k] = u_z_tmp[k] * norm_fac;
			rhs[k]  = cimag(pre_fac * (conv[k]));
		}		
	}


	///---------------
	/// Extended RHS
	///---------------
	// calculate the jacobian
	int temp;
	int index;
	for (int kk = k0 + 1; kk < num_osc; ++kk) {
		temp = (kk - (k0 + 1)) * (num_osc - (k0 + 1));
		for (int kp = k0 + 1; kp < num_osc; ++kp) {
			index = temp + (kp - (k0 + 1));			
			if(kk == kp) { // Diagonal elements
				if(kk + kp <= num_osc - 1) {
					jac_tmp[index] = ((double) kk) * cimag( (u_z[kp] * conj( u_z[kk + kp] )) / conj( u_z[kk] ) );
					jac_tmp[index] -= ((double) kk / 2.0) * cimag( conv[kp] / u_z[kk] );
				} else {
					jac_tmp[index] = 0.0;
					jac_tmp[index] -= ((double) kk / 2.0) * cimag( conv[kp] / u_z[kk] );
				} 
			} else { // Off diagonal elements
				if (kk + kp > num_osc - 1)	{
					if (kk - kp < -k0) {
						jac_tmp[index] = ((double) kk) * cimag( (u_z[kp] * conj( u_z[abs(kk - kp)] )) / u_z[kk] );
					} else if (kk - kp > k0) {
						jac_tmp[index] = ((double) kk) * cimag( (u_z[kp] * u_z[kk - kp]) / u_z[kk] );
					} else {
						jac_tmp[index] = 0.0;
					}					
				} else {
					if (kk - kp < -k0) {
						jac_tmp[index] = ((double) kk) * cimag( (u_z[kp] * conj( u_z[abs(kk - kp)] )) / u_z[kk] ) + ((double) kk) * cimag( (u_z[kp] * conj( u_z[kk + kp] )) / conj( u_z[kk] ) );;
					} else if (kk - kp > k0) {
						jac_tmp[index] = ((double) kk) * cimag( (u_z[kp] * u_z[kk - kp]) / u_z[kk] ) + ((double) kk) * cimag( (u_z[kp] * conj( u_z[kk + kp] )) / conj( u_z[kk] ) );;
					} else {
						jac_tmp[index] = ((double) kk) * cimag( (u_z[kp] * conj( u_z[kk + kp] )) / conj( u_z[kk] ) );;
					}
				}
			}
		}
	}


	// Call matrix matrix multiplication - C = alpha*A*B + beta*C => rhs_ext = alpha*jac_tmp*pert + 0.0*rhs_ext
	// variables setup
	double alpha = 1.0;     // prefactor of A*B
	double beta  = 0.0;     // prefactor of C
	int M = num_osc - (k0 + 1);    // no. of rows of A
	int N = num_osc - (k0 + 1);    // no. of cols of B
	int K = num_osc - (k0 + 1);    // no. of cols of A / rows of B
	int lda = num_osc - (k0 + 1);  // leading dim of A - length of elements between consecutive rows
	int ldb = num_osc - (k0 + 1);  // leading dim of B
	int ldc = num_osc - (k0 + 1);  // leading dim of C

	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, 1.0, jac_tmp, lda, pert, ldb, 0.0, rhs_ext, ldc);


	// Free tmp arrays
	free(jac_tmp);
	free(u_tmp);
	fftw_free(conv);
	fftw_free(u_z_tmp);
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

	// transform back to Real Space
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

	free(u_tmp);
	fftw_free(uz_pad);
}