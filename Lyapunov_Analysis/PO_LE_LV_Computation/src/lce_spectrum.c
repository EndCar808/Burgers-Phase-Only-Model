// Enda Carroll
// May 2020
// File including functions to perform the Benettin et al., algorithm
// for computing the Lyapunov spectrum of the Phase Only Burgers Equation


// ---------------------------------------------------------------------
//  Standard Libraries and Headers
// ---------------------------------------------------------------------
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <float.h>
#include <string.h>
#include <complex.h>
#include <fftw3.h>
#include <hdf5.h>
#include <hdf5_hl.h>
#include <omp.h>
#include <gsl/gsl_cblas.h>
#include <lapacke.h>


// ---------------------------------------------------------------------
//  User Libraries and Headers
// ---------------------------------------------------------------------
#include "data_types.h"
#include "utils.h"




// ---------------------------------------------------------------------
//  Function Definitions
// ---------------------------------------------------------------------
/**
 * Function that performs the RHS of the extended system - both the system of phases and tangent space system.
 * For the phases: the convolution is performed by writing the input modes padded array (2N pading), transforming 
 * 				   to realspace, squaring in realspace, transforming back to Fourier (with a normalization factor), before then 
 *        		   computing the RHS
 * For the tangent space sys: Form the jacobian from the input modes, evolve forward the tangent space matrix, 
 * 							  pert, by left multiply by the Jacobian 
 * 							  
 * @param rhs          Output array of size num_osc to store the result of the RHS of the phases
 * @param rhs_ext      Output array of size numLEs x numLes to store the result of the RHS of the tangent space system
 * @param u_z          Input array of size num_osc holding the modes
 * @param pert         Input array of size numLEs x numLEs holding the tangent space vectors
 * @param plan_c2r_pad FFTW plan for forward transform for the padded array
 * @param plan_r2c_pad FFTW plan for the backward transform for the padded arry
 * @param kx           Inuput array of length num_osc holding the wavenumbers
 * @param n            Number of collocation points
 * @param num_osc      Number of oscillators (including the phi_0)
 * @param k0           The number of modes killed
 * @param numLEs       The number of LEs we are computing = dimensions of the tangent space matrix
 */
void po_rhs_extended(double* rhs, double* rhs_ext, fftw_complex* u_z, double* pert, fftw_plan *plan_c2r_pad, fftw_plan *plan_r2c_pad, int* kx, int n, int num_osc, int k0, int numLEs) {


	///---------------
	/// RHS
	///---------------
	// Initialize variables
	int kmin        = k0 + 1;
	int dof         = num_osc - kmin;
	int m           = 2 * n;             // size of the padded array
	double norm_fac = 1.0 / (double) m;  // Normalization factor

	// variable to hold the prefactor for the RHS
	fftw_complex pre_fac;

	// Allocate temporary arrays
	double* u_tmp   = (double* ) malloc(m*sizeof(double));
	mem_chk(u_tmp, "u_tmp");
	double* jac_tmp = (double* ) malloc(dof * dof *sizeof(double));
	mem_chk(jac_tmp, "jac_tmp");

	fftw_complex* conv    = (fftw_complex* ) fftw_malloc(num_osc * sizeof(fftw_complex));
	mem_chk(conv, "conv");
	fftw_complex* u_z_tmp = (fftw_complex* ) fftw_malloc((2 * num_osc - 1) * sizeof(fftw_complex));
	mem_chk(u_z_tmp, "u_z_tmp");


	// Write data to padded array
	for (int i = 0; i < (2 * num_osc - 1); ++i)	{
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
	
	// Update RHS and conv array (used in the Jacobian)
	for (int k = 0; k < num_osc; ++k) {
		if (k <= k0) {
			rhs[k]  = 0.0;
			conv[k] = 0.0 + 0.0*I; 
		} else {
			conv[k] = u_z_tmp[k] * norm_fac;
			// pre_fac = (-I * kx[k]) / (2.0 * u_z[k]);
			// rhs[k]  = cimag(pre_fac * (conv[k]));
			pre_fac = - kx[k] / cabs(u_z[k]);
			rhs[k]  = pre_fac * creal(cexp(- I * carg(u_z[k])) * (u_z_tmp[k] * norm_fac));
		}		
	}


	///---------------
	/// Extended RHS
	///---------------
	// calculate the jacobian
	int temp;
	int index;
	for (int kk = kmin; kk < num_osc; ++kk) {
		temp = (kk - (kmin)) * (num_osc - (kmin));
		for (int kp = kmin; kp < num_osc; ++kp) {
			index = temp + (kp - (kmin));			
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
	

	// Perform Matrix multiplication 
	// variables setup
	double alpha = 1.0;  // prefactor of A*B
	double beta  = 0.0;  // prefactor of C
	int M      = dof;    // no. of rows of A
	int N      = numLEs; // no. of cols of B
	int K      = dof;    // no. of cols of A / rows of B
	int lda    = dof;    // leading dim of A - length of elements between consecutive rows
	int ldb    = numLEs; // leading dim of B
	int ldc    = numLEs; // leading dim of C
	int incr_x = 1;      // increment in x array
	int incr_y = 1;      // increment in y array

	if (numLEs == 1) {
		// Call matrix vector multiplication - y = alpha*A*x + beta*y => rhs_ext = alpha*jac_tmp*pert + 0.0*C
		cblas_dgemv(CblasRowMajor, CblasNoTrans, M, M, alpha, jac_tmp, lda, pert, 1, beta, rhs_ext, 1);
	}
	else {
		// Call matrix matrix multiplication - C = alpha*A*B + beta*C => rhs_ext = alpha*jac_tmp*pert + 0.0*C
		cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, alpha, jac_tmp, lda, pert, ldb, beta, rhs_ext, ldc);
	}


	// Free tmp arrays
	free(jac_tmp);
	free(u_tmp);
	fftw_free(conv);
	fftw_free(u_z_tmp);
}

/**
 * Function to compute the amplitude normalization factor (convolution of amplitudes) for computing the scale-dependent order parameters
 *
 * @param norm    Output array of size num_osc to hold the value amplitude convolution for each wavenumber k
 * @param amp     Input array of size num_osc that holds the value of the individual amplitudes
 * @param num_osc Variable defining the size of the input and output arrays
 * @param k0      Variable defining the number of modes set to 0
 */

void amp_normalize(double* norm, double* amp, int num_osc, int k0) {

	// Initialize variables
	int k1;
	int N = num_osc - 1;

	// Compute the sum for each k
	for (int kk = 0; kk <= N; ++kk) {
		if (kk <= k0) {
			norm[kk] = 0.0;
		}
		else {
			for (int k_1 = 0; k_1 <= 2 * N; ++k_1) {
				// Adjust for the correct k1 value
				if (k_1 <= N) {     
					k1 = -N + k_1;
				}
				else  {
					k1 = k_1 - N;
				}
				
				// Compute the convolution
				if (k1 >= - N + kk) {
					norm[kk] +=  amp[abs(k1)] * amp[abs(kk - k1)];
				}
			}
		}
		
	}
}


/**
 * Function to compute the Jacobian from the modes - NOT USED
 * 
 * @param jac     Output array of size (num_osc - k_0 - 1) * (num_osc - k_0 - 1) to hold the Jacobian
 * @param u_z     Input array of size num_osc holding the modes
 * @param num_osc Number of oscillators (includes phi_0)
 * @param k0      Number of modes killed
 */
void jacobian(double* jac, fftw_complex* u_z, int num_osc, int k0) {

	// Initialize temp vars
	int temp;
	int index;

	// initialize array for the convolution
	fftw_complex* conv = (fftw_complex* ) fftw_malloc(num_osc*sizeof(fftw_complex));
	mem_chk(conv, "conv");
	
	// Call convolution for diagonal elements
	conv_direct(conv, u_z, num_osc, k0);

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

	///---------------
	/// Free temp memory
	///---------------
	fftw_free(conv);
}

/**
 * Function to compute the trace of the Jacobian by just computing the diagonal elements  
 * 
 * @param  u_z     Input array of size num_osc holding the modes
 * @param  num_osc Number of the oscillators (includes phi_0)
 * @param  k0      Number of killed modes
 * @return         Returns the trace in tra
 */
double trace(fftw_complex* u_z, int num_osc, int k0) {

	double tra;

	// initialize array for the convolution
	fftw_complex* conv = (fftw_complex* ) fftw_malloc(num_osc*sizeof(fftw_complex));
	mem_chk(conv, "conv");
	
	// Call convolution for diagonal elements
	conv_direct(conv, u_z, num_osc, k0);

	tra = 0;
	for (int i = k0 + 1; i < num_osc; ++i)	{
		if(2*i <= num_osc - 1) {
			tra +=  ((double) i) * cimag( (u_z[i] * conj( u_z[2*i] )) / conj( u_z[i] ) );
			tra -= ((double) i / 2.0) * cimag( conv[i] / u_z[i] );
		} else {
			tra += 0.0;
			tra -= ((double) i / 2.0) * cimag( conv[i] / u_z[i] );
		} 
	}

	///---------------
	/// Free temp memory
	///---------------
	fftw_free(conv);

	return tra;
}

/**
 * Function to perform an othogonalization of column vectors of a matrix using the modified Grahm-Schmidt algorithm 
 * 
 * @param q       Input array of size (num_osc - kmin) * (num_osc - kmin) holding the column vectors to be orthogonalized
 * @param r       Output array of size (num_osc - kmin) to hold the diagonal elements of the R matrix
 * @param num_osc The number of osciallotrs (includes phi_0)
 * @param kmin    Wavenumber of the first nonzero mode
 */
void modified_gs(double* q, double* r, int num_osc, int kmin) {

	int dim       = num_osc - kmin;
	double* r_tmp = (double* )malloc(sizeof(double) * dim * dim);
	double* a_tmp = (double* )malloc(sizeof(double) * dim * dim);

	// Write input to temporary matrix
	for (int i = 0; i < dim; ++i){
		for (int j = 0; j < dim; ++j) {
			a_tmp[i * dim + j ] = q[i * dim + j ] ;
		}
	}
	
	double norm;
	for (int i = 0; i < dim; ++i) {
		norm = 0.0;
		for (int k = 0; k < dim; ++k) {
				norm += a_tmp[k * dim + i]*a_tmp[k * dim + i];
		}
		r_tmp[i * dim + i] = sqrt(norm);

		// Write diagonal to ouotput
		r[i] = r_tmp[i * dim + i];

		for (int k = 0; k < dim; ++k) {
		 	q[k * dim + i]  =  a_tmp[k * dim + i] / r_tmp[i * dim + i]; 
		} 

		for (int j = i + 1; j < dim; ++j) {
			for (int k = 0; k < dim; ++k) {
				r_tmp[i * dim + j] += q[k * dim + i] * a_tmp[k * dim + j];
			}
			for (int k = 0; k < dim; ++k) {
				a_tmp[k * dim + j] -= r_tmp[i * dim + j] * q[k * dim + i]; 
			}
		}
	} 
}


/**
 * Wrapper function used to perform an othogonalization of the tangent space vectors by a QR decomposition using the LAPACKE function LAPACKE_dgeqrf();
 * or if only one (max) LE is being computed simply normalize it using CBLAS function cblas_dnrm2
 * 
 * @param pert    Array of size (num_osc - kmin) * (numLES) holding the input tangent space matrix - this will be overwitten 
 *                by the Q matrix after the QR has been performed
 * @param R_tmp   Array of size (num_osc - kmin) * (numLEs) to store the R matrix from the QR decomposition
 * @param num_osc The number of oscillators (includes phi_0)
 * @param kmin    The first nonzero wavenumber
 * @param numLEs  The number of LEs (and CLVs) to compute
 */
void orthonormalize(double* pert, double* R_tmp, int num_osc, int kmin, int numLEs) {

	// Initialize vars
	int kdim = num_osc - kmin;

	if (numLEs == 1) {
		// Setup vars
		int n      = kdim;  // dimension of vector
		int incr_x = 1;     // increment to be used

		// Find Norm of vector
		R_tmp[0]  = cblas_dnrm2(n, pert, incr_x);

		// Normalize Vector
		double a = 1 / R_tmp[0];	
		cblas_dscal(n, a, pert, incr_x);
	}
	else {

		// Initialize lapack vars
		lapack_int info;
		lapack_int m   = kdim;
		lapack_int n   = numLEs;
		lapack_int lda = numLEs;
		
		// Allocate temporary memory
		double* tau = (double* )malloc(sizeof(double) * numLEs);
		mem_chk(tau, "tau");	
		
		///---------------
		/// Perform QR Fac
		///---------------
		// Performs a preliminary step - pert holds results - upper trianglur part is R, 
		// lower triangular part holds the house-holder reflecters which are to be used by
		// LAPACKE_dorgqr (as well as tau) to form the Q matrix
		info = LAPACKE_dgeqrf(LAPACK_ROW_MAJOR, m, n, pert, lda, tau);
		if (info < 0 ) {
	    	fprintf(stderr, "ERROR!! |in file \"%s\"-line:%d | LAPACKE Error - %d-th argument contains an illegal value;\n", __FILE__, __LINE__, info);
			exit( 1 );
	    }

		// extract the upper diagonals
		for (int i = 0; i < kdim; ++i) {		
			for (int j = 0 ; j < kdim; ++j) {
				if (j >= i) {
					R_tmp[i * kdim + j] = pert[i * kdim + j];
				}
				else {
					R_tmp[i * kdim + j] = 0.0;
				}
			}
		}

		///---------------
		/// Form the Q matrix
		///---------------
	    info = LAPACKE_dorgqr(LAPACK_ROW_MAJOR, m, n, n, pert, lda, tau);
	    if (info < 0 ) {
	    	fprintf(stderr, "ERROR!! |in file \"%s\"-line:%d | LAPACKE Error - %d-th argument contains an illegal value;\n", __FILE__, __LINE__, info);
			exit( 1 );
	    }

	  	// Free memory
		free(tau);
	}
	
}

/**
 * Function to compute the angles (\in [0, \pi/2)) between the CLVs - uses CBLAS function cblas_ddot() to perform dot product
 * @param angles Output array of size DOF * numLEs to hold the angles - lower triangular
 * @param CLV    Input array of size DOF * numLEs holding the CLVs (in the columns)
 * @param DOF    Degrees of freedom (= num_osc - k_0 - 1)
 * @param numLEs The number of LEs (and CLVs) to be computed
 */
void compute_angles(double* angles, double* CLV, int DOF, int numLEs) {

	double tmp;

	for (int i = 0; i < DOF; ++i) {
		for (int j = 0; j < i; ++j)	{
			
			// Compute the dot product
			tmp = cblas_ddot(DOF, &CLV[i], numLEs, &CLV[j], numLEs);

			// Compute the angle
			angles[i * numLEs + j] = acos(fabs(tmp)); 
		}
	}
}

/**
 * Function to perform the backward dynamics part of the Ginelli et. al., algorith to compute the CLVs
 * First initializes an upper triangular matrix C with random elements. This matrix is evolved backwards 
 * according to C_n = R_nC_{n -1} using the LAPACKE function LAPACKE_dgesv to perform the backwards solve
 * The columns of the resulting matrix are then normalized. Once enough iterations have passed the vectors
 * have converged, the CLVs can then be computed in the tangent space basis V = Q * C. This performed by 
 * the LAPACKE function cblas_dgemm. If needed the angles between these vectors are also computed
 * 
 * @param file_space  Array containing the HDF5 file handle for writing CLVs and angles data to output file 
 * @param data_set    Array containing the HDF5 handles for the datasets of the output file - used for writing to file
 * @param mem_space   Array containing the HDF5 handles for the memory space of datasets - used for writing to file
 * @param R           Array of size DOF * numLEs * m_rev_iters holding all the R matrices from the QR decomps of the forward 
 *                    dynamics
 * @param GS          Array of size DOF * numLEs * (m_rev_iters - m_rev_trans) holding all the tangent space matrices at each 
 *                    iteration of the forward dynamics
 * @param DOF         The number of degrees of freedom (= num_osc - k_0 - 1)
 * @param numLEs      The number of LEs (and CLVs) to be computed
 * @param m_rev_iters The number of iterations of the backward dynamics to perform
 * @param m_rev_trans The number of transient iterations of the backward dynamics
 */
void compute_CLVs(hid_t* file_space, hid_t* data_set, hid_t* mem_space, double* R, double* GS, int DOF, int numLEs, int m_rev_iters, int m_rev_trans) {

	///---------------------------------------
	/// Setup and Initialization
	///---------------------------------------
	// Allocate Memory
	double* R_tmp  = (double* )malloc(sizeof(double) * DOF * numLEs);	
	mem_chk(R_tmp, "R_tmp");
	double* GS_tmp = (double* )malloc(sizeof(double) * DOF * numLEs);	
	mem_chk(GS_tmp, "GS_tmp");
	double* C      = (double* )malloc(sizeof(double) * DOF * numLEs);	
	mem_chk(C, "C");
	double* CLV    = (double* )malloc(sizeof(double) * DOF * numLEs);	
	mem_chk(CLV, "CLV");
	#ifdef __ANGLES
	double* angles = (double* )malloc(sizeof(double) * DOF * numLEs);	
	mem_chk(angles, "angles");
	#endif
	int* pivot  = (int* )malloc(sizeof(double) * numLEs);
	mem_chk(pivot, "pivot");
	double* sum = (double* )malloc(sizeof(double) * numLEs);
	mem_chk(sum, "sum");

	// LAPACKE variables for dgesv function to perform solver on A*X = B
	lapack_int info;
	lapack_int lpk_m   = DOF;    // no. of systems - rows of A
	lapack_int lpk_n   = numLEs; // no. of rhs - cols of B
	lapack_int lpk_lda = DOF;    // leading dimension of A

	// CBLAS variables for dgemm function to perform C = alpha*A*B + beta*C
	double alpha = 1.0;     // prefactor of A*B
	double beta  = 0.0;     // prefactor of C
	int m = DOF;            // no. of rows of A
	int n = numLEs;         // no. of cols of B
	int k = numLEs;         // no. of cols of A / rows of B
	int lda = numLEs;       // leading dim of A - length of elements between consecutive rows
	int ldb = numLEs;       // leading dim of B
	int ldc = numLEs;       // leading dim of C

	// iterator for saving CLVs to file - start in reverse
	int save_clv_indx = (m_rev_iters - m_rev_trans) / SAVE_CLV_STEP - 1;

	///---------------------------------------
	/// Initialize the Coefficients matrix
	///---------------------------------------
	for (int i = 0; i < DOF; ++i) {
		for (int j = 0; j < numLEs; ++j) {
			if(j >= i) {
				// C[i * numLEs + j] = (double) rand() / (double) RAND_MAX;
				C[i * numLEs + j] =  (1.0) / ((double)j + 1.0);
			} 
			else {
				C[i * numLEs + j] = 0.0;
			}
			CLV[i * numLEs + j] = 0.0;
		}
	}


	///---------------------------------------
	/// Backward dynamics part of Ginelli Algo
	///---------------------------------------
	for (int p = (m_rev_iters - 1); p >= 0; --p)
	{	
		//////////////////////
		// Backwards Solve
		//////////////////////
		// Get current R matrix
		for (int i = 0; i < DOF; ++i) {
			for (int j = 0; j < numLEs; ++j) {
				if (j >= i) {
					R_tmp[i * numLEs + j] = R[p * DOF * numLEs + i * numLEs + j];
				}
				else {
					R_tmp[i * numLEs + j] = 0.0;
				}
			}
		}
		
		// Solve the system R_tmp*C_n-1 = C_n to iterate C matrix backwards in time
		info = LAPACKE_dgesv(LAPACK_ROW_MAJOR, lpk_m, lpk_n, R_tmp, lpk_lda, pivot, C, lda);
		if (info > 0) {
			fprintf(stderr, "ERROR!! |in file \"%s\"-line:%d | LAPACKE Error - Diagonal element of the triagnular factor of A,\nU(%i,%i) is zero, so that A is singular;\n", __FILE__, __LINE__, info, info);
			exit( 1 );
		}

		
		//////////////////////
		// Normalize Columns
		//////////////////////
		for (int j = 0; j < numLEs; ++j) {
			sum[j] = 0.0;
			for (int k = 0; k < DOF; ++k) {
				sum[j] += pow(C[k * numLEs + j], 2);
			}
		}
		for (int j = 0; j < numLEs; ++j) {
			for (int k = 0; k < DOF; ++k) {
				C[k * numLEs + j] /= sqrt(sum[j]);
			}
		}

		//////////////////////
		// Compute the CLVs
		//////////////////////
		if (p < (m_rev_iters - m_rev_trans)) {
			// Extract current GS matrix
			for (int i = 0; i < DOF; ++i) {
				for (int j = 0; j < numLEs; ++j) {
					GS_tmp[i * numLEs + j] = GS[p * DOF * numLEs + i * numLEs + j];
				}
			}

			// Perform GS_tmp*C to compute the CLVs in the tangent space basis (spanned by GS vectors)
			cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, alpha, GS_tmp, lda, C, ldb, beta, CLV, ldc);

			// Write CLVs to file
			if (p % SAVE_CLV_STEP == 0) {
				// Write CLVs
				write_hyperslab_data(file_space[4], data_set[4], mem_space[4], H5T_NATIVE_DOUBLE, CLV, "CLV", DOF * numLEs, save_clv_indx);

				#ifdef __ANGLES
				// compute the angles between the CLVs
				compute_angles(angles, CLV, DOF, numLEs);

				// Write angles
				write_hyperslab_data(file_space[5], data_set[5], mem_space[5], H5T_NATIVE_DOUBLE, angles, "Angles", DOF * numLEs, save_clv_indx);
				#endif

				// #ifdef __CLV_STATS
				// // Compute CLVs stats running average
				// compute_clv_stats(CLV, tmp_clv_avg, amp, dof, numLEs, kmin);
				// #endif

				// decrement for next iter
				save_clv_indx--;
			}
		}
	}
	///---------------------------------------
	/// End of Ginelli Algo
	///---------------------------------------

	// for (int i = 0; i < DOF; ++i) {
	// 	printf("Row: %d\n", i);
	// 	for (int j = 0; j < numLEs; ++j) {
	// 		printf("CLVs[%d]: %5.16lf \n", i * numLEs + j, CLV[i * numLEs + j]);
	// 	}
	// }
	// printf("\n\n");


	// Cleanup and free memory
	free(CLV);
	free(C);
	free(R_tmp);
	free(GS_tmp);
	free(sum);
	free(pivot);
	#ifdef __ANGLES
	free(angles);
	#endif
	
}

/**
 * Function to compute the statistics of the CLVs: the CLVs are projected to remove the nuetral direction and then sqaured and averaged over the time direction.
 * Dot products and norms use the energy norm def i.e., v.d = \sum_k a_k v_k d_k
 * @param clv         Input array containing the CLVs, size is dof by numLEs
 * @param tmp_clv_avg Temporary array of size dof to read in the current CLV
 * @param amp         Input array holding the amplitudes
 * @param dof         Variable indicating the avaible degrees of freedom
 * @param numLEs      Variable indicating the number of LEs / CLVs are being considered
 * @param kmin        Variable indicating the first non-zero wavenumber
 */
void compute_clv_stats(double* clv, double* tmp_clv_avg, double* amp, int dof, int numLEs, int kmin) {

	// Allocate local memory
	double* p_k     = (double* )malloc(sizeof(double) * dof);	
	mem_chk(p_k, "p_k");
	double* tmp_clv = (double* )malloc(sizeof(double) * dof);
	mem_chk(tmp_clv, "tmp_clv");

	// norm variable
	double norm;

	// dot prod variables
	double alpha;
	double beta;

	for (int j = 0; j < numLEs; ++j) {
		// Get the current clv
		for (int i = 0; i < dof; ++i) {
			tmp_clv[i] = clv[i * numLEs + j];
		}

		// Get the dot products of v and the neutral direction
		alpha = 0.0;
		beta  = 0.0; 
		for (int i = 0; i < dof; ++i) {
			alpha += pow(amp[i + kmin], 2) * tmp_clv[i] * (i + kmin);
			beta  += pow(amp[i + kmin], 2) * pow(i + kmin, 2);
		}

		// Remove neutral direction
		for (int i = 0; i < dof; ++i) {
			p_k[i] = tmp_clv[i] - (alpha / beta) * (i + kmin);
		}
		
		// Normalize the projected vector
		norm = 1 / cblas_dnrm2(dof, tmp_clv, 1);
		cblas_dscal(dof, norm, p_k, 1);

		// Update running sum used to find the time average
		for (int i = 0; i < dof; ++i) {
			tmp_clv_avg[i * numLEs + j] += pow(p_k[i], 2);
		}
	}	
	
	// Free memory
	free(p_k);
	free(tmp_clv);
}


/**
 * Function to the Benettin et. al., and Ginelli et. al., algorithms to compute the Lyapunov Spectrum and the corresponding 
 * CLVs of the Phase-only Burgers equation
 * 
 * @param N      The number of collocation points
 * @param a      The value of alpha - used to define the amplitudes a_k
 * @param b      The value of beta - used to define the amplitudes a_k
 * @param u0     String containing the initial condition to be used
 * @param k0     The number of modes to kill - used to define the amplitudes
 * @param m_end  The number of iterations of the Benettin/Ginelli et. al., algorith to perform
 * @param m_iter The numbe of integration steps to perform before an iteration of the Benettin/Ginelli et. al., 
 *               algorithm is performed
 * @param numLEs The number of LEs (and CLVs) to compute
 */
void compute_lce_spectrum_clvs(int N, double a, double b, char* u0, int k0, int m_end, int m_iter, int numLEs) {

	// ------------------------------
	//  Variable Definitions
	// ------------------------------
	// Number of modes (incl 0 mode)
	int num_osc = (N / 2) + 1; 

	// padded array size
	int M = 2 * N;

	// Forcing wavenumber
	int kmin = k0 + 1;
	int kmax = num_osc - 1;

	// Triad phases array dims
	int k_range  = kmax - kmin + 1;
	int k1_range = (int)((kmax - kmin + 1)/ 2.0);

	// print update every x iterations
	int print_every = (m_end >= 10 ) ? (int)((double)m_end * 0.1) : 1;
	

	// Looping variables
	int tmp2;
	int tmp;
	int indx;

	// LCE variables
	int dof = num_osc - kmin;
	double lce_sum;
	double dim_sum;
	int dim_indx;


	// ------------------------------
	//  Allocate memory
	// ------------------------------
	// Allocate mode related arrays
	int* kx       = (int* )malloc(sizeof(int) * num_osc);
	mem_chk(kx, "kx");
	double* amp   = (double* )malloc(sizeof(double) * num_osc);
	mem_chk(amp, "amp");
	double* phi   = (double* )malloc(sizeof(double) * num_osc);
	mem_chk(phi, "phi");
	double* u_pad = (double* )malloc(sizeof(double) * M);
	mem_chk(u_pad, "u_pad");
	fftw_complex* u_z     = (fftw_complex* )fftw_malloc(sizeof(fftw_complex) * num_osc);
	mem_chk(u_z, "u_z");
	fftw_complex* u_z_pad = (fftw_complex* )fftw_malloc(sizeof(fftw_complex) * (2 * num_osc - 1));
	mem_chk(u_z_pad, "u_z_pad");
	#ifdef __REALSPACE
	double* u = (double* ) malloc(N * sizeof(double));
	mem_chk(u, "u");
	for (int i = 0; i < N; ++i) u[i] = 0.0;
	#endif
	#ifdef __GRAD
	fftw_complex* u_z_grad = (fftw_complex* ) fftw_malloc(sizeof(fftw_complex) * num_osc);
	mem_chk(u_z_grad, "u_z_grad");
	double* u_grad = (double* ) malloc(sizeof(double) * N);
	mem_chk(u_grad, "u_grad");
	for (int i = 0; i < N; ++i) {
		u_grad[i] = 0.0;
		if (i < num_osc){
			u_z_grad[i] = 0.0 + 0.0 * I;
		}
	}
	#endif

	// LCE Spectrum Arrays
	double* znorm   = (double* )malloc(sizeof(double) * numLEs);	
	mem_chk(znorm, "znorm");
	double* lce     = (double* )malloc(sizeof(double) * numLEs);
	mem_chk(lce, "lce");
	double* run_sum = (double* )malloc(sizeof(double) * numLEs);
	mem_chk(run_sum, "run_sum");

	#ifdef __TRIADS
	// Allocate array for triads
	double* triads = (double* )malloc(sizeof(double) * k_range * k1_range);
	mem_chk(triads, "triads");
	
	// initialize triad array to handle empty elements
	for (int i = 0; i < k_range; ++i) {
		for (int j = 0; j < k1_range; ++j) {
			triads[i * k1_range + j] = -10.0;
		}
	}
	
	// Initilaize Phase Order peramter
	fftw_complex triad_phase_order;
	triad_phase_order = 0.0 + I * 0.0;
	#endif

	#ifdef __TRIAD_ORDER
	// Allocate scale dependent order paramter arrays
	fftw_complex* adler_order_k = (fftw_complex* )fftw_malloc(sizeof(fftw_complex) * num_osc);
	mem_chk(adler_order_k, "adler_order_k");
	fftw_complex* phase_shift_order_k = (fftw_complex* )fftw_malloc(sizeof(fftw_complex) * num_osc);
	mem_chk(phase_shift_order_k, "phase_shift_order_k");
	fftw_complex* theta_time_order_k = (fftw_complex* )fftw_malloc(sizeof(fftw_complex) * num_osc);
	mem_chk(theta_time_order_k, "theta_time_order_k");
	fftw_complex* tmp_time_order_k = (fftw_complex* )fftw_malloc(sizeof(fftw_complex) * num_osc);
	mem_chk(tmp_time_order_k, "tmp_time_order_k");

	// Allocate arrays needed to compute the phase order parameters
	fftw_complex* conv = (fftw_complex* )fftw_malloc(sizeof(fftw_complex) * num_osc);
	mem_chk(conv, "conv");
	double* amp_norm = (double* )malloc(sizeof(double) * num_osc);
	mem_chk(amp_norm, "amp_norm");

	// Allocate memory for the order parameter
	double* R_k = (double * )malloc(sizeof(double) * num_osc);	

	// Initialize normalization counter
	int t_count = 1;

	// Initialize the arrays
	for (int i = 0; i < num_osc; ++i) {
		// Order params
		adler_order_k[i] = 0.0 + 0.0 * I;
		phase_shift_order_k[i] = 0.0 + 0.0 * I;

		// Auxillary arrays
		conv[i]     = 0.0 + 0.0 * I;
		amp_norm[i] = 0.0;
		R_k[i]      = 0.0;
	}
	#endif
	
	
	// ------------------------------
	// Runge-Kutta Variables / Arrays
	// ------------------------------
	// Define RK4 variables
	static double C2 = 0.5, A21 = 0.5, \
				  C3 = 0.5,           A32 = 0.5, \
				  C4 = 1.0,                      A43 = 1.0, \
				            B1 = 1.0/6.0, B2 = 1.0/3.0, B3 = 1.0/3.0, B4 = 1.0/6.0; 

	// Memory fot the four RHS evaluations in the stages 
	double* RK1, *RK2, *RK3, *RK4;
	RK1 = (double* )fftw_malloc(sizeof(double) * num_osc);
	RK2 = (double* )fftw_malloc(sizeof(double) * num_osc);
	RK3 = (double* )fftw_malloc(sizeof(double) * num_osc);
	RK4 = (double* )fftw_malloc(sizeof(double) * num_osc);
	mem_chk(RK1, "RK1");
	mem_chk(RK2, "RK2");
	mem_chk(RK3, "RK3");
	mem_chk(RK4, "RK4");

	// Temp array for intermediate modes
	fftw_complex* u_z_tmp = (fftw_complex* )fftw_malloc(sizeof(fftw_complex) * num_osc);
	mem_chk(u_z_tmp, "u_z_tmp");

	// Memory for the four RHS evalutions for the perturbed system
	double* RK1_pert, *RK2_pert, *RK3_pert, *RK4_pert;
	RK1_pert = (double* )fftw_malloc(sizeof(double) * dof * numLEs);
	RK2_pert = (double* )fftw_malloc(sizeof(double) * dof * numLEs);
	RK3_pert = (double* )fftw_malloc(sizeof(double) * dof * numLEs);
	RK4_pert = (double* )fftw_malloc(sizeof(double) * dof * numLEs);
	mem_chk(RK1_pert, "RK1_pert");
	mem_chk(RK2_pert, "RK2_pert");
	mem_chk(RK3_pert, "RK3_pert");
	mem_chk(RK4_pert, "RK4_pert");

	// Memory for the solution to the perturbed system
	double* pert     = (double* ) malloc(sizeof(double) * dof * numLEs);
	mem_chk(pert, "pert");
	double* pert_tmp = (double* ) malloc(sizeof(double) * dof * numLEs);
	mem_chk(pert_tmp, "pert_tmp");

	// ------------------------------
	//  Create FFTW plans
	// ------------------------------
	// create fftw3 plans objects
	fftw_plan fftw_plan_r2c_pad, fftw_plan_c2r_pad;

	// create plans - ensure no overwriting - fill arrays after
	fftw_plan_r2c_pad = fftw_plan_dft_r2c_1d(M, u_pad, u_z_pad, FFTW_PRESERVE_INPUT); 
	fftw_plan_c2r_pad = fftw_plan_dft_c2r_1d(M, u_z_pad, u_pad, FFTW_PRESERVE_INPUT);
	#if defined(__REALSPACE) || defined(__GRAD)
	fftw_plan fftw_plan_r2c, fftw_plan_c2r;
	fftw_plan_r2c = fftw_plan_dft_r2c_1d(N, u, u_z, FFTW_PRESERVE_INPUT); 
	fftw_plan_c2r = fftw_plan_dft_c2r_1d(N, u_z, u, FFTW_PRESERVE_INPUT);
	#endif



	// ------------------------------
	//  Get Initial Condition
	// ------------------------------
	// Set the initial condition of the perturbed system to the identity matrix
	initial_conditions_lce(pert, phi, amp, u_z, kx, num_osc, k0, kmin, a, b, u0, numLEs);

	// Print IC if small system size
	if (num_osc <= 32) {
		for (int i = 0; i < num_osc; ++i) {
			printf("k[%d]: %d | a: %5.15lf   p: %5.16lf   | u_z[%d]: %+5.16lf\t%+ 5.16lfI \n", i, kx[i], amp[i], phi[i], i, creal(amp[i] * cexp(I * phi[i])), cimag(amp[i] * cexp(I * phi[i])));
		}
		printf("\n\n");
	}


	// ------------------------------
	//  Get Timestep & Integration Vars
	// ------------------------------
	// Get timestep
	double dt = get_timestep(amp, fftw_plan_c2r_pad, fftw_plan_r2c_pad, kx, N, num_osc, k0);

	printf("\n\nTimestep:%6.16lf\n", dt);


	// Get number of transient iterations
	#ifdef __TRANSIENTS
	#ifdef TRANS_STEPS
	// Get no. of transient steps as a percentage of total iterations
	int trans_iters = (int ) (TRANS_STEPS * (m_iter * m_end));	
	int trans_m     = (int ) (TRANS_STEPS * (m_end));
	#else
	// Get no. of transient steps as a ratio of fastest oscillator to slowest
	int trans_iters = get_transient_iters(amp, fftw_plan_c2r_pad, fftw_plan_r2c_pad, kx, N, num_osc, k0);
	int trans_m     = (int ) (trans_iters / m_iter);
	#endif
	#else
	// If no transients - set to 0
	int trans_iters = 0;
	int trans_m     = 0;
	#endif	

	// Get saving variables
	int tot_m_save_steps = (int) m_end / SAVE_LCE_STEP;
	int tot_t_save_steps = (int) (m_iter * m_end)  / SAVE_DATA_STEP;

	// Solver time varibales 
	#ifdef __TRANSIENTS	
	double t0 = trans_m * dt;
	#else
	double t0 = 0.0;
	tot_t_save_steps += 1;    // if no transients include room for saving initial condition
	#endif
	double T  = t0 + m_iter * dt;	

	// printf("Trans_iters: %d", trans_iters);
	// printf("Trans_m: %d\n", trans_m);
	// printf("Trans Iters: %d\nTrans m: %d\nTotal Save steps t: %d\nTotal Save steps m: %d\n\n", trans_iters, trans_m, tot_t_save_steps, tot_m_save_steps);

	// ------------------------------
	//  CLVs Setup
	// ------------------------------
	double* R_tmp = (double* )malloc(sizeof(double) * dof * numLEs);	
	mem_chk(R_tmp, "R_tmp");
	#ifdef __CLVs
	// CLV arrays	
	double* R  = (double* )malloc(sizeof(double) * dof * numLEs * (m_end));	
	mem_chk(R, "R");
	double* GS = (double* )malloc(sizeof(double) * dof * numLEs * (m_end - trans_m));	
	mem_chk(GS, "GS");
	#endif
	#ifdef __CLV_MAX
	// Allocate array to capturing the maxial CLV
	double* max_clv = (double* )malloc(sizeof(double) * dof);
	#endif	
	#ifdef __CLV_STATS
	// Allocate memory for computing CLV stats
	double* tmp_clv_avg = (double* )malloc(sizeof(double) * dof * numLEs);
	mem_chk(tmp_clv_avg, "tmp_clv_avg");
	#ifdef __CLV_MAX
	int max_clv_stats_counter = 0;
	#endif
	#ifdef __CLVs
	int clvs_stats_counter = 0;
	#endif
	#endif
	// saving steps for CLVs
	int tot_clv_save_steps = (int) (m_end - trans_m) / SAVE_CLV_STEP;


	// ------------------------------
	//  HDF5 File Create
	// ------------------------------
	// Create the HDF5 file handle
	hid_t HDF_file_handle;

	// Create variable to handle HDF5 errors
	herr_t status;

	// create hdf5 handle identifiers for hyperslabing the full evolution data
	hid_t HDF_file_space[10];
	hid_t HDF_data_set[10];
	hid_t HDF_mem_space[10];

	// get output file name
	char output_file_name[512];
	get_output_file_name(output_file_name, N, k0, a, b, u0, (m_iter * m_end), m_end, m_iter, trans_iters, numLEs);

	// Create complex datatype for hdf5 file if modes are being recorded
	#if defined(__MODES) || defined(__TRIAD_ORDER)
	hid_t COMPLEX_DATATYPE = create_complex_datatype();
	#else
	hid_t COMPLEX_DATATYPE = -1;
	#endif
	
	// open output file and create hyperslabbed datasets 
	open_output_create_slabbed_datasets_lce(&HDF_file_handle, output_file_name, HDF_file_space, HDF_data_set, HDF_mem_space, COMPLEX_DATATYPE, tot_t_save_steps, tot_m_save_steps, tot_clv_save_steps, num_osc, k_range, k1_range, kmin, numLEs);

	// Create arrays for time and phase order to save after algorithm is finished
	double* time_array      = (double* )malloc(sizeof(double) * (tot_t_save_steps));
	mem_chk(time_array, "time_array");
	#ifdef __TRIADS
	double* phase_order_R   = (double* )malloc(sizeof(double) * (tot_t_save_steps));
	mem_chk(phase_order_R, "phase_order_R");
	double* phase_order_Phi = (double* )malloc(sizeof(double) * (tot_t_save_steps));
	mem_chk(phase_order_Phi, "phase_order_Phi");
	#endif
	

	// ------------------------------
	//  Write Initial Conditions to File
	// ------------------------------
	#ifndef __TRANSIENTS
	#ifdef __PHASES
	write_hyperslab_data(HDF_file_space[0], HDF_data_set[0], HDF_mem_space[0], H5T_NATIVE_DOUBLE, phi, "phi", num_osc, 0);
	#endif

	#ifdef __TRIADS
	// compute triads for initial conditions
	triad_phases(triads, &triad_phase_order, phi, kmin, kmax);
	
	// then write the current triads to this hyperslab
	write_hyperslab_data(HDF_file_space[1], HDF_data_set[1], HDF_mem_space[1], H5T_NATIVE_DOUBLE, triads, "triads", k_range * k1_range, 0);

	#ifdef __MODES
	// Write Initial condition for modes
	write_hyperslab_data(HDF_file_space[7], HDF_data_set[7], HDF_mem_space[7], COMPLEX_DATATYPE, u_z, "u_z", num_osc, 0);
	#endif

	#ifdef __REALSPACE 
	// transform back to Real Space
	fftw_execute_dft_c2r(fftw_plan_c2r, u_z, u);

	// Write Initial condition for real space
	write_hyperslab_data(HDF_file_space[8], HDF_data_set[8], HDF_mem_space[8], H5T_NATIVE_DOUBLE, u, "u", N, 0);
	#endif
	
	phase_order_R[0]   = cabs(triad_phase_order);
	phase_order_Phi[0] = carg(triad_phase_order);
	#endif

	// Write initial time
	time_array[0] = 0.0;
	#endif

	// Algorithm variables
	int m    = 1;
	double t = 0.0;
	int iter = 1;	
	#ifdef __TRANSIENTS
	int save_data_indx = 0;
	int save_lce_indx  = 0;
	#else
	int save_data_indx = 1;
	int save_lce_indx  = 0;
	#endif
	// ------------------------------
	//  Begin Algorithm
	// ------------------------------
	while (m <= (trans_m + m_end)) {

		// ------------------------------
		//  Integrate System Forward
		// ------------------------------
		for (int p = 0; p < m_iter; ++p) {

			// Construct the modes
			for (int i = 0; i < num_osc; ++i) {
				u_z_tmp[i] = amp[i] * cexp(I * phi[i]);
			}

			//////////////
			// STAGES
			//////////////
			/*---------- STAGE 1 ----------*/
			// find RHS first and then update stage
			po_rhs_extended(RK1, RK1_pert, u_z_tmp, pert, &fftw_plan_c2r_pad, &fftw_plan_r2c_pad, kx, N, num_osc, k0, numLEs);
			for (int i = 0; i < num_osc; ++i) {
				u_z_tmp[i] = amp[i] * cexp(I * (phi[i] + A21 * dt * RK1[i]));
				if (i < dof) {
					tmp = i * numLEs;
					for (int j = 0; j < numLEs; ++j) {
						indx = tmp + j;
						pert_tmp[indx] = pert[indx] + A21 * dt * RK1_pert[indx];
					}
				}
				// if (m == 1) {
				// 	printf("RK1[%d]: %6.16lf\n", i, RK1[i]);
				// }
			}
			// if (m == 1) {
			// 	printf("\n\n");
			// }


			

			/*---------- STAGE 2 ----------*/
			// find RHS first and then update stage
			po_rhs_extended(RK2, RK2_pert, u_z_tmp, pert_tmp, &fftw_plan_c2r_pad, &fftw_plan_r2c_pad, kx, N, num_osc, k0, numLEs) ;
			for (int i = 0; i < num_osc; ++i) {
				u_z_tmp[i] = amp[i] * cexp(I * (phi[i] + A32 * dt * RK2[i]));
				if (i < dof) {
					tmp = i * numLEs;
					for (int j = 0; j < numLEs; ++j) {
						indx = tmp + j;
						pert_tmp[indx] = pert[indx] + A21 * dt * RK2_pert[indx];
					}
				}
			// printf("RK2[%d]: %6.16lf\n", i, RK2[i]);
			}
			// printf("\n\n");
			

			/*---------- STAGE 3 ----------*/
			// find RHS first and then update stage
			po_rhs_extended(RK3, RK3_pert, u_z_tmp, pert_tmp, &fftw_plan_c2r_pad, &fftw_plan_r2c_pad, kx, N, num_osc, k0, numLEs) ;
			for (int i = 0; i < num_osc; ++i) {
				u_z_tmp[i] = amp[i] * cexp(I * (phi[i] + A43 * dt * RK3[i]));
				if (i < dof) {
					tmp = i * numLEs;
					for (int j = 0; j < numLEs; ++j) {
						indx = tmp + j;
						pert_tmp[indx] = pert[indx] + A43 * dt * RK3_pert[indx];
					}
				}
			// printf("RK3[%d]: %6.16lf\n", i, RK3[i]);
			}
			// printf("\n\n");

			
			/*---------- STAGE 4 ----------*/
			// find RHS first and then update 
			po_rhs_extended(RK4, RK4_pert, u_z_tmp, pert_tmp, &fftw_plan_c2r_pad, &fftw_plan_r2c_pad, kx, N, num_osc, k0, numLEs) ;

			
			//////////////
			// Update
			//////////////
			for (int i = 0; i < num_osc; ++i) {
				phi[i] = phi[i] + (dt * B1) * RK1[i] + (dt * B2) * RK2[i] + (dt * B3) * RK3[i] + (dt * B4) * RK4[i];  
					if (i < dof) {
						tmp = i * numLEs;
						for (int j = 0; j < numLEs; ++j) {
							indx = tmp + j;
							pert[indx] = pert[indx] + (dt * B1) * RK1_pert[indx] + (dt * B2) * RK2_pert[indx] + (dt * B3) * RK3_pert[indx] + (dt * B4) * RK4_pert[indx];  
						}
					}
			// printf("RK4[%d]: %6.16lf\n", i, RK4[i]);
			}
			// printf("\n\n");
			
			// for (int i = 0; i < num_osc; ++i)
			// {
			// 	printf("phi[%d]: %6.16lf\n", i, phi[i]);
			// }
			// printf("\n\n");

			// for (int i = 0; i < dof; ++i)
			// {
			// 	for (int j = 0; j < numLEs; ++j)
			// 	{
			// 		printf("RK1_pert[%d]: %5.16lf \n", i * numLEs + j, RK1_pert[i * numLEs + j]);
			// 	}
			// 	printf("\n");
			// }
			// printf("\n\n");

			// printf("Iter: %d\n", iter);

			// printf("\n\n");

			/////////////////
			// Print to file
			/////////////////
			if ((iter > trans_iters) && (iter % SAVE_DATA_STEP == 0)) {


				// Write phases
				#ifdef __PHASES
				write_hyperslab_data(HDF_file_space[0], HDF_data_set[0], HDF_mem_space[0], H5T_NATIVE_DOUBLE, phi, "phi", num_osc, save_data_indx);
				#endif

				#ifdef __TRIADS
				// compute triads for initial conditions
				triad_phases(triads, &triad_phase_order, phi, kmin, kmax);

				// write triads
				write_hyperslab_data(HDF_file_space[1], HDF_data_set[1], HDF_mem_space[1], H5T_NATIVE_DOUBLE, triads, "triads", k_range * k1_range, save_data_indx);

				// save phase order parameters
				phase_order_R[save_data_indx]   = cabs(triad_phase_order);
				phase_order_Phi[save_data_indx] = carg(triad_phase_order);
				#endif

				#if defined(__MODES) || defined(__GRAD) || defined(__REALSPACE) || defined(__TRIAD_ORDER)
				// Construct the modes
				for (int i = 0; i < num_osc; ++i) {
					u_z[i] = amp[i] * cexp(I * phi[i]);

					#ifdef __GRAD
					u_z_grad[i] = I * kx[i] * u_z[i];
					#endif
				}
				#endif
				#ifdef __MODES
				// Write current modes
				write_hyperslab_data(HDF_file_space[7], HDF_data_set[7], HDF_mem_space[7], COMPLEX_DATATYPE, u_z, "u_z", num_osc, save_data_indx);
				#endif
				
				#ifdef __REALSPACE
				// transform back to Real Space
				fftw_execute_dft_c2r(fftw_plan_c2r, u_z, u);			
				for (int i = 0; i < N; ++i)	{
					u[i] /= sqrt((double) N);  // Normalize inverse transfom	
					// printf("u[%d]: %6.16lf\n", i, u[i]);	
				}
				// printf("\n\n");
				// Write real space
				write_hyperslab_data(HDF_file_space[8], HDF_data_set[8], HDF_mem_space[8], H5T_NATIVE_DOUBLE, u, "u", N, save_data_indx);
				#endif
				
				#ifdef __GRAD
				// transform back to Real Space
				fftw_execute_dft_c2r(fftw_plan_c2r, u_z_grad, u_grad);
				for (int i = 0; i < N; ++i)	{
					u_grad[i] /= sqrt((double) N);  // Normalize inverse transfom
				// printf("u_g[%d]: %6.16lf\n", i, u_grad[i]);	
				}
				// printf("\n\n");
				
				// Write real space
				write_hyperslab_data(HDF_file_space[9], HDF_data_set[9], HDF_mem_space[9], H5T_NATIVE_DOUBLE, u_grad, "u_grad", N, save_data_indx);
				#endif

				#ifdef __TRIAD_ORDER
				// Get the normalization constant
				if (save_data_indx == 0) {
					amp_normalize(amp_norm, amp, num_osc, k0);
				}

				// Get the convolution
				conv_2N_pad(conv, u_z, &fftw_plan_r2c_pad, &fftw_plan_c2r_pad, N, num_osc, k0);

				// compute scale dependent phase order parameter
				for (int i = k0 + 1; i < num_osc; ++i) {
					// Proposed scale dependent Kuramoto order parameters
					adler_order_k[i]       = -I * (cexp(I * 2.0 *  phi[i]) * conj(conv[i])); // / amp_norm[i]);
					
					// Scale dependent Phase shift order parameter
					phase_shift_order_k[i] = I * (conv[i] * cexp(-I * phi[i])); // / amp_norm[i]);

					// Scale dependent Kuramoto order parameter in time of the phase shift
					tmp_time_order_k[i]   += cexp(I * carg(phase_shift_order_k[i]));
					theta_time_order_k[i] = tmp_time_order_k[i] / t_count;
				}

				// increment the counter			
				t_count++;
				#endif


				////////////////////////////////////////////////// Should this be here??
				#ifdef __CLV_MAX
				// Get the first vector from the perturbation matrix
				for (int i = 0; i < dof; ++i) {
					max_clv[i] = pert[i * numLEs];
				}

				#ifdef __CLV_STATS
				// Do running clv stats calculation
				compute_clv_stats(max_clv, tmp_clv_avg, amp, dof, 1, kmin);
				
				// Update normalization counter
				max_clv_stats_counter++;
				#endif

				// Write maximal CLV to file
				write_hyperslab_data(HDF_file_space[6], HDF_data_set[6], HDF_mem_space[6], H5T_NATIVE_DOUBLE, max_clv, "MaxCLV", dof, save_lce_indx);				
				#endif

				// save time
				time_array[save_data_indx] = iter * dt;
				
				// increment indx for next iteration
				save_data_indx += 1;
			}
			

			// increment
			t    = iter*dt;			
			iter += 1;			
		}		
		// ------------------------------
		//  End Integration
		// ------------------------------
		
		// ------------------------------
		//  Orthonormalize 
		// ------------------------------
		orthonormalize(pert, R_tmp, num_osc, kmin, numLEs);
		// ------------------------------
		//  Orthonormalize 
		// ------------------------------
		
		// ------------------------------
		//  Compute LCEs & Write To File
		// ------------------------------
		if (m > trans_m) {
			
			// Record the GS vectors and R matrix and extract the diagonals of R
			tmp2 = (m - trans_m - 1) * dof * numLEs;
			for (int i = 0; i < dof; ++i) {
				tmp = i * numLEs;		
				for (int j = 0 ; j < numLEs; ++j) {

					// Record diagonals of R matrix (checking for sign correction)
					if (i == j) {
						znorm[i] = fabs(R_tmp[tmp + i]);
					} 

					#ifdef __CLVs
					// Record upper triangular R matrix
					if (j >= i) {
						R[tmp2 + tmp + j] = R_tmp[tmp + j];					
					}

					// Record the GS vectors
					if (m < (m_end - trans_m)) {
						GS[tmp2 + tmp + j] = pert[tmp + j];
					}
					#endif
				}
			}
			
			////////////////
			// Compute LCEs
			////////////////
			for (int i = 0; i < numLEs; ++i) {
				// Compute LCE
				run_sum[i] = run_sum[i] + log(znorm[i]);
				lce[i]     = run_sum[i] / (t - t0);
			}

			// then write the current LCEs to this hyperslab
			if (m % SAVE_LCE_STEP == 0) {		
				#ifdef __LCE_ALL	
				write_hyperslab_data(HDF_file_space[2], HDF_data_set[2], HDF_mem_space[2], H5T_NATIVE_DOUBLE, lce, "lce", numLEs, save_lce_indx);
				#endif
				#ifdef __RNORM
				write_hyperslab_data(HDF_file_space[3], HDF_data_set[3], HDF_mem_space[3], H5T_NATIVE_DOUBLE, znorm, "rNorm", numLEs, save_lce_indx);
				#endif
				
				save_lce_indx += 1;
			}

			// Print update to screen
			#ifdef PRINT_SCREEN
			if (m % print_every == 0) {
				lce_sum = 0.0;
				dim_sum = 0.0;
				dim_indx = 0;
				for (int i = 0; i < numLEs; ++i) {
					// Get spectrum sum
					lce_sum += lce[i];

					// Compute attractor dim
					if (dim_sum + lce[i] > DBL_EPSILON) {
						dim_sum += lce[i];
						dim_indx += 1;
					}
					else {
						continue;
					}
				}
				printf("Iter: %d / %d | t: %6.3lf tsteps: %d | k0:%d alpha: %2.3lf beta: %2.3lf | Sum: %5.9lf | Dim: %5.9lf\n", m, m_end + trans_m, t, m_end * m_iter, k0, a, b, lce_sum, (dim_indx + (dim_sum / fabs(lce[dim_indx]))));
				#ifdef PRINT_LCEs
				printf("k: \n");
				for (int j = 0; j < numLEs; ++j) {
					printf("%5.6lf ", lce[j]);
				}
				printf("\n\n");
				#endif
			}
			#endif
			#if defined(__CLV_STATS) && defined(__CLVs)
			// update counter for computing CLVs stats
			clvs_stats_counter++;
			#endif
		}

		// ------------------------------
		//  Update For Next Iteration
		// ------------------------------
		T = T + m_iter * dt;
		m += 1;
		#ifdef __TRANSIENTS
		if (m - 1 == trans_m) {
			printf("\n\t!!Transient Iterations Complete!! - Iters: %d\n\n\n", iter - 1);
		}
		#endif
	}
	// ------------------------------
	//  Compute the CLVs
	// ------------------------------
	#ifdef __CLVs
	compute_CLVs(HDF_file_space, HDF_data_set, HDF_mem_space, R, GS, dof, numLEs, m_end, trans_m);
	#endif
	// ------------------------------
	// End Algorithm
	// ------------------------------
	


	// ------------------------------
	//  Write 1D Arrays Using HDF5Lite
	// ------------------------------
	hid_t D1 = 1;
	hid_t D1dims[D1];

	// Write amplitudes
	D1dims[0] = num_osc;
	if ( (H5LTmake_dataset(HDF_file_handle, "Amps", D1, D1dims, H5T_NATIVE_DOUBLE, amp)) < 0){
		fprintf(stderr, "\n\n!!Failed to make - Amps - Dataset!!\n\n");
	}

	// Wtie time
	D1dims[0] = tot_t_save_steps;
	if ( (H5LTmake_dataset(HDF_file_handle, "Time", D1, D1dims, H5T_NATIVE_DOUBLE, time_array)) < 0) {
		fprintf(stderr, "\n\n!!Failed to make - Time - Dataset!!\n\n");
	}

	#ifdef __LCE_LAST
	// Write the final state of the LCE computation
	D1dims[0] = numLEs;
	if ( (H5LTmake_dataset(HDF_file_handle, "FinalLCE", D1, D1dims, H5T_NATIVE_DOUBLE, lce)) < 0) {
		fprintf(stderr, "\n\n!!Failed to make - FinalLCE - Dataset!!\n\n");
	}
	#endif
	#ifdef __TRIADS
	// Write Phase Order R
	D1dims[0] = tot_t_save_steps;
	if ( (H5LTmake_dataset(HDF_file_handle, "PhaseOrderR", D1, D1dims, H5T_NATIVE_DOUBLE, phase_order_R)) < 0) {
		fprintf(stderr, "\n\n!!Failed to make - PhaseOrderR - Dataset!!\n\n");
	}

	// Write Phase Order Phi
	D1dims[0] = tot_t_save_steps;
	if ( (H5LTmake_dataset(HDF_file_handle, "PhaseOrderPhi", D1, D1dims, H5T_NATIVE_DOUBLE, phase_order_Phi)) < 0) {
		fprintf(stderr, "\n\n!!Failed to make - PhaseOrderPhi - Dataset!!\n\n");
	}
	#endif
	#ifdef __TRIAD_ORDER
	// Write the final state of the scale dependent triad order parameter
	for (int i = 0; i < num_osc; ++i) {
		R_k[i] = cabs(theta_time_order_k[i]);
	}
	D1dims[0] = num_osc;
	if ( (H5LTmake_dataset(HDF_file_handle, "R_k", D1, D1dims, H5T_NATIVE_DOUBLE, R_k)) < 0) {
		fprintf(stderr, "\n\n!!Failed to make - R_k - Dataset!!\n\n");
	}
	#endif
	#ifdef __CLV_STATS
	// Normalize the CLV stats
	for (int i = 0; i < dof; ++i) {
		for (int j = 0; j < numLEs; ++j) {
			#ifdef __CLV_MAX
			tmp_clv_avg[i * numLEs + j] /= max_clv_stats_counter;
			#endif
			#ifdef __CLVs
			tmp_clv_avg[i * numLEs + j] /= clvs_stats_counter;
			#endif
		}
	}

	// Write clv stats to file
	#ifdef __CLV_MAX
	D1dims[0] = dof;
	if ( (H5LTmake_dataset(HDF_file_handle, "MaxCLVStats", D1, D1dims, H5T_NATIVE_DOUBLE, tmp_clv_avg)) < 0) {
		fprintf(stderr, "\n\n!!Failed to make - MaxCLVStats - Dataset!!\n\n");
	}
	#endif
	#ifdef __CLVs
	hid_t D2 = 2;
	hid_t D2dims[D2];
	D2dims[0] = dof;
	D2dims[1] = numLEs;
	if ( (H5LTmake_dataset(HDF_file_handle, "CLVsStats", D2, D2dims, H5T_NATIVE_DOUBLE, tmp_clv_avg)) < 0) {
	fprintf(stderr, "\n\n!!Failed to make - CLVStats - Dataset!!\n\n");
	}
	#endif
	#endif
	


	// printf("Total Iters: %d\nTotal m Iters: %d \n\n", iter, m_iter);
	// for (int i = 0; i < num_osc; ++i) {
	// 	printf("phi[%d]: %5.16lf\n", i, phi[i]);
	// }
	// printf("\n");

	// for (int i = 0; i < dof; ++i) {
	// 	for (int j = 0; j < numLEs; ++j) {
	// 		printf("pert[%d]: %5.16lf ", i * dof + j, pert[i * dof + j]);
	// 	}
	// 	printf("\n");
	// }
	// printf("\n\n");

	// for (int i = 0; i < numLEs; ++i) {
	// 	printf("LCE[%d]: %5.16lf\n", i, lce[i]);
	// }
	// printf("\n");
	




	// ------------------------------
	//  Clean Up & Exit
	// ------------------------------
	// Destroy DFT plans
	#if defined(__REALSPACE) || defined(__GRAD)
	fftw_destroy_plan(fftw_plan_r2c);
	fftw_destroy_plan(fftw_plan_c2r);
	#endif
	fftw_destroy_plan(fftw_plan_r2c_pad);
	fftw_destroy_plan(fftw_plan_c2r_pad);
	
	// Free memory
	#ifdef __TRIADS
	free(triads);
	free(phase_order_Phi);
	free(phase_order_R);
	#endif
	#ifdef __CLVs
	free(R);
	free(GS);
	#endif
	#ifdef __TRIAD_ORDER
	fftw_free(adler_order_k);
	fftw_free(phase_shift_order_k);	
	fftw_free(theta_time_order_k);	
	fftw_free(conv);
	free(amp_norm);
	free(R_k);
	#endif
	#ifdef __CLV_MAX
	free(max_clv);
	#endif
	#ifdef __CLV_STATS
	free(tmp_clv_avg);
	#endif
	#ifdef __REALSPACE
	free(u);
	#endif
	#ifdef _GRAD
	fftw_free(u_z_grad);
	free(u_grad);
	#endif
	free(R_tmp);
	free(kx);
	free(amp);
	free(phi);
	free(u_pad);
	free(znorm);
	free(lce);
	free(run_sum);
	free(time_array);
	free(RK1);
	free(RK2);
	free(RK3);
	free(RK4);
	free(RK1_pert);
	free(RK2_pert);
	free(RK3_pert);
	free(RK4_pert);
	free(pert);
	free(pert_tmp);
	fftw_free(u_z);
	fftw_free(u_z_tmp);
	fftw_free(u_z_pad);

	// destroy the complex comound datatype
	#if defined(__MODES) || defined(__TRIAD_ORDER)
	status = H5Tclose(COMPLEX_DATATYPE);
	#endif
	

	// Close HDF5 handles
	#ifdef __TRIADS
	status = H5Sclose( HDF_mem_space[1] );
	status = H5Dclose( HDF_data_set[1] );
	status = H5Sclose( HDF_file_space[1] );
	#endif
	#ifdef __CLVs
	status = H5Sclose( HDF_mem_space[4] );
	status = H5Dclose( HDF_data_set[4] );
	status = H5Sclose( HDF_file_space[4] );
	#ifdef __ANGLES
	status = H5Sclose( HDF_mem_space[5] );
	status = H5Dclose( HDF_data_set[5] );
	status = H5Sclose( HDF_file_space[5] );
	#endif
	#endif
	#ifdef __RNORM
	status = H5Sclose( HDF_mem_space[3] );
	status = H5Dclose( HDF_data_set[3] );
	status = H5Sclose( HDF_file_space[3] );
	#endif
	#ifdef __PHASES
	status = H5Sclose( HDF_mem_space[0] );
	status = H5Dclose( HDF_data_set[0] );
	status = H5Sclose( HDF_file_space[0] );
	#endif
	#ifdef __LCE_ALL
	status = H5Sclose( HDF_mem_space[2] );
	status = H5Dclose( HDF_data_set[2] );
	status = H5Sclose( HDF_file_space[2] );
	#endif 
	#ifdef __CLV_MAX
	status = H5Sclose( HDF_mem_space[6] );
	status = H5Dclose( HDF_data_set[6] );
	status = H5Sclose( HDF_file_space[6] );
	#endif
	#ifdef __MODES
	status = H5Sclose( HDF_mem_space[7] );
	status = H5Dclose( HDF_data_set[7] );
	status = H5Sclose( HDF_file_space[7] );
	#endif
	#ifdef __REALSPACE
	status = H5Sclose( HDF_mem_space[8] );
	status = H5Dclose( HDF_data_set[8] );
	status = H5Sclose( HDF_file_space[8] );
	#endif
	#ifdef __GRAD
	status = H5Sclose( HDF_mem_space[9] );
	status = H5Dclose( HDF_data_set[9] );
	status = H5Sclose( HDF_file_space[9] );
	#endif

	// Close output file
	status = H5Fclose(HDF_file_handle);
}
// ---------------------------------------------------------------------
//  End of File
// ---------------------------------------------------------------------