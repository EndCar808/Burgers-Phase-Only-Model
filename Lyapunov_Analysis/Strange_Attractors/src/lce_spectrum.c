// Enda Carroll
// Sept 2020
// File including functions to perform the Benettin et al. and Ginelli et al, 
// algorithms for computing the Lyapunov spectrum and vectors of the Lorenz system


// ---------------------------------------------------------------------
//  Standard Libraries and Headers
// ---------------------------------------------------------------------
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <float.h>
#include <string.h>
#include <hdf5.h>
#include <hdf5_hl.h>
#include <gsl/gsl_cblas.h>
#include <lapacke.h>


// ---------------------------------------------------------------------
//  Definitions, User Libraries and Headers
// ---------------------------------------------------------------------
#define RHO 28.0
#define SIGMA 10.0
#define BETA 2.6666666666666667

#define SAVE_DATA_STEP 100 
#define SAVE_LCE_STEP  100
#define SAVE_CLV_STEP  10


// ---------------------------------------------------------------------
//  Function Definitions
// ---------------------------------------------------------------------
void mem_chk (void *arr_ptr, char *name) {
  if (arr_ptr == NULL ) {
    fprintf(stderr, "ERROR!! |in file \"%s\"-line:%d | Unable to malloc required memory for %s, now exiting!\n", __FILE__, __LINE__, name);
    exit(1);
  }
}



void get_output_file_name(char* output_file_name, double dt, int m_iter, int m_trans, int m_avg) {

	// Create Output File Locatoin
	char output_dir[512] = "./Data/";
	char output_dir_tmp[512];
	sprintf(output_dir_tmp,  "LorenzData_RHO[%.2lf]_TSTEP[%.6lf]_INTSTEPS[%d]_TRANSSTEPS[%d]_AVGSTEPS[%d].h5", RHO, dt, m_iter, m_trans, m_avg);
	strcat(output_dir, output_dir_tmp);
	strcpy(output_file_name, output_dir);

	// Print file name to screen
	printf("\nOutput File: %s \n\n", output_file_name);	
}


void open_output_create_slabbed_datasets_lce(hid_t* file_handle, char* output_file_name, hid_t* file_space, hid_t* data_set, hid_t* mem_space, int num_t_steps, int num_m_steps, int num_clv_steps, int N, int numLEs) {
	// ------------------------------
	//  Create file
	// ------------------------------	
	// create datafile - H5F_ACC_TRUNC overwrites file if it exists already
	*file_handle = H5Fcreate(output_file_name, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);


	// ------------------------------
	//  Create datasets with hyperslabing
	// ------------------------------
	//
	//---------- PhaseSpace -----------//
	//
	// create hdf5 dimension arrays for creating the hyperslabs
	static const int dimensions = 2;
	hsize_t dims[dimensions];      // array to hold dims of full evolution data
	hsize_t maxdims[dimensions];   // array to hold max dims of full evolution data
	hsize_t chunkdims[dimensions]; // array to hold dims of the hyperslab chunks

	// initialize the hyperslab arrays
	dims[0]      = num_t_steps;             // number of timesteps
	dims[1]      = N;                 // number of oscillators
	maxdims[0]   = H5S_UNLIMITED;           // setting max time index to unlimited means we must chunk our data
	maxdims[1]   = N;                 // same as before = number of modes
	chunkdims[0] = 1;                       // 1D chunk to be saved 
	chunkdims[1] = N;                 // 1D chunk of size number of modes

	// create the 2D dataspace - setting the no. of dimensions, expected and max size of the dimensions
	file_space[0] = H5Screate_simple(dimensions, dims, maxdims);

	// must create a propertly list to enable data chunking due to max time dimension being unlimited
	// create property list 
	hid_t plist;
	plist = H5Pcreate(H5P_DATASET_CREATE);

	// using this property list set the chuncking - stores the chunking info in plist
	H5Pset_chunk(plist, dimensions, chunkdims);

	// Create the dataset in the previouosly created datafile - using the chunk enabled property list and new compound datatype
	data_set[0] = H5Dcreate(*file_handle, "PhaseSpace", H5T_NATIVE_DOUBLE, file_space[0], H5P_DEFAULT, plist, H5P_DEFAULT);
	
	// create the memory space for the slab
	dims[0] = 1;
	dims[1] = N;

	// setting the max dims to NULL defaults to same size as dims
	mem_space[0] = H5Screate_simple(dimensions, dims, NULL);

	H5Pclose(plist);


	//---------- LCE -----------//
	//
	// initialize the hyperslab arrays
	dims[0]      = num_m_steps;             // number of timesteps
	dims[1]      = numLEs;          // number of oscillators
	maxdims[0]   = H5S_UNLIMITED;           // setting max time index to unlimited means we must chunk our data
	maxdims[1]   = numLEs;          // same as before = number of modes
	chunkdims[0] = 1;                       // 1D chunk to be saved 
	chunkdims[1] = numLEs;          // 1D chunk of size number of modes

	// create the 2D dataspace - setting the no. of dimensions, expected and max size of the dimensions
	file_space[3] = H5Screate_simple(dimensions, dims, maxdims);

	// must create a propertly list to enable data chunking due to max time dimension being unlimited
	// create property list 
	hid_t plist3;
	plist3 = H5Pcreate(H5P_DATASET_CREATE);

	// using this property list set the chuncking - stores the chunking info in plist
	H5Pset_chunk(plist3, dimensions, chunkdims);

	// Create the dataset in the previouosly created datafile - using the chunk enabled property list and new compound datatype
	data_set[3] = H5Dcreate(*file_handle, "LCE", H5T_NATIVE_DOUBLE, file_space[3], H5P_DEFAULT, plist3, H5P_DEFAULT);
	
	// create the memory space for the slab
	dims[0] = 1;
	dims[1] = numLEs;

	// setting the max dims to NULL defaults to same size as dims
	mem_space[3] = H5Screate_simple(dimensions, dims, NULL);

	H5Pclose(plist3);

	
	//
	//---------- CLVs -----------//
	//
	// create hdf5 dimension arrays for creating the hyperslabs
	static const int dim2D = 2;
	hsize_t dims2D[dim2D];      // array to hold dims of full evolution data
	hsize_t maxdims2D[dim2D];   // array to hold max dims of full evolution data
	hsize_t chunkdims2D[dim2D]; // array to hold dims of the hyperslab chunks

	// initialize the hyperslab arrays
	dims2D[0]      = num_clv_steps;  // number of timesteps + initial condition
	dims2D[1]      = N * numLEs;     // size of CLV array
	maxdims2D[0]   = H5S_UNLIMITED;  // setting max time index to unlimited means we must chunk our data
	maxdims2D[1]   = N * numLEs;     // size of CLV array
	chunkdims2D[0] = 1;              // 1D chunk to be saved 
	chunkdims2D[1] = N * numLEs;     // size of CLV array

	// create the 2D dataspace - setting the no. of dimensions, expected and max size of the dimensions
	file_space[1] = H5Screate_simple(dim2D, dims2D, maxdims2D);

	// must create a propertly list to enable data chunking due to max time dimension being unlimited
	// create property list 
	hid_t plist5;
	plist5 = H5Pcreate(H5P_DATASET_CREATE);

	// using this property list set the chuncking - stores the chunking info in plist
	H5Pset_chunk(plist5, dim2D, chunkdims2D);

	// Create the dataset in the previouosly created datafile - using the chunk enabled property list and new compound datatype
	data_set[1] = H5Dcreate(*file_handle, "CLVs", H5T_NATIVE_DOUBLE, file_space[1], H5P_DEFAULT, plist5, H5P_DEFAULT);
	
	// create the memory space for the slab
	dims2D[0] = 1;
	dims2D[1] = N * numLEs;

	// setting the max dims to NULL defaults to same size as dims
	mem_space[1] = H5Screate_simple(dim2D, dims2D, NULL);

	// Create attribute data for the CLV dimensions
	hid_t CLV_attr, CLV_attr_space;

	hsize_t CLV_adims[2];
	CLV_adims[0] = 1;
	CLV_adims[1] = 2;

	CLV_attr_space = H5Screate_simple (2, CLV_adims, NULL);

	CLV_attr = H5Acreate(data_set[1], "CLV_Dims", H5T_NATIVE_INT, CLV_attr_space, H5P_DEFAULT, H5P_DEFAULT);

	int CLV_dims[2];
	CLV_dims[0] = N;
	CLV_dims[1] = numLEs;

    herr_t status = H5Awrite(CLV_attr, H5T_NATIVE_INT, CLV_dims);

	// close the created property list
	status = H5Aclose(CLV_attr);
    status = H5Sclose(CLV_attr_space);
	status = H5Pclose(plist5);

	//
	//---------- Angles -----------//
	//	
	// initialize the hyperslab arrays
	dims2D[0]      = num_clv_steps; // number of timesteps + initial condition
	dims2D[1]      = N * numLEs;    // size of angles array
	maxdims2D[0]   = H5S_UNLIMITED; // setting max time index to unlimited means we must chunk our data
	maxdims2D[1]   = N * numLEs;    // size of angles array
	chunkdims2D[0] = 1;             // 1D chunk to be saved 
	chunkdims2D[1] = N * numLEs;    // size of angles array

	// create the 2D dataspace - setting the no. of dimensions, expected and max size of the dimensions
	file_space[2] = H5Screate_simple(dim2D, dims2D, maxdims2D);

	// must create a propertly list to enable data chunking due to max time dimension being unlimited
	// create property list 
	hid_t plist6;
	plist6 = H5Pcreate(H5P_DATASET_CREATE);

	// using this property list set the chuncking - stores the chunking info in plist
	H5Pset_chunk(plist6, dim2D, chunkdims2D);

	// Create the dataset in the previouosly created datafile - using the chunk enabled property list and new compound datatype
	data_set[2] = H5Dcreate(*file_handle, "Angles", H5T_NATIVE_DOUBLE, file_space[2], H5P_DEFAULT, plist6, H5P_DEFAULT);
	
	// create the memory space for the slab
	dims2D[0] = 1;
	dims2D[1] = N * numLEs;

	// setting the max dims to NULL defaults to same size as dims
	mem_space[2] = H5Screate_simple(dim2D, dims2D, NULL);

	// Create attribute data for the Angles dimensions
	CLV_adims[0] = 1;
	CLV_adims[1] = 2;

	// Create attribute data for the CLV dimensions
	hid_t Angles_attr, Angles_attr_space;

	Angles_attr_space = H5Screate_simple (2, CLV_adims, NULL);

	Angles_attr = H5Acreate(data_set[2], "Angle_Dims", H5T_NATIVE_INT, Angles_attr_space, H5P_DEFAULT, H5P_DEFAULT);

	CLV_dims[0] = N;
	CLV_dims[1] = numLEs;

  	status = H5Awrite(Angles_attr, H5T_NATIVE_INT, CLV_dims);

	// close the created property list
	status = H5Aclose(Angles_attr);
    status = H5Sclose(Angles_attr_space);
	status = H5Pclose(plist6);

}

void write_hyperslab_data_d(hid_t file_space, hid_t data_set, hid_t mem_space, double* data, char* data_name, int n, int index) {

	// Create dimension arrays for hyperslab
	hsize_t start_index[2]; // stores the index in the hyperslabbed dataset to start writing to
	hsize_t count[2];       // stores the size of hyperslab to write to the dataset

	count[0]       = 1;		// 1D slab so first dim is 1
	count[1]       = n;		// 1D slab of size of data array
	start_index[0] = index;	// set the starting row index to index in the global dataset to write slab to
	start_index[1] = 0;		// set column index to 0 to start writing from the first column

	// select appropriate hyperslab 
	if ((H5Sselect_hyperslab(file_space, H5S_SELECT_SET, start_index, NULL, count, NULL)) < 0) {
		printf("\n!!Error Selecting Hyperslab!! - For %s at Index: %d \n", data_name, index);
	}

	// then write the current modes to this hyperslab
	if ((H5Dwrite(data_set, H5T_NATIVE_DOUBLE, mem_space, file_space, H5P_DEFAULT, data)) < 0) {
		printf("\n!!Error Writing Slabbed Data!! - For %s at Index: %d \n", data_name, index);
	}
}


void initial_conditions_lce(double* pert, double* x, int N, int numLEs) {

	// Initial State
	x[0] = -1.01;
	x[1] = 3.01;
	x[2] = 2.01;

	// Initial perturbation matrix
	memset(pert, 0.0, sizeof(double) * N * numLEs);

	for (int i = 0; i < numLEs; ++i) {
		pert[i * numLEs + i] = 1.0;
	}
}

void rhs_extended(double* rhs, double* rhs_ext, double* x_tmp, double* x_pert, int n, int numLEs) {

	// Update RHS
	rhs[0] = SIGMA * (x_tmp[1] - x_tmp[0]);
	rhs[1] = x_tmp[0] * (RHO - x_tmp[2]) - x_tmp[1];
	rhs[2] = x_tmp[0] * x_tmp[1] - BETA * x_tmp[2];

	// Create the jacobian
	double* jac = (double* )malloc(sizeof(double) * n * numLEs);
	jac[0] = -SIGMA;
	jac[1] = SIGMA;
	jac[2] = 0;
	jac[3] = RHO - x_tmp[2];
	jac[4] = -1;
	jac[5] = -x_tmp[0];
	jac[6] = x_tmp[1];
	jac[7] = x_tmp[0];
	jac[8] = -BETA;

	// Call matrix matrix multiplication - C = alpha*A*B + beta*C => rhs_ext = alpha*jac_tmp*pert + 0.0*C
	// variables setup
	double alpha = 1.0;     // prefactor of A*B
	double beta  = 0.0;     // prefactor of C
	int M = n;          // no. of rows of A
	int N = numLEs;    // no. of cols of B
	int K = numLEs;    // no. of cols of A / rows of B
	int lda = numLEs;  // leading dim of A - length of elements between consecutive rows
	int ldb = numLEs;  // leading dim of B
	int ldc = numLEs;  // leading dim of C

	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, alpha, jac, lda, x_pert, ldb, beta, rhs_ext, ldc);

	// Free memory
	free(jac);
}



void orthonormalize(double* pert, double* R_tmp, int N, int numLEs) {


	// Initialize lapack vars
	lapack_int info;
	lapack_int m   = N;
	lapack_int n   = numLEs;
	lapack_int lda = numLEs;
	
	// Allocate temporary memory
	double* tau = (double* )malloc(sizeof(double) * numLEs);
	mem_chk(tau, "tau");	
	
	///---------------
	/// Perform QR Fac
	///---------------
	info = LAPACKE_dgeqrf(LAPACK_ROW_MAJOR, m, n, pert, lda, tau);
	if (info < 0 ) {
    	fprintf(stderr, "ERROR!! |in file \"%s\"-line:%d | LAPACKE Error - %d-th argument contains an illegal value;\n", __FILE__, __LINE__, info);
		exit( 1 );
    }

	// extract the diagonals of R
	for (int i = 0; i < N; ++i) {		
		for (int j = 0 ; j < numLEs; ++j) {
			if (j >= i) {
				R_tmp[i * numLEs + j] = pert[i * numLEs + j];
			}
			else {
				R_tmp[i * numLEs + j] = 0.0;
			}
		}
	}

	///---------------
	/// Form the Q matrix
	///---------------
    info = LAPACKE_dorgqr(LAPACK_ROW_MAJOR, m, m, n, pert, lda, tau);
    if (info < 0 ) {
    	fprintf(stderr, "ERROR!! |in file \"%s\"-line:%d | LAPACKE Error - %d-th argument contains an illegal value;\n", __FILE__, __LINE__, info);
		exit( 1 );
    }

  	// Free memory
	free(tau);
}

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
	double* angles = (double* )malloc(sizeof(double) * DOF * numLEs);	
	mem_chk(angles, "angles");
	memset(angles, 0.0, sizeof(double) * DOF * numLEs);
	int* pivot  = (int* )malloc(sizeof(double) * numLEs);
	mem_chk(pivot, "pivot");
	double* sum = (double* )malloc(sizeof(double) * numLEs);
	mem_chk(sum, "sum");

	// LAPACKE variables for dgesv function to perform solver on A*X = B
	lapack_int info;
	lapack_int lpk_m   = DOF;      // no. of systems - rows of A
	lapack_int lpk_n   = numLEs; // no. of rhs - cols of B
	lapack_int lpk_lda = DOF;      // leading dimension of A

	// CBLAS variables for dgemm function to perform C = alpha*A*B + beta*C
	double alpha = 1.0;     // prefactor of A*B
	double beta  = 0.0;     // prefactor of C
	int m = DOF;              // no. of rows of A
	int n = numLEs;         // no. of cols of B
	int k = numLEs;         // no. of cols of A / rows of B
	int lda = numLEs;       // leading dim of A - length of elements between consecutive rows
	int ldb = numLEs;       // leading dim of B
	int ldc = numLEs;       // leading dim of C

	// iterator for saving CLVs to file - start at end and decrement
	int save_clv_indx = (int) (m_rev_iters - m_rev_trans) / SAVE_CLV_STEP - 1;
	
	printf("save_indx: %d\n", save_clv_indx);
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

			printf("C[%d]: %5.12lf\t", i * numLEs + j, C[i * numLEs + j]);
		}
		printf("\n");
	}
	printf("\n\n");


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
		
		// Solve the system R_tmp*C_n = C_n-1 to iterate C matrix backwards in time
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
				write_hyperslab_data_d(file_space[1], data_set[1], mem_space[1], CLV, "CLV", DOF * numLEs, save_clv_indx);

				// compute the angles between the CLVs
				compute_angles(angles, CLV, DOF, numLEs);

				// // Write angles
				write_hyperslab_data_d(file_space[2], data_set[2], mem_space[2], angles, "Angles", DOF * numLEs, save_clv_indx);

				// Decrement for next iter
				save_clv_indx--;
			}
		}
		if (p == 0 || p == (m_rev_iters - m_rev_trans - 1)) {
			printf("CLV\n");
			for (int i = 0; i < DOF; ++i)
			{
				for (int j = 0; j < numLEs; ++j)
				{
					printf("CLV[%d]:\t%5.12lf\t", i * numLEs + j, CLV[i * numLEs + j]);
				}
				printf("\n");
			}
			printf("\n\n");

			printf("Angles\n");
			for (int i = 0; i < DOF; ++i)
			{
				for (int j = 0; j < numLEs; ++j)
				{
					printf("A[%d]:\t%5.12lf\t", i * numLEs + j, angles[i * numLEs + j]);
				}
				printf("\n");
			}
			printf("\n\n");
		}
	}
	//---------------------------------------
	// End of Ginelli Algo
	//---------------------------------------
	

	// Cleanup and free memory
	free(CLV);
	free(C);
	free(R_tmp);
	free(GS_tmp);
	free(angles);
	free(sum);
	free(pivot);
}



void compute_lce_spectrum(int N, int numLEs, int m_trans, int m_rev_trans, int m_end, int m_iter) {

	// ------------------------------
	//  Variable Definitions
	// ------------------------------
	// print update every x iterations
	int print_every = (m_end >= 10 ) ? (int)((double)m_end * 0.1) : 1;
	
	// Looping variables
	int tmp2;
	int tmp;
	int indx;

	// LCE variables
	double lce_sum;
	double dim_sum;
	int dim_indx;


	// ------------------------------
	//  Allocate memory
	// ------------------------------
	// Allocate mode related arrays
	double* x     = (double* ) malloc(sizeof(double) * N);
	mem_chk(x, "x");
	double* x_pert = (double* ) malloc(sizeof(double) * N * numLEs);
	mem_chk(x_pert, "x_pert");

	// LCE Spectrum Arrays
	double* znorm    = (double* )malloc(sizeof(double) * numLEs);	
	mem_chk(znorm, "znorm");
	double* lce      = (double* )malloc(sizeof(double) * numLEs);
	mem_chk(lce, "lce");
	double* run_sum  = (double* )malloc(sizeof(double) * numLEs);
	mem_chk(run_sum, "run_sum");
	memset(run_sum, 0.0, sizeof(double) * numLEs);

	// CLV arrays
	double* R_tmp = (double* )malloc(sizeof(double) * N * numLEs);	
	mem_chk(R_tmp, "R_tmp");
	double* R     = (double* )malloc(sizeof(double) * N * numLEs * (m_end - m_trans));	
	mem_chk(R, "R");
	double* GS    = (double* )malloc(sizeof(double) * N * numLEs * (m_end - 2 * m_trans));	
	mem_chk(GS, "GS");

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
	RK1 = (double* )malloc(sizeof(double) * N);
	RK2 = (double* )malloc(sizeof(double) * N);
	RK3 = (double* )malloc(sizeof(double) * N);
	RK4 = (double* )malloc(sizeof(double) * N);
	mem_chk(RK1, "RK1");
	mem_chk(RK2, "RK2");
	mem_chk(RK3, "RK3");
	mem_chk(RK4, "RK4");

	// Temp array for intermediate modes
	double* x_tmp = (double* )malloc(sizeof(double) * N);
	mem_chk(x_tmp, "x_tmp");

	// Memory for the four RHS evalutions for the perturbed system
	double* RK1_pert, *RK2_pert, *RK3_pert, *RK4_pert;
	RK1_pert = (double* )malloc(sizeof(double) * N * numLEs);
	RK2_pert = (double* )malloc(sizeof(double) * N * numLEs);
	RK3_pert = (double* )malloc(sizeof(double) * N * numLEs);
	RK4_pert = (double* )malloc(sizeof(double) * N * numLEs);
	mem_chk(RK1_pert, "RK1_pert");
	mem_chk(RK2_pert, "RK2_pert");
	mem_chk(RK3_pert, "RK3_pert");
	mem_chk(RK4_pert, "RK4_pert");

	double* x_pert_tmp = (double* ) malloc(sizeof(double) * N * numLEs);
	mem_chk(x_pert_tmp, "x_pert_tmp");


	// ------------------------------
	//  Get Initial Condition
	// ------------------------------
	// Set the initial condition of the perturb system to the identity matrix
	initial_conditions_lce(x_pert, x, N, numLEs);


	// ------------------------------
	//  Get Timestep & Integration Vars
	// ------------------------------
	// Get timestep
	double dt = 0.0005;

	// LCE algorithm varibales
	int trans_iters = m_trans * m_iter;

	// Get saving variables
	int tot_clv_save_steps = (int) (m_end - 2 * m_trans) / SAVE_CLV_STEP;
	int tot_m_save_steps = (int) (m_end) / SAVE_LCE_STEP;
	int tot_t_save_steps = (int) ((m_iter * m_end) / SAVE_DATA_STEP);
	if (m_trans == 0) {
		tot_t_save_steps += 1;
	}

	// Solver time varibales 
	double t0 = (m_trans - 1) * dt;
	if (m_trans == 0) {
		t0 = 0.0;
	}	
	double T  = t0 + m_iter * dt;	
	

	// ------------------------------
	//  HDF5 File Create
	// ------------------------------
	// Create the HDF5 file handle
	hid_t HDF_file_handle;

	// create hdf5 handle identifiers for hyperslabing the full evolution data
	hid_t HDF_file_space[4];
	hid_t HDF_data_set[4];
	hid_t HDF_mem_space[4];

	// get output file name
	char output_file_name[512];
	get_output_file_name(output_file_name, dt, m_iter, m_trans, m_end - m_trans);
	
	// open output file and create hyperslabbed datasets 
	open_output_create_slabbed_datasets_lce(&HDF_file_handle, output_file_name, HDF_file_space, HDF_data_set, HDF_mem_space, tot_t_save_steps, tot_m_save_steps, tot_clv_save_steps, N, numLEs);

	// Create arrays for time and phase order to save after algorithm is finished
	double* time_array = (double* )malloc(sizeof(double) * (tot_t_save_steps));
	mem_chk(time_array, "time_array");


	// ------------------------------
	//  Write Initial Conditions to File
	// ------------------------------
	if (m_trans == 0) {
		// write initial state
		write_hyperslab_data_d(HDF_file_space[0], HDF_data_set[0], HDF_mem_space[0], x, "x", N, 0);

		// Write initial time
		time_array[0] = 0.0;
	}


	// ------------------------------
	//  Begin Algorithm
	// ------------------------------
	double t = 0.0;
	int m    = 0;
	int iter = 1;
	int save_data_indx = 0;
	int save_lce_indx  = 0;
	if (m_trans == 0) {
		save_data_indx += 1;
	}
	while (m < m_end) {

		// ------------------------------
		//  Integrate System Forward
		// ------------------------------
		for (int p = 0; p < m_iter; ++p) {

			// Construct the modes
			for (int i = 0; i < N; ++i) {
				x_tmp[i] = x[i];
			}

			//////////////
			// STAGES
			//////////////
			/*---------- STAGE 1 ----------*/
			// find RHS first and then update stage
			rhs_extended(RK1, RK1_pert, x_tmp, x_pert, N, numLEs);
			for (int i = 0; i < N; ++i) {
				x_tmp[i] = x[i] + A21 * dt * RK1[i];
				tmp = i * (numLEs);
				for (int j = 0; j < (numLEs); ++j) {
					indx = tmp + j;
					x_pert_tmp[indx] = x_pert[indx] + A21 * dt * RK1_pert[indx];
				}
			}


			/*---------- STAGE 2 ----------*/
			// find RHS first and then update stage
			rhs_extended(RK2, RK2_pert, x_tmp, x_pert_tmp, N, numLEs);
			for (int i = 0; i < N; ++i) {
				x_tmp[i] = x[i] + A32 * dt * RK2[i];
				tmp = i * (numLEs);
				for (int j = 0; j < (numLEs); ++j) {
					indx = tmp + j;
					x_pert_tmp[indx] = x_pert[indx] + A21 * dt * RK2_pert[indx];
				}
			}
			

			/*---------- STAGE 3 ----------*/
			// find RHS first and then update stage
			rhs_extended(RK3, RK3_pert, x_tmp, x_pert_tmp, N, numLEs);
			for (int i = 0; i < N; ++i) {
				x_tmp[i] = x[i] + A43 * dt * RK3[i];
				tmp = i * (numLEs);
				for (int j = 0; j < (numLEs); ++j) {
					indx = tmp + j;
					x_pert_tmp[indx] = x_pert[indx] + A43 * dt * RK3_pert[indx];
				}
			}

			
			/*---------- STAGE 4 ----------*/
			// find RHS first and then update 
			rhs_extended(RK4, RK4_pert, x_tmp, x_pert_tmp, N, numLEs);

			
			//////////////
			// Update
			//////////////
			for (int i = 0; i < N; ++i) {
				x[i] = x[i] + (dt * B1) * RK1[i] + (dt * B2) * RK2[i] + (dt * B3) * RK3[i] + (dt * B4) * RK4[i];  
				tmp = i * (numLEs);
				for (int j = 0; j < (numLEs); ++j) {
					indx = tmp + j;
					x_pert[indx] = x_pert[indx] + (dt * B1) * RK1_pert[indx] + (dt * B2) * RK2_pert[indx] + (dt * B3) * RK3_pert[indx] + (dt * B4) * RK4_pert[indx];  
				}
			}		



			//////////////
			// Print to file
			//////////////
			if ((iter >= trans_iters) && (iter % SAVE_DATA_STEP == 0)) {
				// Write phases
				write_hyperslab_data_d(HDF_file_space[0], HDF_data_set[0], HDF_mem_space[0], x, "x", N, save_data_indx);

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
		orthonormalize(x_pert, R_tmp, N, numLEs);
		// ------------------------------
		//  Orthonormalize 
		// ------------------------------
		
		// ------------------------------
		//  Compute LCEs & Write To File
		// ------------------------------
		if (m >= m_trans) {

			// Record the GS vectors and R matrix and extract the diagonals of R
			tmp2 = (m - m_trans) * N * numLEs;
			for (int i = 0; i < N; ++i) {
				tmp = i * numLEs;		
				for (int j = 0 ; j < numLEs; ++j) {
					// Record upper triangular R matrix
					if (j >= i) {
						R[tmp2 + tmp + j] = R_tmp[tmp + j];

						// Record diagonals of R matrix (checking for sign correction)
						if (i == j) {
							znorm[i] = fabs(R_tmp[tmp + i]);
						} 
					}
					
					// Record the GS vectors
					if (m < (m_end - m_trans)) {
						GS[tmp2 + tmp + j] = x_pert[tmp + j];
					}
				}			
			}

			
			// Compute the LCEs for the current iteration
			for (int i = 0; i < numLEs; ++i) {
				// Compute LCE
				run_sum[i] = run_sum[i] + log(znorm[i]);
				lce[i]     = run_sum[i] / (t - t0);
			}

			// then write the current LCEs to this hyperslab
			if (m % SAVE_LCE_STEP == 0) {			
				write_hyperslab_data_d(HDF_file_space[3], HDF_data_set[3], HDF_mem_space[3], lce, "lce", numLEs, save_lce_indx);

				save_lce_indx += 1;
			}

			// Print update to screen
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
				printf("Iter: %d / %d | t: %5.6lf tsteps: %d | Sum: %5.9lf | Dim: %5.9lf\n", m, m_end, t, m_end * m_iter, lce_sum, (dim_indx + (dim_sum / fabs(lce[dim_indx]))));
				printf("k: \n");
				for (int j = 0; j < numLEs; ++j) {
					printf("%5.6lf ", lce[j]);
				}
				printf("\n\n");
			}
		}


		// ------------------------------
		//  Update For Next Iteration
		// ------------------------------
		T = T + m_iter * dt;
		m += 1;
		if (m - 1 == m_trans) {
			printf("\n\t!!Transient Iterations Complete!! - Iters: %d\n\n\n", iter - 1);
		}
	}
	// ------------------------------
	//  Compute the CLVs
	// ------------------------------
	compute_CLVs(HDF_file_space, HDF_data_set, HDF_mem_space, R, GS, N, numLEs, m_end - m_trans, m_trans);
	// ------------------------------
	// End Algorithm
	// ------------------------------
	

	// ------------------------------
	//  Write 1D Arrays Using HDF5Lite
	// ------------------------------
	hid_t D1 = 1;
	hid_t D1dims[D1];

	// Wtie time
	D1dims[0] = tot_t_save_steps;
	if ( (H5LTmake_dataset(HDF_file_handle, "Time", D1, D1dims, H5T_NATIVE_DOUBLE, time_array)) < 0) {
		printf("\n\n!!Failed to make - Time - Dataset!!\n\n");
	}



	printf("\n\n----------------------------------------------------------------\n\n");

	
	printf("Soln:\n");
	for (int i = 0; i < N; ++i)
	{
		printf("x[%d]:\t%5.16lf\n", i, x[i]);
	}
	printf("\n\n");

	printf("Pert:\n");
	for (int i = 0; i < N; ++i)
	{
		for (int j = 0; j < numLEs; ++j)
		{
			printf("x_pert[%d]:\t%5.16lf\t", i * numLEs + j, x_pert[i * numLEs + j]);
		}
		printf("\n");
	}
	printf("\n\n");

	// printf("\n\n----------------------------------------------------------------\n\n");

	// printf("R_0\n");
	// tmp2 = 0 * N * numLEs;
	// for (int i = 0; i < N; ++i) {
	// 	tmp = i * numLEs;		
	// 	for (int j = 0 ; j < numLEs; ++j) {
	// 		printf("R[%d]:\t%5.12lf\t", tmp2 + tmp + j, R[tmp2 + tmp + j]);
	// 	}
	// 	printf("\n");
	// }
	// printf("\n\n");

	// printf("R_end\n");
	// tmp2 = (m - m_trans - 1) * N * numLEs;
	// for (int i = 0; i < N; ++i) {
	// 	tmp = i * numLEs;		
	// 	for (int j = 0 ; j < numLEs; ++j) {
	// 		printf("R[%d]:\t%5.12lf\t", tmp2 + tmp + j, R[tmp2 + tmp + j]);
	// 	}
	// 	printf("\n");
	// }
	// printf("\n\n");


	// printf("GS_0\n");
	// tmp2 = 0 * N * numLEs;
	// for (int i = 0; i < N; ++i) {
	// 	tmp = i * numLEs;		
	// 	for (int j = 0 ; j < numLEs; ++j) {
	// 		printf("GS[%d]:\t%5.12lf\t", tmp2 + tmp + j, GS[tmp2 + tmp + j]);
	// 	}
	// 	printf("\n");
	// }
	// printf("\n\n");

	// printf("GS_end\n");
	// tmp2 = (m_end - 2 * m_trans - 1) * N * numLEs;
	// for (int i = 0; i < N; ++i) {
	// 	tmp = i * numLEs;		
	// 	for (int j = 0 ; j < numLEs; ++j) {
	// 		printf("GS[%d]:\t%5.12lf\t", tmp2 + tmp + j, GS[tmp2 + tmp + j]);
	// 	}
	// 	printf("\n");
	// }
	// printf("\n\n");


	

	// ------------------------------
	//  Clean Up & Exit
	// ------------------------------	
	// Free memory
	free(R);
	free(R_tmp);
	free(GS);
	free(x);
	free(x_tmp);
	free(x_pert);
	free(x_pert_tmp);
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
	

	// Close HDF5 handles
	H5Sclose( HDF_mem_space[1] );
	H5Dclose( HDF_data_set[1] );
	H5Sclose( HDF_file_space[1] );
	H5Sclose( HDF_mem_space[3] );
	H5Dclose( HDF_data_set[3] );
	H5Sclose( HDF_file_space[3] );
	H5Sclose( HDF_mem_space[0] );
	H5Dclose( HDF_data_set[0] );
	H5Sclose( HDF_file_space[0] );
	H5Sclose( HDF_mem_space[2] );
	H5Dclose( HDF_data_set[2] );
	H5Sclose( HDF_file_space[2] );


	// Close output file
	H5Fclose(HDF_file_handle);
}
// ---------------------------------------------------------------------
//  End of File
// ---------------------------------------------------------------------