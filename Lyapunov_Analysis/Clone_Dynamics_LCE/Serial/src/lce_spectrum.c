// Enda Carroll
// May 2020
// File including functions to compute the Lyapunov spectrum using 
// Jacobian free approach for a Phase Only model of the 1D Burgers 
// equation


// ---------------------------------------------------------------------
//  Standard Libraries and Headers
// ---------------------------------------------------------------------
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
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
#include "utils.h"
#include "solver.h"



// ---------------------------------------------------------------------
//  Function Definitions
// ---------------------------------------------------------------------
void pert_initial_condition(double* phi, double* amp, int* kx, int num_osc, int k0, double a, double b, double pert, int pert_dir) {

	// set the seed for the random number generator
	srand(123456789);

	// Spectrum cutoff
	double cutoff = ((double) num_osc - 1.0) / 2.0;

	for (int i = 0; i < num_osc; ++i) {

		// fill the wavenumbers array
		kx[i] = i;

		// fill amp and phi arrays
		if(i <= k0) {
			amp[i] = 0.0;
			phi[i] = 0.0;
		} else if(i == pert_dir){
			amp[i] = pow((double)i, -a) * exp(-b * pow((double) kx[i]/cutoff, 2) );
			phi[i] = (M_PI / 2.0) * (1.0 + 1e-10 * pow((double) i, 0.9)) + pert;	
			// phi[i] = M_PI*( (double) rand() / (double) RAND_MAX);	
		} else {
			amp[i] = pow((double)i, -a) * exp(-b * pow((double) kx[i]/cutoff, 2) );
			phi[i] = (M_PI / 2.0) * (1.0 + 1e-10 * pow((double) i, 0.9));	
			// phi[i] = M_PI*( (double) rand() / (double) RAND_MAX);	
		}
	}
}

double get_global_timestep(double a, double b, int n, int k0) {

	// Initialize vars
	double dt;
	double max_val;
	int num_osc = n / 2 + 1;

	double* amps = (double* ) malloc(num_osc * sizeof(double));
	int* kx = (int* ) malloc(num_osc * sizeof(int));
	for (int i = 0; i < num_osc; ++i)
	{
		kx[i] = i;
		if (i <= k0) {
			amps[i] = 0.0;
		} else {
			amps[i] = pow((double)i, -a) * exp(-b * pow((double) kx[i]/(n / 2.0), 2));
		}
	}

	// fftw arrays
	double* u_pad = (double* ) malloc((2 * n) * sizeof(double));
	fftw_complex* u_z_pad = (fftw_complex* ) fftw_malloc((n + 1) * sizeof(fftw_complex)); 

	// create fftw3 plans objects
	fftw_plan plan_r2c, plan_c2r;

	// create plans - ensure no overwriting - fill arrays after
	plan_r2c = fftw_plan_dft_r2c_1d(2 * n, u_pad, u_z_pad, FFTW_PRESERVE_INPUT); 
	plan_c2r = fftw_plan_dft_c2r_1d(2 * n, u_z_pad, u_pad, FFTW_PRESERVE_INPUT);


	// Initialize temp memory
	double* tmp_rhs;
	tmp_rhs = (double* ) fftw_malloc(num_osc*sizeof(double));

	fftw_complex* u_z_tmp;
	u_z_tmp = (fftw_complex* )fftw_malloc(num_osc*sizeof(fftw_complex));

	// Create modes for RHS evaluation
	for (int i = 0; i < num_osc; ++i) {
		if (i <= k0) {
			u_z_tmp[i] = 0.0 + 0.0*I;
		} else {
			u_z_tmp[i] = amps[i] * exp(I * 0.0);
		}
	}

	// Call the RHS
	po_rhs(tmp_rhs, u_z_tmp, &plan_c2r, &plan_r2c, kx, n, num_osc, k0);

	// Find  the fastest moving oscillator
	max(tmp_rhs, num_osc, k0, &max_val);

	// Get timestep
	dt = 1.0 / max_val;


	// destroy fftw plans
	fftw_destroy_plan(plan_r2c);
	fftw_destroy_plan(plan_c2r);

	free(tmp_rhs);
	free(kx);
	free(amps);
	free(u_pad);
	fftw_free(u_z_pad);
	fftw_free(u_z_tmp);	
	
	return dt;
}


void open_output_create_slabbed_datasets_lce(hid_t* file_handle, char* output_file_name, hid_t* file_space, hid_t* data_set, hid_t* mem_space, int num_t_steps, int num_m_steps, int num_osc, int k_range, int k1_range, int kmin) {

	// ------------------------------
	//  Create file
	// ------------------------------
	
	// create datafile - H5F_ACC_TRUNC overwrites file if it exists already
	*file_handle = H5Fcreate(output_file_name, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);


	// ------------------------------
	//  Create datasets with hyperslabing
	// ------------------------------
	//
	//---------- PHASES -----------//
	//
	// create hdf5 dimension arrays for creating the hyperslabs
	static const int dimensions = 2;
	hsize_t dims[dimensions];      // array to hold dims of full evolution data
	hsize_t maxdims[dimensions];   // array to hold max dims of full evolution data
	hsize_t chunkdims[dimensions]; // array to hold dims of the hyperslab chunks

	// initialize the hyperslab arrays
	dims[0]      = num_t_steps + 1;             // number of timesteps
	dims[1]      = num_osc;                 // number of oscillators
	maxdims[0]   = H5S_UNLIMITED;           // setting max time index to unlimited means we must chunk our data
	maxdims[1]   = num_osc;                 // same as before = number of modes
	chunkdims[0] = 1;                       // 1D chunk to be saved 
	chunkdims[1] = num_osc;                 // 1D chunk of size number of modes

	// create the 2D dataspace - setting the no. of dimensions, expected and max size of the dimensions
	file_space[0] = H5Screate_simple(dimensions, dims, maxdims);

	// must create a propertly list to enable data chunking due to max time dimension being unlimited
	// create property list 
	hid_t plist;
	plist = H5Pcreate(H5P_DATASET_CREATE);

	// using this property list set the chuncking - stores the chunking info in plist
	H5Pset_chunk(plist, dimensions, chunkdims);

	// Create the dataset in the previouosly created datafile - using the chunk enabled property list and new compound datatype
	data_set[0] = H5Dcreate(*file_handle, "Phases", H5T_NATIVE_DOUBLE, file_space[0], H5P_DEFAULT, plist, H5P_DEFAULT);
	
	// create the memory space for the slab
	dims[0] = 1;
	dims[1] = num_osc;

	// setting the max dims to NULL defaults to same size as dims
	mem_space[0] = H5Screate_simple(dimensions, dims, NULL);

	H5Pclose(plist);

	// #ifdef __TRIADS
	//
	//---------- TRIADS -----------//
	//
	// create hdf5 dimension arrays for creating the hyperslabs
	static const int dim2 = 2;
	hsize_t dims2[dim2];      // array to hold dims of full evolution data
	hsize_t maxdims2[dim2];   // array to hold max dims of full evolution data
	hsize_t chunkdims2[dim2]; // array to hold dims of the hyperslab chunks

	// initialize the hyperslab arrays
	dims2[0]      = num_t_steps + 1;             // number of timesteps + initial condition
	dims2[1]      = k_range * k1_range;      // size of triads array
	maxdims2[0]   = H5S_UNLIMITED;           // setting max time index to unlimited means we must chunk our data
	maxdims2[1]   = k_range * k1_range;      // size of triads array
	chunkdims2[0] = 1;                       // 1D chunk to be saved 
	chunkdims2[1] = k_range*k1_range;         // size of triad array

	// create the 2D dataspace - setting the no. of dimensions, expected and max size of the dimensions
	file_space[1] = H5Screate_simple(dim2, dims2, maxdims2);

	// must create a propertly list to enable data chunking due to max time dimension being unlimited
	// create property list 
	hid_t plist2;
	plist2 = H5Pcreate(H5P_DATASET_CREATE);

	// using this property list set the chuncking - stores the chunking info in plist
	H5Pset_chunk(plist2, dim2, chunkdims2);

	// Create the dataset in the previouosly created datafile - using the chunk enabled property list and new compound datatype
	data_set[1] = H5Dcreate(*file_handle, "Triads", H5T_NATIVE_DOUBLE, file_space[1], H5P_DEFAULT, plist2, H5P_DEFAULT);
	
	// create the memory space for the slab
	dims2[0] = 1;
	dims2[1] = k_range*k1_range;

	// setting the max dims to NULL defaults to same size as dims
	mem_space[1] = H5Screate_simple(dim2, dims2, NULL);

	// Create attribute data for the triad dimensions
	hid_t triads_attr, triads_attr_space;

	hsize_t adims[2];
	adims[0] = 1;
	adims[1] = 2;

	triads_attr_space = H5Screate_simple (2, adims, NULL);

	triads_attr = H5Acreate(data_set[1], "Triad_Dims", H5T_NATIVE_INT, triads_attr_space, H5P_DEFAULT, H5P_DEFAULT);

	int triads_dims[2];
	triads_dims[0] = k_range;
	triads_dims[1] = k1_range;

    herr_t status = H5Awrite(triads_attr, H5T_NATIVE_INT, triads_dims);

	// close the created property list
	status = H5Aclose(triads_attr);
    status = H5Sclose(triads_attr_space);
	status = H5Pclose(plist2);
	// #endif


	//---------- LCE -----------//
	//
	// initialize the hyperslab arrays
	dims[0]      = num_m_steps;             // number of timesteps
	dims[1]      = num_osc - kmin;          // number of oscillators
	maxdims[0]   = H5S_UNLIMITED;           // setting max time index to unlimited means we must chunk our data
	maxdims[1]   = num_osc - kmin;          // same as before = number of modes
	chunkdims[0] = 1;                       // 1D chunk to be saved 
	chunkdims[1] = num_osc - kmin;          // 1D chunk of size number of modes

	// create the 2D dataspace - setting the no. of dimensions, expected and max size of the dimensions
	file_space[2] = H5Screate_simple(dimensions, dims, maxdims);

	// must create a propertly list to enable data chunking due to max time dimension being unlimited
	// create property list 
	hid_t plist3;
	plist3 = H5Pcreate(H5P_DATASET_CREATE);

	// using this property list set the chuncking - stores the chunking info in plist
	H5Pset_chunk(plist3, dimensions, chunkdims);

	// Create the dataset in the previouosly created datafile - using the chunk enabled property list and new compound datatype
	data_set[2] = H5Dcreate(*file_handle, "LCE", H5T_NATIVE_DOUBLE, file_space[2], H5P_DEFAULT, plist3, H5P_DEFAULT);
	
	// create the memory space for the slab
	dims[0] = 1;
	dims[1] = num_osc - kmin;

	// setting the max dims to NULL defaults to same size as dims
	mem_space[2] = H5Screate_simple(dimensions, dims, NULL);

	H5Pclose(plist3);
}


void orthonormalize(double* pert, double* znorm, int num_osc, int kmin) {

	// Initialize vars
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
	double* tau        = (double* )malloc(sizeof(double) * kdim);	
	double* col_change =  (double* )malloc(sizeof(double) * (kdim) * (kdim));
	double* rhs_pert   =  (double* )malloc(sizeof(double) * (kdim) * (kdim));

	// Initialize col_change matrix
	for (int i = 0; i < kdim; ++i) {
		for (int j = 0; j < kdim; ++j) {
			col_change[i * (kdim) + j] = 0.0; 
			rhs_pert[i * (kdim) + j]   = pert[i * (kdim) + j];
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
			znorm[i] = -1.0 *rhs_pert[i * (kdim) + i];
		} else {
			col_change[i * (kdim) + i] = 1.0;
			znorm[i] = rhs_pert[i * (kdim) + i];
		}
	}


	///---------------
	/// Form the Q matrix
	///---------------
    info = LAPACKE_dorgqr(LAPACK_ROW_MAJOR, m, m, n, rhs_pert, lda, tau);


   	// Correct the orientation of the Q matrix columns 
	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, K, alpha, rhs_pert, lda, col_change, ldb, beta, pert, ldc);
	

	free(tau);
	free(col_change);
	free(rhs_pert);

}

double* pert_solver(double* phi_base, double* pertMat, double* amp, int* kx, fftw_plan fftw_plan_c2r, fftw_plan fftw_plan_r2c, int N, int k0,  int iters, int m, double pert, int pert_dir) {

	// ------------------------------
	//  Variable Definitions
	// ------------------------------
	// Number of modes
	int num_osc = (N / 2) + 1; 

	// padded array size
	int M = 2 * N;

	// Forcing wavenumber
	int kmin = k0 + 1;
	int kmax = num_osc - 1;
	
	
	// ------------------------------
	//  Allocate memory
	// ------------------------------
	// Oscillator arrays
	double* phi = (double* ) malloc(sizeof(double) * num_osc);

	// // modes array
	fftw_complex* u_z = (fftw_complex* ) fftw_malloc(num_osc * sizeof(fftw_complex));
		

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

	// temporary memory to store stages
	fftw_complex* u_z_tmp = (fftw_complex* )fftw_malloc(num_osc*sizeof(fftw_complex));



	// ------------------------------
	//  Generate Initial Conditions
	// ------------------------------
	// Copy memory
	memcpy(phi, phi_base, sizeof(double) * num_osc); 
	if (m == 1) {
		phi[pert_dir] += pert;
	} else {
		for (int i = kmin; i < num_osc; ++i) {
			phi[i] =  phi_base[i] + pert * pertMat[(i - kmin) * (num_osc - kmin) + (pert_dir - kmin)];
		}
	}

	// for (int i = 0; i < num_osc; ++i) {
	// 	printf("phi[%d]: %5.16lf \n", i, phi[i]);
	// }
	// printf("\n\n");
		

	// ------------------------------
	//  Get Timestep & Time variables
	// ------------------------------
	double dt = get_timestep(amp, fftw_plan_c2r, fftw_plan_r2c, kx, N, num_osc, k0);	

	// time varibales
	int ntsteps   = iters; 
	double t0     = 0.0;
	double T      = t0 + ntsteps * dt;
	
		
	// ------------------------------
	//  Integration & Looping Params
	// ------------------------------
	int iter = 1;
	double t = 0.0;	

	// ------------------------------
	//  Begin Integration
	// ------------------------------
	while (t < T) {

		// Construct the modes
		for (int i = 0; i < num_osc; ++i) {
			u_z_tmp[i] = amp[i] * cexp(I * phi[i]);
		}


		//////////////
		// STAGES
		//////////////
		/*---------- STAGE 1 ----------*/
		// find RHS first and then update stage
		po_rhs(RK1, u_z_tmp, &fftw_plan_c2r, &fftw_plan_r2c, kx, N, num_osc, k0);
		for (int i = 0; i < num_osc; ++i) {
			u_z_tmp[i] = amp[i] * cexp(I * (phi[i] + A21 * dt * RK1[i]));
		}
		

		/*---------- STAGE 2 ----------*/
		// find RHS first and then update stage
		po_rhs(RK2, u_z_tmp, &fftw_plan_c2r, &fftw_plan_r2c, kx, N, num_osc, k0);
		for (int i = 0; i < num_osc; ++i) {
			u_z_tmp[i] = amp[i] * cexp(I * (phi[i] + A32 * dt * RK2[i]));
		}

		/*---------- STAGE 3 ----------*/
		// find RHS first and then update stage
		po_rhs(RK3, u_z_tmp, &fftw_plan_c2r, &fftw_plan_r2c, kx, N, num_osc, k0);
		for (int i = 0; i < num_osc; ++i) {
			u_z_tmp[i] = amp[i] * cexp(I * (phi[i] + A43 * dt * RK3[i]));
		}


		/*---------- STAGE 4 ----------*/
		// find RHS first and then update 
		po_rhs(RK4, u_z_tmp, &fftw_plan_c2r, &fftw_plan_r2c, kx, N, num_osc, k0);


		//////////////
		// Update
		//////////////
		for (int i = 0; i < num_osc; ++i) {
			phi[i] = phi[i] + (dt * B1) * RK1[i] + (dt * B2) * RK2[i] + (dt * B3) * RK3[i] + (dt * B4) * RK4[i];  
		}
		
		
		// increment
		t   = iter*dt;
		iter++;
	}

	// free memory
	fftw_free(u_z);
	fftw_free(RK1);
	fftw_free(RK2);
	fftw_free(RK3);
	fftw_free(RK4);
	fftw_free(u_z_tmp);


	return phi;
}


void compute_spectrum(int N, int k0, double a, double b, int m_end, int m_iter, double pert) {

	// ------------------------------
	//  Variable Definitions
	// ------------------------------
	// Number of modes
	int num_osc = (N / 2) + 1; 

	// padded array size
	int M = 2 * N;

	// Forcing wavenumber
	int kmin = k0 + 1;
	int kmax = num_osc - 1;

	// Perturbed direction
	int pert_dir;

	// print update every x iterations
	int print_every = (m_end >= 10 ) ? (int)((double)m_end * 0.1) : 10;


	int save_step = 1;

	int index;
	int temp;
	
	
	// ------------------------------
	//  Allocate memory
	// ------------------------------
	// wavenumbers
	int* kx = (int* ) malloc(num_osc * sizeof(int));

	// Oscillator arrays
	double* amp = (double* ) malloc(num_osc * sizeof(double));
	double* phi = (double* ) malloc(num_osc * sizeof(double));

	// // modes array
	fftw_complex* u_z = (fftw_complex* ) fftw_malloc(num_osc * sizeof(fftw_complex));

	// padded solution arrays
	double* u_pad = (double* ) malloc(M * sizeof(double));
	fftw_complex* u_z_pad = (fftw_complex* ) fftw_malloc((2 * num_osc - 1) * sizeof(fftw_complex));

	// Triad phases array
	int k_range  = kmax - kmin + 1;
	int k1_range = (int)((kmax - kmin + 1)/ 2.0);

	double* triads = (double* )malloc(k_range * k1_range * sizeof(double));
	// initialize triad array to handle empty elements
	for (int i = 0; i < k_range; ++i) {
		int tmp = i * k1_range;
		for (int j = 0; j < k1_range; ++j) {
			int indx = tmp + j;
			triads[indx] = -10.0;
		}
	}	

	fftw_complex triad_phase_order;
	triad_phase_order = 0.0 + I * 0.0;


	// ------------------------------
	// Runge-Kutta Variables / Arrays
	// ------------------------------
	// Define RK4 variables
	static double C2 = 0.5, A21 = 0.5, \
				  C3 = 0.5,           A32 = 0.5, \
				  C4 = 1.0,                      A43 = 1.0, \
				            B1 = 1.0/6.0, B2 = 1.0/3.0, B3 = 1.0/3.0, B4 = 1.0/6.0; 

	// Memory fot the four RHS evaluations in the stages 
	double* RK1 = (double* )fftw_malloc(num_osc*sizeof(double));
	double* RK2 = (double* )fftw_malloc(num_osc*sizeof(double));
	double* RK3 = (double* )fftw_malloc(num_osc*sizeof(double));
	double* RK4 = (double* )fftw_malloc(num_osc*sizeof(double));

	// temporary memory to store stages
	fftw_complex* u_z_tmp = (fftw_complex* )fftw_malloc(num_osc*sizeof(fftw_complex));


	// ------------------------------
	// LCE Arrays
	// ------------------------------
	// Perturbed trajectory arrays
	double *phi_base   = (double* )malloc(sizeof(double) * num_osc);
	double *phi_tmp    = (double* )malloc(sizeof(double) * num_osc);
	double *phi_tracjs = (double* )malloc(sizeof(double) * num_osc * (num_osc - kmin)); 

	// LCE spectrum arrays
	double *pertMat  = (double* )malloc(sizeof(double) * (num_osc - kmin) * (num_osc - kmin));
	double *znorm    = (double* )malloc(sizeof(double) * (num_osc - kmin));
	double* lce      = (double* )malloc(sizeof(double) * (num_osc - kmin));
	double* run_sum  = (double* )malloc(sizeof(double) * (num_osc - kmin));	

	

	// ------------------------------
	//  Create FFTW plans
	// ------------------------------
	// create fftw3 plans objects
	fftw_plan fftw_plan_r2c, fftw_plan_c2r;

	// create plans - ensure no overwriting - fill arrays after
	fftw_plan_r2c = fftw_plan_dft_r2c_1d(M, u_pad, u_z_pad, FFTW_PRESERVE_INPUT); 
	fftw_plan_c2r = fftw_plan_dft_c2r_1d(M, u_z_pad, u_pad, FFTW_PRESERVE_INPUT);



	// ------------------------------
	//  Generate Initial Conditions
	// ------------------------------
	initial_condition(phi_base, amp, kx, num_osc, k0, a, b);
	
	if (N <= 32) {
		for (int i = 0; i < num_osc; ++i) {
			phi[i] = phi_base[i];
			printf("k[%d]: %d | a: %5.15lf   p: %5.16lf   | u_z[%d]: %5.16lf  %5.16lfI \n", i, kx[i], amp[i], phi[i], i, creal(amp[i] * cexp(I * phi[i])), cimag(amp[i] * cexp(I * phi[i])));
		}
		printf("\n");
	}
	
	

	// ------------------------------
	//  Get Timestep & Time variables
	// ------------------------------
	// Timestep
	double dt = get_timestep(amp, fftw_plan_c2r, fftw_plan_r2c, kx, N, num_osc, k0);

	// time varibales
	double t0     = 0.0;
	double T      = t0 + m_iter * dt;

	// LCE Algorithm variables
	int m = 1;

	// Saving variables
	int tot_m_save_steps = (int) m_end / save_step;
	int tot_t_save_steps = (int) ((m_iter * m_end) / save_step);



	// // ------------------------------
	// //  HDF5 File Create
	// // ------------------------------
	// // Create the HDF5 file handle
	// hid_t HDF_file_handle;


	// // create hdf5 handle identifiers for hyperslabing the full evolution data
	// hid_t HDF_file_space[4];
	// hid_t HDF_data_set[4];
	// hid_t HDF_mem_space[4];

	// // // define filename - const because it doesnt change
	// char output_file_name[128] = "../../Data/Output/LCE/LCE_Runtime_Data";
	// char output_file_data[128];

	// // form the filename of the output file
	// sprintf(output_file_data,  "_N[%d]_k0[%d]_ALPHA[%1.3lf]_BETA[%1.3lf]_u0[%s]_ITERS[%d].h5", N, k0, a, b, "ALIGNED", m_iter * m_end);
	// strcat(output_file_name, output_file_data);
	
	// // Print file name to screen
	// printf("\nOutput File: %s \n\n", output_file_name);


	// // open output file and create hyperslabbed datasets 
	// open_output_create_slabbed_datasets_lce(&HDF_file_handle, output_file_name, HDF_file_space, HDF_data_set, HDF_mem_space, tot_t_save_steps, tot_m_save_steps, num_osc, k_range, k1_range, kmin);


	// // Create arrays for time and phase order to save after algorithm is finished
	// double* time_array      = (double* )malloc(sizeof(double) * (tot_t_save_steps + 1));
	// double* phase_order_R   = (double* )malloc(sizeof(double) * (tot_t_save_steps + 1));
	// double* phase_order_Phi = (double* )malloc(sizeof(double) * (tot_t_save_steps + 1));

	// // ------------------------------
	// //  Write Initial Conditions to File
	// // ------------------------------
	// write_hyperslab_data_d(HDF_file_space[0], HDF_data_set[0], HDF_mem_space[0], phi, "phi", num_osc, 0);


	// // #ifdef __TRIADS
	// // compute triads for initial conditions
	// triad_phases(triads, &triad_phase_order, phi, kmin, kmax);
	
	// // then write the current modes to this hyperslab
	// write_hyperslab_data_d(HDF_file_space[1], HDF_data_set[1], HDF_mem_space[1], triads, "triads", k_range * k1_range, 0);
	
	// phase_order_R[0]   = cabs(triad_phase_order);
	// phase_order_Phi[0] = carg(triad_phase_order);
	// // #endif

	// // Write initial time
	// // #ifdef TRANSIENT
	// // time_array[0] = trans_iters * dt;
	// // #else
	// time_array[0] = 0.0;
	// // #endif

	
	// ------------------------------
	//  Integration & Looping Params
	// ------------------------------
	double t = 0.0;
	int iter = 1;
	int save_data_indx = 1;
	int save_lce_indx  = 1;

	// ------------------------------
	//  Begin Integration
	// ------------------------------
	while (m <= m_end) {

		// ------------------------------
		//  Integrate Base System Forward
		// ------------------------------
		for (int p = 0; p < m_iter; ++p) {

			// Construct the modes
			for (int i = 0; i < num_osc; ++i) {
				u_z_tmp[i] = amp[i] * cexp(I * phi[i]);
			}

			// // Print Update - Energy and Enstrophy
			// if (iter % (int)((m_iter * m_end) * 0.1) == 0) {
			// 	printf("Iter: %d/%d | t = %4.4lf |\n", iter, (m_iter * m_end), t);
			// }		


			//////////////
			// STAGES
			//////////////
			/*---------- STAGE 1 ----------*/
			// find RHS first and then update stage
			po_rhs(RK1, u_z_tmp, &fftw_plan_c2r, &fftw_plan_r2c, kx, N, num_osc, k0);
			for (int i = 0; i < num_osc; ++i) {
				u_z_tmp[i] = amp[i] * cexp(I * (phi[i] + A21 * dt * RK1[i]));
			}

			/*---------- STAGE 2 ----------*/
			// find RHS first and then update stage
			po_rhs(RK2, u_z_tmp, &fftw_plan_c2r, &fftw_plan_r2c, kx, N, num_osc, k0);
			for (int i = 0; i < num_osc; ++i) {
				u_z_tmp[i] = amp[i] * cexp(I * (phi[i] + A32 * dt * RK2[i]));
			}

			/*---------- STAGE 3 ----------*/
			// find RHS first and then update stage
			po_rhs(RK3, u_z_tmp, &fftw_plan_c2r, &fftw_plan_r2c, kx, N, num_osc, k0);
			for (int i = 0; i < num_osc; ++i) {
				u_z_tmp[i] = amp[i] * cexp(I * (phi[i] + A43 * dt * RK3[i]));
			}


			/*---------- STAGE 4 ----------*/
			// find RHS first and then update 
			po_rhs(RK4, u_z_tmp, &fftw_plan_c2r, &fftw_plan_r2c, kx, N, num_osc, k0);


			//////////////
			// Update
			//////////////
			for (int i = 0; i < num_osc; ++i) {
				phi[i] = phi[i] + (dt * B1) * RK1[i] + (dt * B2) * RK2[i] + (dt * B3) * RK3[i] + (dt * B4) * RK4[i];  
			}

			//////////////
			// Print to file
			//////////////
			// if (iter % save_step == 0) {
			// 	// Write phases
			// 	write_hyperslab_data_d(HDF_file_space[0], HDF_data_set[0], HDF_mem_space[0], phi, "phi", num_osc, save_data_indx);

			// 	// compute triads for initial conditions
			// 	triad_phases(triads, &triad_phase_order, phi, kmin, kmax);
				
			// 	// write triads
			// 	write_hyperslab_data_d(HDF_file_space[1], HDF_data_set[1], HDF_mem_space[1], triads, "triads", k_range * k1_range, save_data_indx);

			// 	// save time and phase order parameter
			// 	time_array[save_data_indx]      = iter * dt;
			// 	phase_order_R[save_data_indx]   = cabs(triad_phase_order);
			// 	phase_order_Phi[save_data_indx] = carg(triad_phase_order);


			// 	// increment indx for next iteration
			// 	save_data_indx++;
			// }
					
			// increment
			t   = iter*dt;
			iter++;
		}
		

		// ------------------------------
		//  Integrate Perturbed Trajectories
		// ------------------------------
		for (int j = 0; j < num_osc - kmin; ++j) {
			// Set the pertubation direction
			pert_dir = k0 + 1 + j;

			// Call solver for current pertubed trajectory
			phi_tmp = pert_solver(phi_base, pertMat, amp, kx, fftw_plan_c2r, fftw_plan_r2c, N, k0, m_iter, m, pert, pert_dir);

			// Save Output
			for (int i = 0; i < num_osc; ++i)	{
				index = i * (num_osc - kmin) + j;

				phi_tracjs[index] = phi_tmp[i];

				// printf("phi_tmp[%d]: %5.16lf\t", index, phi_tmp[i]);

				// Create Perturbation Matrix
				if (i > k0) {
					pertMat[(i - kmin) * (num_osc - kmin) + j] = phi[i] - phi_tmp[i];
				}
			}
			// printf("\n");
		}
		// printf("\n\n");


		// for (int i = 0; i < num_osc - kmin; ++i)
		// {
		// 	for (int j = 0; j < num_osc - kmin; ++j)
		// 	{
		// 		printf("pertM[%d]: %5.16lf \t", i*(num_osc - kmin) + j, pertMat[i*(num_osc - kmin) + j]);
		// 	}
		// 	printf("\n");
		// }
		// printf("\n\n");


		// ------------------------------
		//  Orthonormalize 
		// ------------------------------
		orthonormalize(pertMat, znorm, num_osc, k0 + 1);


		// for (int i = 0; i < num_osc - kmin; ++i)
		// {
		// 	for (int j = 0; j < num_osc - kmin; ++j)
		// 	{
		// 		printf("pertM[%d]: %5.16lf \t", i*(num_osc - kmin) + j, pertMat[i*(num_osc - kmin) + j]);
		// 	}
		// 	printf("\n");
		// }
		// printf("\n\n");


		// for (int i = 0; i < num_osc; ++i)
		// {
		// 	for (int j = 0; j < num_osc - kmin; ++j)
		// 	{
		// 		if (i <= k0) {
		// 			printf("phi[%d]: %5.16lf \t", i*(num_osc - kmin) + j, 0.0);
		// 		}else {
		// 			printf("phi[%d]: %5.16lf \t", i*(num_osc - kmin) + j, phi[i] + pert * pertMat[(i - kmin)*(num_osc - kmin) + (j)]);
		// 		}
		// 	}
		// 	printf("\n");
		// }
		// printf("\n\n");


		// for (int i = 0; i < num_osc; ++i)
		// {
		// 	printf("phi[%d]: %5.16lf \n", i, phi[i]);
		// }
		
		// ------------------------------
		//  Compute LCEs & Write To File
		// ------------------------------
		for (int i = 0; i < num_osc - kmin; ++i) {
			// Compute LCE
			run_sum[i] = run_sum[i] + log(znorm[i] / pert);
			lce[i]     = run_sum[i] / (t - t0);
		}

		// // then write the current LCEs to this hyperslab
		// if (m % SAVE_LCE_STEP == 0) {			
		// 	write_hyperslab_data_d(HDF_file_space[2], HDF_data_set[2], HDF_mem_space[2], lce, "lce", num_osc - kmin, save_lce_indx - 1);
		// }

		// Print update to screen
		if (m % print_every == 0) {
			double lce_sum = 0.0;
			double dim_sum = 0.0;
			int dim_indx   = 0;
			for (int i = 0; i < num_osc - kmin; ++i) {
				// Get spectrum sum
				lce_sum += lce[i];

				// Compute attractor dim
				if (dim_sum + lce[i] > 0) {
					dim_sum += lce[i];
					dim_indx += 1;
				}
				else {
					continue;
				}
			}
			printf("Iter: %d / %d | t: %5.6lf tsteps: %d | k0:%d alpha: %5.6lf beta: %5.6lf | Sum: %5.9lf | Dim: %5.9lf\n", m, m_end, t, m_end * m_iter, k0, a, b, lce_sum, (dim_indx + (dim_sum / fabs(lce[dim_indx]))));
			printf("k: \n");
			for (int j = 0; j < num_osc - kmin; ++j) {
				printf("%5.6lf ", lce[j]);
			}
			printf("\n\n");
		}


		// ------------------------------
		//  Update For Next Iteration
		// ------------------------------
		// Update Base trajectory 
		for (int i = 0; i < num_osc; ++i)
		{
			phi_base[i] = phi[i];
		}

		// Update Iterators
		T = T + m_iter * dt;
		m += 1;
	}
	// ------------------------------
	//  Write 1D Arrays Using HDF5Lite
	// ------------------------------
	// hid_t D2 = 2;
	// hid_t D2dims[D2];

	// // Write amplitudes
	// D2dims[0] = 1;
	// D2dims[1] = num_osc;
	// if ( (H5LTmake_dataset(HDF_file_handle, "Amps", D2, D2dims, H5T_NATIVE_DOUBLE, amp)) < 0){
	// 	printf("\n\n!!Failed to make - Amps - Dataset!!\n\n");
	// }
	
	// // Wtie time
	// D2dims[0] = tot_t_save_steps + 1;
	// D2dims[1] = 1;
	// if ( (H5LTmake_dataset(HDF_file_handle, "Time", D2, D2dims, H5T_NATIVE_DOUBLE, time_array)) < 0) {
	// 	printf("\n\n!!Failed to make - Time - Dataset!!\n\n");
	// }
	
	// // Write Phase Order R
	// D2dims[0] = tot_t_save_steps + 1;
	// D2dims[1] = 1;
	// if ( (H5LTmake_dataset(HDF_file_handle, "PhaseOrderR", D2, D2dims, H5T_NATIVE_DOUBLE, phase_order_R)) < 0) {
	// 	printf("\n\n!!Failed to make - PhaseOrderR - Dataset!!\n\n");
	// }
	// // Write Phase Order Phi
	// D2dims[0] = tot_t_save_steps + 1;
	// D2dims[1] = 1;
	// if ( (H5LTmake_dataset(HDF_file_handle, "PhaseOrderPhi", D2, D2dims, H5T_NATIVE_DOUBLE, phase_order_Phi)) < 0) {
	// 	printf("\n\n!!Failed to make - PhaseOrderPhi - Dataset!!\n\n");
	// }



	// ------------------------------
	//  Clean Up
	// ------------------------------
	// destroy fftw plans
	fftw_destroy_plan(fftw_plan_r2c);
	fftw_destroy_plan(fftw_plan_c2r);

	// free memory
	free(kx);
	free(u_pad);
	free(triads);
	// free(time_array);
	// free(phase_order_Phi);
	// free(phase_order_R);
	fftw_free(u_z);
	fftw_free(RK1);
	fftw_free(RK2);
	fftw_free(RK3);
	fftw_free(RK4);
	fftw_free(u_z_tmp);
	fftw_free(u_z_pad);

	// // close HDF5 handles
	// H5Sclose( HDF_mem_space[0] );
	// H5Dclose( HDF_data_set[0] );
	// H5Sclose( HDF_file_space[0] );
	// H5Sclose( HDF_mem_space[1] );
	// H5Dclose( HDF_data_set[1] );
	// H5Sclose( HDF_file_space[1] );

	// // Close pipeline to output file
	// H5Fclose(HDF_file_handle);
}
// ---------------------------------------------------------------------
//  End of File
// ---------------------------------------------------------------------