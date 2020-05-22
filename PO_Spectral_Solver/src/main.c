 // Enda Carroll
// May 2020
// Main file for calling the pseudospectral solver for the 1D Burgers equation

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


// ---------------------------------------------------------------------
//  User Libraries and Headers
// ---------------------------------------------------------------------
// #include "po_data_types.h"
#include "utils.h"
#include "solver.h"





// ---------------------------------------------------------------------
//  Main function
// ---------------------------------------------------------------------
int main(int argc, char** argv) {

	clock_t begin = clock();

	// ------------------------------
	//  Variable Definitions
	// ------------------------------
	// Collocation points
	int N = atoi(argv[1]);

	// Number of modes
	int num_osc = (N / 2) + 1; 

	// padded array size
	int M = 2 * N;

	// Forcing wavenumber
	int k0 = 0;
	int kmin = k0 + 1;
	int kmax = num_osc - 1;
	
	// Spectrum vars
	double a = 1.0;
	double b = 0.0;
	double cutoff = ((double) num_osc - 1.0) / 2.0;
	
	// spatial variables
	double leftBoundary  = 0.0;
	double rightBoundary = 2.0*M_PI;
	double dx            = (rightBoundary - leftBoundary) / (double)N;


	// ------------------------------
	//  Allocate memory
	// ------------------------------
	// wavenumbers
	int* kx;
	kx = (int* ) malloc(num_osc * sizeof(int));

	// Oscillator arrays
	double* amp;
	amp = (double* ) malloc(num_osc * sizeof(double));
	double* phi;
	phi = (double* ) malloc(num_osc * sizeof(double));

	// // modes array
	fftw_complex* u_z;
	u_z	= (fftw_complex* ) fftw_malloc(num_osc * sizeof(fftw_complex));

	// padded solution arrays
	double* u_pad;
	u_pad = (double* ) malloc(M * sizeof(double));

	fftw_complex* u_z_pad;
	u_z_pad	= (fftw_complex* ) fftw_malloc(2 * num_osc * sizeof(fftw_complex));

	// Triad phases array
	int k_range  = kmax - kmin + 1;
	int k1_range = (int)((kmax - kmin + 1)/ 2.0);

	double* triads;
	triads = (double* )malloc(k_range * k1_range * sizeof(double));
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
	double* RK1, *RK2, *RK3, *RK4;
	RK1 = (double* )fftw_malloc(num_osc*sizeof(double));
	RK2 = (double* )fftw_malloc(num_osc*sizeof(double));
	RK3 = (double* )fftw_malloc(num_osc*sizeof(double));
	RK4 = (double* )fftw_malloc(num_osc*sizeof(double));

	// temporary memory to store stages
	fftw_complex* u_z_tmp;
	u_z_tmp = (fftw_complex* )fftw_malloc(num_osc*sizeof(fftw_complex));




	// ------------------------------
	//  Create FFTW plans
	// ------------------------------
	// create fftw3 plans objects
	fftw_plan fftw_plan_r2c, fftw_plan_c2r;

	// create plans - ensure no overwriting - fill arrays after
	fftw_plan_r2c = fftw_plan_dft_r2c_1d(M, u_pad, u_z_pad, FFTW_PRESERVE_INPUT); 
	fftw_plan_c2r = fftw_plan_dft_c2r_1d(M, u_z_pad, u_pad, FFTW_PRESERVE_INPUT);



	// ------------------------------
	//  Generate initial conditions
	// ------------------------------
	initial_condition(phi, amp, kx, num_osc, k0, cutoff, a, b);

	// for (int i = 0; i < num_osc; ++i) {
	// 	printf("k[%d]: %d | a: %5.15lf   p: %5.16lf   | u_z[%d]: %5.16lf  %5.16lfI \n", i, kx[i], amp[i], phi[i], i, creal(amp[i] * cexp(I * phi[i])), cimag(amp[i] * cexp(I * phi[i])));
	// }
	// printf("\n");



	// ------------------------------
	//  Get Timestep & Time variables
	// ------------------------------
	double dt;
	dt = get_timestep(amp, fftw_plan_c2r, fftw_plan_r2c, kx, N, num_osc, k0);

	// time varibales
	int iters     = 1e2;
	int save_step = 1;
	int ntsteps   = iters / save_step; 
	double t0     = 0.0;
	double T      = t0 + ntsteps * dt;


	// ------------------------------
	//  HDF5 File Create
	// ------------------------------
	// create hdf5 file identifier handle
	hid_t HDF_file_handle;

	// create hdf5 handle identifiers for hyperslabing the full evolution data
	hid_t HDF_file_space[2];
	hid_t HDF_data_set[2];
	hid_t HDF_mem_space[2];

	// define filename - const because it doesnt change
	char output_file_name[128] = "../Data/Output/Runtime_Data";
	char output_file_data[128];

	// form the filename of the output file
	sprintf(output_file_data,  "_N[%d]_k0[%d]_ALPHA[%1.3lf]_BETA[%1.3lf]_u0[%s]_ITERS[%d].h5", N, k0, a, b, "ALIGNED", ntsteps);
	strcat(output_file_name, output_file_data);
	
	// Print file name to screen
	printf("\nOutput File: %s \n\n", output_file_name);

	// open output file and create hyperslabbed datasets 
	open_output_create_slabbed_datasets(&HDF_file_handle, output_file_name, HDF_file_space, HDF_data_set, HDF_mem_space, ntsteps, num_osc, k_range, k1_range);


	// ------------------------------
	//  Write Initial Conditions to File
	// ------------------------------
	write_hyperslab_data_d(HDF_file_space[0], HDF_data_set[0], HDF_mem_space[0], phi, num_osc, 0);

	// compute triads for initial conditions
	triad_phases(triads, &triad_phase_order, phi, kmin, kmax);
	
	// // then write the current modes to this hyperslab
	write_hyperslab_data_d(HDF_file_space[1], HDF_data_set[1], HDF_mem_space[1], triads, k_range * k1_range, 0);

	

	// ------------------------------
	//  Integration & Looping Params
	// ------------------------------
	int iter = 1;
	double t = 0.0;	
	int save_data_indx = 1;

	// ------------------------------
	//  Begin Integration
	// ------------------------------
	while (t < T) {

		// Construct the modes
		for (int i = 0; i < num_osc; ++i) {
			u_z_tmp[i] = amp[i] * cexp(I * phi[i]);
		}

		// Print Update - Energy and Enstrophy
		if (iter % (int)(ntsteps * 0.1) == 0) {
			printf("Iter: %d/%d | t = %4.4lf | Energy = %4.8lf,   Enstrophy = %4.8lf\n", iter, ntsteps, t, system_energy(u_z, N), system_enstrophy(u_z, kx, N));
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

		// for (int i = 0; i < num_osc; ++i) {
		// 	printf("RHS[%d]: %5.16lf \n", i, RK1[i]);
		// }
		// printf("\n\n");

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
		if (iter % save_step == 0) {
			// Write phases
			write_hyperslab_data_d(HDF_file_space[0], HDF_data_set[0], HDF_mem_space[0], phi, num_osc, save_data_indx);

			// compute triads for initial conditions
			triad_phases(triads, &triad_phase_order, phi, kmin, kmax);
			
			// write triads
			write_hyperslab_data_d(HDF_file_space[1], HDF_data_set[1], HDF_mem_space[1], triads, k_range * k1_range, save_data_indx);

			// increment indx for next iteration
			save_data_indx++;
		}
				

		// increment
		t   = iter*dt;
		iter++;
	}


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
	fftw_free(u_z);
	fftw_free(RK1);
	fftw_free(RK2);
	fftw_free(RK3);
	fftw_free(RK4);
	fftw_free(u_z_tmp);
	fftw_free(u_z_pad);


	// close hdf5 datafile
	H5Sclose( HDF_mem_space[0] );
	H5Dclose( HDF_data_set[0] );
	H5Sclose( HDF_file_space[0] );
	H5Sclose( HDF_mem_space[1] );
	H5Dclose( HDF_data_set[1] );
	H5Sclose( HDF_file_space[1] );
	H5Fclose(HDF_file_handle);
	

	clock_t end = clock();

	// calculate execution time
	double time_spent = (double)(end - begin) / CLOCKS_PER_SEC;

	printf("Time: %20.16lf\n", time_spent);
	printf("\n\n");

	return 0;
}