// Enda Carroll
// June 2020
// File including functions to perform post-processing stats


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
#include "data_types.h"
#include "utils.h"




// ---------------------------------------------------------------------
//  Function Definitions
// ---------------------------------------------------------------------
void open_stats_output_file() {

	// Make output file name
	char stats_file_data[512];
	sprintf(stats_file_data,  "/StatsData_ITERS[%d]_TRANS[%d].h5", sys_vars->post_trans_iters, sys_vars->trans_iters);
	strcpy(stats_vars->output_file_name, file_info->output_dir);
	strcat(stats_vars->output_file_name, stats_file_data);

	// Open output file
	stats_vars->output_file_handle = H5Fcreate(stats_vars->output_file_name, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
}

void write_output() {


	// Dimension Arrays
	hid_t HDF_D2ndims = 2;
	hid_t D2dims[HDF_D2ndims];
	hid_t HDF_D3ndims = 3;
	hid_t D3dims[HDF_D3ndims];

	// Write Velocity Increments Stats
	D3dims[0] = num_tsteps;
	D3dims[1] = IncLen;
	D3dims[2] = 5;
	if((H5Lexists(stats_vars->output_file_handle, "VelInc_Stats", H5P_DEFAULT)) <= 0 ) {
		printf("Writing ""%s""...\n", "VelInc_Stats");
		if((H5LTmake_dataset(stats_vars->output_file_handle, "VelInc_Stats", HDF_D3ndims, D3dims, H5T_NATIVE_DOUBLE, VelInc_Stats[0][0])) < 0 ) {
			fprintf(stderr, "\nERROR in file:%s @line:%d | unable to make dataset %s in HDF5 file %s\n", __FILE__, __LINE__, "VelIncs_Stats", stats_vars->output_file_name);
			exit(1);		
		}
	} else {
		printf("\nDidn't Write \"%s\", dataset already exists!\n", "VelInc_Stats");
	}

	// Write Derivative
	D2dims[0] = 1;
	D2dims[1] = num_tsteps * N;
	if((H5Lexists(stats_vars->output_file_handle, "Derivative", H5P_DEFAULT)) <= 0 ) {
		printf("Writing ""%s""...\n", "Derivative");
		if((H5LTmake_dataset(stats_vars->output_file_handle, "Derivative", HDF_D2ndims, D2dims, H5T_NATIVE_DOUBLE, stat_vars->dudx)) < 0 ) {
			fprintf(stderr, "\nERROR!! | in file:%s @line:%d | unable to make dataset %s in HDF5 file %s\n", __FILE__, __LINE__, "Derivative", stats_vars->output_file_name);
			exit(1);		
		}
	} else {
		printf("\nDidn't Write \"%s\", dataset already exists!\n", "Derivative");
	}

	// // Write VelInc
	// D2dims[0] = 1;
	// D2dims[1] = num_tsteps * N;
	// if((H5Lexists(stats_vars->output_file_handle, "VelocityIncrements", H5P_DEFAULT)) <= 0 ) {
	// 	printf("Writing ""%s""...\n", "VelocityIncrements");
	// 	if((H5LTmake_dataset(stats_vars->output_file_handle, "VelocityIncrements", HDF_D2ndims, D2dims, H5T_NATIVE_DOUBLE, stat_vars->du_r)) < 0 ) {
	// 		fprintf(stderr, "\nERROR!! | in file:%s @line:%d | unable to make dataset %s in HDF5 file %s\n", __FILE__, __LINE__, "VelocityIncrements", stats_vars->output_file_name);
	// 		exit(1);		
	// 	}
	// } else {
	// 	printf("\nDidn't Write \"%s\", dataset already exists!\n", "VelocityIncrements");
	// }

	// Write VelInc
	D2dims[0] = 1;
	D2dims[1] = num_tsteps * N;
	if((H5Lexists(stats_vars->output_file_handle, "VelIncSmall", H5P_DEFAULT)) <= 0 ) {
		printf("Writing ""%s""...\n", "VelIncSmall");
		if((H5LTmake_dataset(stats_vars->output_file_handle, "VelIncSmall", HDF_D2ndims, D2dims, H5T_NATIVE_DOUBLE, stat_vars->du_r_small)) < 0 ) {
			fprintf(stderr, "\nERROR!! | in file:%s @line:%d | unable to make dataset %s in HDF5 file %s\n", __FILE__, __LINE__, "VelIncSmall", stats_vars->output_file_name);
			exit(1);		
		}
	} else {
		printf("\nDidn't Write \"%s\", dataset already exists!\n", "VelIncSmall");
	}
	// Write VelInc
	D2dims[0] = 1;
	D2dims[1] = num_tsteps * N;
	if((H5Lexists(stats_vars->output_file_handle, "VelIncMid", H5P_DEFAULT)) <= 0 ) {
		printf("Writing ""%s""...\n", "VelIncMid");
		if((H5LTmake_dataset(stats_vars->output_file_handle, "VelIncMid", HDF_D2ndims, D2dims, H5T_NATIVE_DOUBLE, stat_vars->du_r_mid)) < 0 ) {
			fprintf(stderr, "\nERROR!! | in file:%s @line:%d | unable to make dataset %s in HDF5 file %s\n", __FILE__, __LINE__, "VelIncMid", stats_vars->output_file_name);
			exit(1);		
		}
	} else {
		printf("\nDidn't Write \"%s\", dataset already exists!\n", "VelIncMid");
	}
	// Write VelInc
	D2dims[0] = 1;
	D2dims[1] = num_tsteps * N;
	if((H5Lexists(stats_vars->output_file_handle, "VelIncLarge", H5P_DEFAULT)) <= 0 ) {
		printf("Writing ""%s""...\n", "VelIncLarge");
		if((H5LTmake_dataset(stats_vars->output_file_handle, "VelIncLarge", HDF_D2ndims, D2dims, H5T_NATIVE_DOUBLE, stat_vars->du_r_large)) < 0 ) {
			fprintf(stderr, "\nERROR!! | in file:%s @line:%d | unable to make dataset %s in HDF5 file %s\n", __FILE__, __LINE__, "VelIncLarge", stats_vars->output_file_name);
			exit(1);		
		}
	} else {
		printf("\nDidn't Write \"%s\", dataset already exists!\n", "VelIncLarge");
	}
}


void compute_stats() {

	// // Histogram
	// gsl_histogram* TriadPDF = gsl_histogram_alloc(NBINS_VELINC);

	// double TriadPDF_Local_Lims[num_tsteps][2];
	// double TriadPDF_Global_Lims[2];
	// TriadPDF_Global_Lims[0] = -1e16;
	// TriadPDF_Global_Lims[1] = +1e16;
	// gsl_rstat_workspace* RunTriadStats = gsl_rstat_alloc();
	// 

	// Initialize stats variables
	int N          = sys_vars->N;
	int NUM_OSC    = sys_vars->NUM_OSC;
	int num_tsteps = sys_vars->num_tsteps;
	int kmax       = sys_vars->kmax;

	// Initialize workspaces
	stat_vars->RunVelStats = gsl_rstat_alloc();	
	const int IncLen = NUM_VEL_INC;
	const int Inc[NUM_VEL_INC] = {1, (kmax / 4), (kmax / 2), (3 * kmax / 4), kmax};
	for (int i = 0; i < IncLen; ++i) {
		printf("Inc[%d]: %d\n", i, Inc[i]);
	}
	int r;

	// Create Velocity Increment Data Array
	stat_vars->du_r_small  = (double* )malloc(sizeof(double) * num_tsteps * N);
	stat_vars->du_r_mid    = (double* )malloc(sizeof(double) * num_tsteps * N);
	stat_vars->du_r_large  = (double* )malloc(sizeof(double) * num_tsteps * N);
	stat_vars->dudx  = (double* )malloc(sizeof(double) * num_tsteps * N);
	double* dudx_tmp = (double* )malloc(sizeof(double) * N);

	fftw_complex* dudx_z_tmp = (fftw_complex* )malloc(sizeof(fftw_complex) * num_tsteps * NUM_OSC);

	// fftw plans
	fftw_plan fftw_plan_c2r;
	fftw_plan_c2r = fftw_plan_dft_c2r_1d(N, dudx_z_tmp, dudx_tmp, FFTW_PRESERVE_INPUT);




	// Open output stats file
	open_stats_output_file();


	// // Initialize stats array	
	// stat_vars->VelInc_Stats[IncLen][5];


	int tmp_z;
	int tmp_r;	
	int indx_z;
	int indx_r;
	// Loop through time
	for (int t = 0; t < num_tsteps; ++t) {
		tmp_z = t * NUM_OSC;
		tmp_r = t * N;
		
		////////////////
		// Derivative //
		////////////////
		for (int k = 0; k < NUM_OSC; ++k) {
			indx_z = tmp_z + k;

			dudx_z_tmp[k] = I * k * run_data->u_z[indx_z];
		}
		fftw_execute_dft_c2r(fftw_plan_c2r, dudx_z_tmp, dudx_tmp);
		for (int i = 0; i < N; ++i) {
			indx_r = tmp_z + i;

			stat_vars->dudx[indx_r] = dudx_tmp[i];
		}

		////////////////
		// Increments //
		////////////////
		for (int i = 0; i < IncLen; i+=2) {
			r = Inc[i];

			// reset running stat accumalator
			gsl_rstat_reset(stat_vars->RunVelStats);

			// Compute increments
			for (int j = 0; j < N; ++j)	{
				indx_r = tmp_r + j;

				// Compute Increment
				stat_vars->du_r_small[indx_r] = run_data->u[(indx_r + Inc[0]) % N] - run_data->u[indx_r];
				stat_vars->du_r_mid[indx_r]   = run_data->u[(indx_r + Inc[2]) % N] - run_data->u[indx_r];
				stat_vars->du_r_large[indx_r] = run_data->u[(indx_r + Inc[4]) % N] - run_data->u[indx_r];


				// Add data to running stats
				gsl_rstat_add(stat_vars->du_r_small[indx_r], stat_vars->RunVelStats);

			}
			// Check Limits

			// // Compute Stats
			// VelInc_Stats[t][i][0] = gsl_rstat_mean(RunVelStats);
			// VelInc_Stats[t][i][1] = gsl_rstat_variance(RunVelStats);
			// VelInc_Stats[t][i][2] = gsl_rstat_rms(RunVelStats);
			// VelInc_Stats[t][i][3] = gsl_rstat_skew(RunVelStats);
			// VelInc_Stats[t][i][4] = gsl_rstat_kurtosis(RunVelStats);
		}
	}
	// ------------------------------
	//  Write Arrays Using HDF5Lite
	// ------------------------------



	// ------------------------------
	//  Clean Up 
	// ------------------------------ 
	// Free memory
	free(dudx_tmp);
	fftw_free(dudx_z_tmp);

	free(run_data->u);
	fftw_free(run_data->u_z);

	// free(stat_vars->du_r);
	free(stat_vars->du_r_small);
	free(stat_vars->du_r_mid);
	free(stat_vars->du_r_large);
	free(stat_vars->dudx);

	gsl_rstat_free(stat_vars->RunVelStats);

	// Destroy FFTW plan
	fftw_destroy_plan(fftw_plan_c2r);
}



// ---------------------------------------------------------------------
//  End of File
// ---------------------------------------------------------------------