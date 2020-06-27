// Enda Carroll
// June 2020
// Main file for calling the post-processing functions for the solver output

// ---------------------------------------------------------------------
//  Standard Libraries and Headers
// ---------------------------------------------------------------------
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <unistd.h>
#include <time.h>
#include <string.h>
#include <complex.h>
#include <fftw3.h>
#include <hdf5.h>
#include <hdf5_hl.h>
#include <omp.h>

#include <gsl/gsl_cblas.h>
#include <gsl/gsl_histogram.h> /**< include the header file for GSL Library for 1D histograms */ 
#include <gsl/gsl_rstat.h>
#include <gsl/gsl_errno.h>
// ---------------------------------------------------------------------
//  User Libraries, Headers Global Definitions
// ---------------------------------------------------------------------
#include "data_types.h"
#include "stats.h"
#include "utils.h"
#include "phase_order.h"


runtime_data_struct*      run_data;
system_vars_struct*       sys_vars;
runstat_vars_struct*     stat_vars;
phaseorder_vars_struct* phase_vars;
HDF_file_info_struct*    file_info;
// ---------------------------------------------------------------------
//  Main function
// ---------------------------------------------------------------------
int main(int argc, char** argv) {

	// Start timer
	clock_t begin = clock();


	// ------------------------------
	//  Set Global Parameters 
	// ------------------------------
	// Create instances of global variables structs
	runtime_data_struct runtime_data;
	system_vars_struct   system_vars;
	runstat_vars_struct stats_vars;
	phaseorder_vars_struct phaseorder_vars;
	HDF_file_info_struct   HDF_file_info;

	// Now point the global struct pointers to these newly created global structs
	run_data   = &runtime_data;
	sys_vars   = &system_vars;
	stat_vars  = &stats_vars;
	phase_vars = &phaseorder_vars;
	file_info  = &HDF_file_info;


	// ------------------------------
	//  Get Input Data 
	// ------------------------------
	if (get_args(argc, argv) != 0) {
		fprintf(stderr, "\n Error: Error in reading in command line aguments, check utils.c file for details");
		exit(1);
	}

	// Read Input
	int N = sys_vars->N;
	int k0 = sys_vars->k0;
	double alpha = sys_vars->alpha;  
	double beta = sys_vars->beta ;    
	int iters = sys_vars->post_trans_iters ;      
	int trans_iters = sys_vars->trans_iters ;

	// Other variables
	int M = sys_vars->M ;      
	int num_osc = sys_vars->NUM_OSC  ;
	int kmin = sys_vars->kmin ;  
	int kmax = sys_vars->kmax ;   



	// ------------------------------
	//  Form Input filename 
	// ------------------------------ 
	char Input_file_name[128] = "../Data/Output/Stats/Runtime_Data";
	char Input_file_data[128];
	sprintf(Input_file_data,  "_N[%d]_k0[%d]_ALPHA[%1.3lf]_BETA[%1.3lf]_u0[%s]_ITERS[%d]_TRANS[%d].h5", N, k0, alpha, beta, "ALIGNED", iters, trans_iters);
	strcat(Input_file_name, Input_file_data);
	
	// Print file name to screen
	printf("\nInput File: %s \n\n", Input_file_name);



	// ------------------------------
	//  Read From Input File 
	// ------------------------------ 
	if(access(Input_file_name, F_OK) != 0) {
		fprintf(stderr, "\nERROR!! | in file:%s @line:%d | unable to find HDF5 input file [%s], check input filename and HDF5 parameters.\n", __FILE__, __LINE__, Input_file_name);		
		exit(1);					
	}

	hid_t HDF_file_handle = H5Fopen(Input_file_name, H5F_ACC_RDWR, H5P_DEFAULT) ;
	if (HDF_file_handle < 0) {
		fprintf(stderr, "\nERROR!! | in file:%s @line:%d | unable to open HDF5 input file [%s], check input parameters.\n", __FILE__, __LINE__, Input_file_name);
		exit(1);
	}


	/////////////////////////////////////////
	//-------------- VELOCITY ------------ //
	/////////////////////////////////////////
	// Check if data exists
	if( (H5Lexists(HDF_file_handle, "RealSpace", H5P_DEFAULT)) < 0 ) {
		fprintf(stderr, "\nERROR!! | in file:%s @line:%d | unable to find HDF dataset \"%s\" in input file [%s], check input filename and HDF5 parameters.\n", __FILE__, __LINE__, "RealSpace", Input_file_name);		
		exit(1);				
	}

	// Get Timesteps info
	hsize_t HDF_D2ndims = 2;
	hsize_t D2dims[HDF_D2ndims];

	if( (H5LTget_dataset_info(HDF_file_handle, "RealSpace", D2dims, NULL, NULL)) < 0) {
		fprintf(stderr, "\nERROR!! | in file:%s @line:%d | unable to get dataset info from dataset \"%s\" in input file [%s], check input filename and HDF5 parameters.\n", __FILE__, __LINE__, "RealSpace", Input_file_name);		
		exit(1);				
	}
	int num_tsteps = D2dims[0];
	if(D2dims[1] != N) {
		fprintf(stderr, "\nERROR!! | in file:%s @line:%d | Dimensions of Dataset \"%s\" in input file [%s] are incorrect, check file and parameters.\n", __FILE__, __LINE__, "RealSpace", Input_file_name);		
		exit(1);				
	}

	// Read In data
	double* u = malloc(sizeof(double) * num_tsteps * N);
	if( (H5LTread_dataset(HDF_file_handle, "RealSpace", H5T_NATIVE_DOUBLE, u)) < 0) {
		fprintf(stderr, "\nERROR!! | in file:%s @line:%d | unable to read in HDF dataset \"%s\" in input file [%s], check input filename and HDF5 parameters.\n", __FILE__, __LINE__, "RealSpace", Input_file_name);		
		exit(1);				
	}
	/////////////////////////////////////////
	//-------------- VELOCITY ------------ //
	/////////////////////////////////////////

	///////////////////////////////////////
	//-------------- TRIADS ------------ //
	///////////////////////////////////////
	// Check if data exists
	if( (H5Lexists(HDF_file_handle, "Triads", H5P_DEFAULT)) < 0 ) {
		fprintf(stderr, "\nERROR!! | in file:%s @line:%d | unable to find HDF dataset \"%s\" in input file [%s], check input filename and HDF5 parameters.\n", __FILE__, __LINE__, "Triads", Input_file_name);		
		exit(1);				
	}

	// Get attributes containing k1 and k3 ranges
	if((H5LTget_attribute_int(HDF_file_handle, "Triads", "Triad_Dims", D2dims)) < 0) {
		fprintf(stderr, "\nERROR!! | in file:%s @line:%d | unable to get attribute info from dataset \"%s\" in input file [%s], check input filename and HDF5 parameters.\n", __FILE__, __LINE__, "Triads", Input_file_name);		
		exit(1);				
	}
	int k3_range = D2dims[0];
	int k1_range = D2dims[1];
	printf("k3_range: %d\n", k3_range);
	printf("k1_range: %d\n", k1_range);
	
	if( (H5LTget_dataset_info(HDF_file_handle, "Triads", D2dims, NULL, NULL)) < 0) {
		fprintf(stderr, "\nERROR!! | in file:%s @line:%d | unable to get dataset info from dataset \"%s\" in input file [%s], check input filename and HDF5 parameters.\n", __FILE__, __LINE__, "Triads", Input_file_name);		
		exit(1);				
	}
	if(D2dims[1] != k3_range * k1_range) {
		fprintf(stderr, "\nERROR!! | in file:%s @line:%d | Dimensions of Dataset \"%s\" in input file [%s] are incorrect, check file and parameters.\n", __FILE__, __LINE__, "Triads", Input_file_name);		
		exit(1);				
	}

	// Read In data
	double* triads = malloc(sizeof(double) * num_tsteps * k3_range * k1_range);
	if( (H5LTread_dataset(HDF_file_handle, "Triads", H5T_NATIVE_DOUBLE, triads)) < 0) {
		fprintf(stderr, "\nERROR!! | in file:%s @line:%d | unable to read in HDF dataset \"%s\" in input file [%s], check input filename and HDF5 parameters.\n", __FILE__, __LINE__, "Triads", Input_file_name);		
		exit(1);				
	}
	///////////////////////////////////////
	//-------------- TRIADS ------------ //
	///////////////////////////////////////

	//////////////////////////////////////
	//-------------- MODES ------------ //
	//////////////////////////////////////
	// Check if data exists
	if( (H5Lexists(HDF_file_handle, "Modes", H5P_DEFAULT)) < 0 ) {
		fprintf(stderr, "\nERROR!! | in file:%s @line:%d | unable to find HDF dataset \"%s\" in input file [%s], check input filename and HDF5 parameters.\n", __FILE__, __LINE__, "Modes", Input_file_name);		
		exit(1);				
	}

	if( (H5LTget_dataset_info(HDF_file_handle, "Modes", D2dims, NULL, NULL)) < 0) {
		fprintf(stderr, "\nERROR!! | in file:%s @line:%d | unable to get dataset info from dataset \"%s\" in input file [%s], check input filename and HDF5 parameters.\n", __FILE__, __LINE__, "Modes", Input_file_name);		
		exit(1);				
	}
	if(D2dims[1] != num_osc) {
		fprintf(stderr, "\nERROR!! | in file:%s @line:%d | Dimensions of Dataset \"%s\" in input file [%s] are incorrect, check file and parameters.\n", __FILE__, __LINE__, "Modes", Input_file_name);		
		exit(1);				
	}

	// Read In data
	fftw_complex* u_z = malloc(sizeof(fftw_complex) * num_tsteps * num_osc);
	hid_t HDF_COMPLEX_DTYPE = create_complex_datatype();
	if( (H5LTread_dataset(HDF_file_handle, "Modes", HDF_COMPLEX_DTYPE, u_z)) < 0) {
		fprintf(stderr, "\nERROR!! | in file:%s @line:%d | unable to read in HDF dataset \"%s\" in input file [%s], check input filename and HDF5 parameters.\n", __FILE__, __LINE__, "Modes", Input_file_name);		
		exit(1);				
	}
	//////////////////////////////////////
	//-------------- MODES ------------ //
	//////////////////////////////////////
	///

	// ------------------------------
	//  STATS
	// ------------------------------ 
	// Histogram
	gsl_histogram* TriadPDF = gsl_histogram_alloc(NBINS_VELINC);

	double TriadPDF_Local_Lims[num_tsteps][2];
	double TriadPDF_Global_Lims[2];
	TriadPDF_Global_Lims[0] = -1e16;
	TriadPDF_Global_Lims[1] = +1e16;


	int r;
	const int IncLen = NUM_VEL_INC;
	const int Inc[NUM_VEL_INC] = {1, (kmax / 4), (kmax / 2), (3 * kmax / 4), kmax};
	for (int i = 0; i < IncLen; ++i) {
		printf("Inc[%d]: %d\n", i, Inc[i]);
	}

	// Create Velocity Increment Data Array
	double* du_r       = (double* )malloc(sizeof(double) * N);
	double* dudx       = (double* )malloc(sizeof(double) * num_tsteps * N);
	double* dudx_tmp   = (double* )malloc(sizeof(double) * N);

	fftw_complex* dudx_z_tmp = (fftw_complex* )malloc(sizeof(fftw_complex) * num_tsteps * num_osc);

	// fftw plans
	fftw_plan fftw_plan_c2r;
	fftw_plan_c2r = fftw_plan_dft_c2r_1d(N, dudx_z_tmp, dudx_tmp, FFTW_PRESERVE_INPUT);


	// Running Stats
	gsl_rstat_workspace* RunVelStats   = gsl_rstat_alloc();
	// gsl_rstat_workspace* RunTriadStats = gsl_rstat_alloc();
	
	
	double VelInc_Stats[num_tsteps][IncLen][5];


	int tmp_z;
	int tmp_r;
	int indx_z;
	int indx_r;
	// Loop through time
	for (int t = 0; t < num_tsteps; ++t) {
		tmp_z = t * num_osc;
		tmp_r = t * N;
		
		////////////////
		// Derivative //
		////////////////
		for (int k = 0; k < num_osc; ++k) {
			indx_z = tmp_z + k;

			dudx_z_tmp[k] = I * k * u_z[indx_z];
		}
		fftw_execute_dft_c2r(fftw_plan_c2r, dudx_z_tmp, dudx_tmp);
		for (int i = 0; i < N; ++i) {
			indx_r = tmp_z + i;

			dudx[indx_r] = dudx_tmp[i];
		}

		////////////////
		// Increments //
		////////////////
		for (int i = 0; i < IncLen; i+=2) {
			r = Inc[i];

			// reset running stat accumalator
			gsl_rstat_reset(RunVelStats);

			// Compute increments
			for (int j = 0; j < N; ++j)	{
				indx_r = tmp_r + j;

				// Compute Increment
				du_r[j] = u[(indx_r + r) % N] - u[indx_r];

				// Add data to running stats
				gsl_rstat_add(du_r[j], RunVelStats);

			}
			// Check Limits

			// Compute Stats
			VelInc_Stats[t][i][0] = gsl_rstat_mean(RunVelStats);
			VelInc_Stats[t][i][1] = gsl_rstat_variance(RunVelStats);
			VelInc_Stats[t][i][2] = gsl_rstat_rms(RunVelStats);
			VelInc_Stats[t][i][3] = gsl_rstat_skew(RunVelStats);
			VelInc_Stats[t][i][4] = gsl_rstat_kurtosis(RunVelStats);
		}


		////////////////
		//   Triads   //
		////////////////
		// double phase_val;
		// for (int k3 = kmin; k3 <= kmax; ++k3) {
		// 	tmp_r = (k3 - kmin) * (int) ((kmax - kmin + 1) / 2.0);
		// 	for (int k1 = kmin; k1 <= (int) (k3 / 2.0); ++k1)	{
		// 		indx_r = tmp + (k1 - kmin);

		// 		// real in val
		// 		phase_val = triads[indx_r];
				
		// 		// Check Local Limits
		// 		if 
		// 	}
		// }
	}
	// ------------------------------
	//  Write Arrays Using HDF5Lite
	// ------------------------------
	// Dimension Arrays
	hid_t HDF_D3ndims = 3;
	hid_t D3dims[HDF_D3ndims];

	// Write Velocity Increments Stats
	D3dims[0] = num_tsteps;
	D3dims[1] = IncLen;
	D3dims[2] = 5;
	if((H5Lexists(HDF_file_handle, "VelInc_Stats", H5P_DEFAULT)) <= 0 ) {
		printf("Writing ""%s""...\n", "VelInc_Stats");
		if((H5LTmake_dataset(HDF_file_handle, "VelInc_Stats", HDF_D3ndims, D3dims, H5T_NATIVE_DOUBLE, VelInc_Stats[0][0])) < 0 ) {
			fprintf(stderr, "\nERROR in file:%s @line:%d | unable to make dataset %s in HDF5 file %s\n", __FILE__, __LINE__, "VelIncs_Stats", Input_file_name);
			exit(1);		
		}
	} else {
		printf("\nDidn't Write \"%s\", dataset already exists!\n", "VelInc_Stats");
	}

	// Write Velocity Increments Stats
	D2dims[0] = 1;
	D2dims[1] = num_tsteps * N;
	if((H5Lexists(HDF_file_handle, "Derivative", H5P_DEFAULT)) <= 0 ) {
		printf("Writing ""%s""...\n", "Derivative");
		if((H5LTmake_dataset(HDF_file_handle, "Derivative", HDF_D2ndims, D2dims, H5T_NATIVE_DOUBLE, dudx)) < 0 ) {
			fprintf(stderr, "\nERROR!! | in file:%s @line:%d | unable to make dataset %s in HDF5 file %s\n", __FILE__, __LINE__, "Derivative", Input_file_name);
			exit(1);		
		}
	} else {
		printf("\nDidn't Write \"%s\", dataset already exists!\n", "Derivative");
	}


	// ------------------------------
	//  Clean Up 
	// ------------------------------ 
	// Free memory
	free(u);
	free(du_r);
	free(dudx);
	free(dudx_tmp);
	fftw_free(u_z);
	fftw_free(dudx_z_tmp);
	
	// Free GSL objects
	gsl_histogram_free(TriadPDF);
	gsl_rstat_free(RunVelStats);



	// Close pipeline to output file
	H5Fclose(HDF_file_handle);


	
	// Finish timing
	clock_t end = clock();

	// calculate execution time
	double time_spent = (double)(end - begin) / CLOCKS_PER_SEC;

	printf("\n\n\tTotal Execution Time: %20.16lf\n\n", time_spent);

	return 0;
}
// ---------------------------------------------------------------------
//  End of File
// ---------------------------------------------------------------------