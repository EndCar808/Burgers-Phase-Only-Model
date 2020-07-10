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
	//  Open Input filename 
	// ------------------------------
	open_input_file();


	// ------------------------------
	//  Read From Input File 
	// ------------------------------ 
	read_input_data();


	

	// ------------------------------
	//  STATS
	// ------------------------------ 
	#ifdef __STATS
	compute_stats();
	#endif


	// ------------------------------
	//  Write Arrays Using HDF5Lite
	// ------------------------------
	// // Dimension Arrays
	// hid_t HDF_D3ndims = 3;
	// hid_t D3dims[HDF_D3ndims];

	// // Write Velocity Increments Stats
	// D3dims[0] = num_tsteps;
	// D3dims[1] = IncLen;
	// D3dims[2] = 5;
	// if((H5Lexists(HDF_file_handle, "VelInc_Stats", H5P_DEFAULT)) <= 0 ) {
	// 	printf("Writing ""%s""...\n", "VelInc_Stats");
	// 	if((H5LTmake_dataset(HDF_file_handle, "VelInc_Stats", HDF_D3ndims, D3dims, H5T_NATIVE_DOUBLE, VelInc_Stats[0][0])) < 0 ) {
	// 		fprintf(stderr, "\nERROR in file:%s @line:%d | unable to make dataset %s in HDF5 file %s\n", __FILE__, __LINE__, "VelIncs_Stats", Input_file_name);
	// 		exit(1);		
	// 	}
	// } else {
	// 	printf("\nDidn't Write \"%s\", dataset already exists!\n", "VelInc_Stats");
	// }

	// // Write Velocity Increments Stats
	// D2dims[0] = 1;
	// D2dims[1] = num_tsteps * N;
	// if((H5Lexists(HDF_file_handle, "Derivative", H5P_DEFAULT)) <= 0 ) {
	// 	printf("Writing ""%s""...\n", "Derivative");
	// 	if((H5LTmake_dataset(HDF_file_handle, "Derivative", HDF_D2ndims, D2dims, H5T_NATIVE_DOUBLE, dudx)) < 0 ) {
	// 		fprintf(stderr, "\nERROR!! | in file:%s @line:%d | unable to make dataset %s in HDF5 file %s\n", __FILE__, __LINE__, "Derivative", Input_file_name);
	// 		exit(1);		
	// 	}
	// } else {
	// 	printf("\nDidn't Write \"%s\", dataset already exists!\n", "Derivative");
	// }


	// ------------------------------
	//  Clean Up 
	// ------------------------------ 
	// Free memory
	free(run_data->kx);

	#ifdef __STATS
	// free(run_data->u);
	// fftw_free(run_data->u_z);

	// free(stats_vars->du_r);
	// free(stats_vars->dudx);

	// gsl_rstat_free(stats_vars->RunVelStats);
	#endif
	
	#ifdef __PHASE_ORDER
	// gsl_histogram_free(TriadPDF);	
	
	free(run_data->amp);
	free(run_data->phi);
	#ifdef __TRIADS
	free(run_data->triads);
	#endif
	#endif




	// // Close pipeline to output file
	if((H5Fclose(file_info->input_file_handle)) < 0 ) {
		fprintf(stderr, "\nERROR!! | in file:%s @line:%d | unable to close input file [%s], check input filename and HDF5 parameters.\n", __FILE__, __LINE__, file_info->input_file_name);		
		exit(1);
	}


	
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