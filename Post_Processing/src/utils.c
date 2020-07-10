// Enda Carroll
// June 2020
// Utility functions file for post-processing solver data


// ---------------------------------------------------------------------
//  Standard Libraries and Headers
// ---------------------------------------------------------------------
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <unistd.h>
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




// ---------------------------------------------------------------------
//  Function Definitions
// ---------------------------------------------------------------------
int get_args(int argc, char** argv) {

	if(argc != 8) {
		fprintf(stderr, "ERROR: Not enough input parameters specified.\nLooking for:\nN\nk0\nAlpha\nBeta\nSaving Iters\nTransient Iters\nInitial Condition\n");
		exit(1);
	} 
	
	// Read Input
	sys_vars->N  = atoi(argv[1]);
	sys_vars->k0 = atoi(argv[2]);
	sys_vars->alpha    = atof(argv[3]);
	sys_vars->beta     = atof(argv[4]);
	sys_vars->post_trans_iters = atoi(argv[5]);
	sys_vars->trans_iters      = atoi(argv[6]);
	strcpy(sys_vars->u0, argv[7]);          

	// Other variables
	sys_vars->M       = 2 * sys_vars->N;
	sys_vars->NUM_OSC = sys_vars->N / 2 + 1; 
	sys_vars->kmin    = sys_vars->k0 + 1;
	sys_vars->kmax    = sys_vars->NUM_OSC - 1;


	return 0;
}

void mem_chk (void *arr_ptr, char *name) {
  if (arr_ptr == NULL ) {
    fprintf(stderr, "ERROR!! |in file \"%s\"-line:%d | Unable to malloc required memory for %s, now exiting!\n", __FILE__, __LINE__, name);
    exit(1);
  }
}

hid_t create_complex_datatype() {
	
	// Initialize hdf5 type handle
	hid_t dtype;

	// Initialize new complex struct
	complex_type_struct cmplex;
	cmplex.re = 0.0;
	cmplex.im = 0.0;

	// create complex compound datatype
	dtype = H5Tcreate (H5T_COMPOUND, sizeof(cmplex));
	if (dtype < 0) {
		fprintf(stderr, "ERROR!! |in file \"%s\"-line:%d | Unable to create complex datatype, now exiting!\n", __FILE__, __LINE__);
    	exit(1);
	}

	// Add real and imaginary members to complex datatype
  	if ((H5Tinsert(dtype, "r", offsetof(complex_type_struct, re), H5T_NATIVE_DOUBLE)) < 0) {
		fprintf(stderr, "ERROR!! |in file \"%s\"-line:%d | Unable to add \"%s\" member to complex datatype, now exiting!\n", __FILE__, __LINE__, "r");
    	exit(1);
	}
  	if ((H5Tinsert(dtype, "i", offsetof(complex_type_struct, im), H5T_NATIVE_DOUBLE)) < 0) {
		fprintf(stderr, "ERROR!! |in file \"%s\"-line:%d | Unable to add \"%s\" member to complex datatype, now exiting!\n", __FILE__, __LINE__, "i");
    	exit(1);
	}

  	return dtype;
}

void open_input_file() {

	// Initialize Input filename and Output dir
	// strncpy((file_info->input_file_name), "../Data/Output/Stats/Runtime_Data", 512);
	// 
	sprintf(file_info->output_dir,  "../Data/RESULTS/RESULTS_N[%d]_k0[%d]_ALPHA[%1.3lf]_BETA[%1.3lf]_u0[%s]", sys_vars->N, sys_vars->k0, sys_vars->alpha, sys_vars->beta, sys_vars->u0);
	strcpy((file_info->input_file_name), (file_info->output_dir));
	
	// Form the input file path
	char Input_file_data[512];
	// sprintf(Input_file_data,  "_N[%d]_k0[%d]_ALPHA[%1.3lf]_BETA[%1.3lf]_u0[%s]_ITERS[%d]_TRANS[%d].h5", sys_vars->N, sys_vars->k0, sys_vars->alpha, sys_vars->beta, sys_vars->u0, sys_vars->post_trans_iters, sys_vars->trans_iters);
	// strcat(file_info->input_file_name, Input_file_data);
	sprintf(Input_file_data,  "/SolverData_ITERS[%d]_TRANS[%d].h5", sys_vars->post_trans_iters, sys_vars->trans_iters);
	strcat(file_info->input_file_name, Input_file_data);
	
	// Print file path to screen
	printf("\nInput File: %s \n\n", file_info->input_file_name);

	// Check if input file can be opened
	if(access(file_info->input_file_name, F_OK) != 0) {
		fprintf(stderr, "\nERROR!! | in file:%s @line:%d | unable to find HDF5 input file [%s], check input filename and HDF5 parameters.\n", __FILE__, __LINE__, file_info->input_file_name);		
		exit(1);					
	}

	// Open file
	file_info->input_file_handle = H5Fopen(file_info->input_file_name, H5F_ACC_RDWR, H5P_DEFAULT) ;
	if (file_info->input_file_handle < 0) {
		fprintf(stderr, "\nERROR!! | in file:%s @line:%d | unable to open HDF5 input file [%s], check input parameters.\n", __FILE__, __LINE__, file_info->input_file_name);
		exit(1);
	}


}

void read_input_data() {

	// Initialize variables
	hsize_t HDF_D2ndims = 2;
	hsize_t D2dims[HDF_D2ndims];

	#ifdef __STATS
	/////////////////////////////////////////
	//-------------- VELOCITY ------------ //
	/////////////////////////////////////////
	// Check if data exists
	if( (H5Lexists(file_info->input_file_handle, "RealSpace", H5P_DEFAULT)) < 0 ) {
		fprintf(stderr, "\nERROR!! | in file:%s @line:%d | unable to find HDF dataset \"%s\" in input file [%s], check input filename and HDF5 parameters.\n", __FILE__, __LINE__, "RealSpace", file_info->input_file_name);		
		exit(1);				
	}

	if( (H5LTget_dataset_info(file_info->input_file_handle, "RealSpace", D2dims, NULL, NULL)) < 0) {
		fprintf(stderr, "\nERROR!! | in file:%s @line:%d | unable to get dataset info from dataset \"%s\" in input file [%s], check input filename and HDF5 parameters.\n", __FILE__, __LINE__, "RealSpace", file_info->input_file_name);		
		exit(1);				
	}
	// Get the number of timestesp
	sys_vars->num_tsteps = D2dims[0];

	// Check if data dimensions match input dimensions
	if(D2dims[1] != sys_vars->N) {
		fprintf(stderr, "\nERROR!! | in file:%s @line:%d | Dimensions of Dataset \"%s\" in input file [%s] are incorrect, check file and parameters.\n", __FILE__, __LINE__, "RealSpace", file_info->input_file_name);		
		exit(1);				
	}

	// Read In data and store in modes array
	run_data->u = malloc(sizeof(double) * sys_vars->num_tsteps * sys_vars->N);
	if( (H5LTread_dataset(file_info->input_file_handle, "RealSpace", H5T_NATIVE_DOUBLE, run_data->u)) < 0) {
		fprintf(stderr, "\nERROR!! | in file:%s @line:%d | unable to read in HDF dataset \"%s\" in input file [%s], check input filename and HDF5 parameters.\n", __FILE__, __LINE__, "RealSpace", file_info->input_file_name);		
		exit(1);				
	}
	/////////////////////////////////////////
	//-------------- VELOCITY ------------ //
	/////////////////////////////////////////
	

	//////////////////////////////////////
	//-------------- MODES ------------ //
	//////////////////////////////////////
	// Check if data exists
	if( (H5Lexists(file_info->input_file_handle, "Modes", H5P_DEFAULT)) < 0 ) {
		fprintf(stderr, "\nERROR!! | in file:%s @line:%d | unable to find HDF dataset \"%s\" in input file [%s], check input filename and HDF5 parameters.\n", __FILE__, __LINE__, "Modes", file_info->input_file_name);		
		exit(1);				
	}

	if( (H5LTget_dataset_info(file_info->input_file_handle, "Modes", D2dims, NULL, NULL)) < 0) {
		fprintf(stderr, "\nERROR!! | in file:%s @line:%d | unable to get dataset info from dataset \"%s\" in input file [%s], check input filename and HDF5 parameters.\n", __FILE__, __LINE__, "Modes", file_info->input_file_name);		
		exit(1);				
	}

	// Check if data dimensions match input dimensions
	if(D2dims[1] != sys_vars->NUM_OSC) {
		fprintf(stderr, "\nERROR!! | in file:%s @line:%d | Dimensions of Dataset \"%s\" in input file [%s] are incorrect, check file and parameters.\n", __FILE__, __LINE__, "Modes", file_info->input_file_name);		
		exit(1);				
	}

	// Read In data
	run_data->u_z = malloc(sizeof(fftw_complex) * sys_vars->num_tsteps * sys_vars->NUM_OSC);
	file_info->C_dtype = create_complex_datatype();
	if( (H5LTread_dataset(file_info->input_file_handle, "Modes", file_info->C_dtype, run_data->u_z)) < 0) {
		fprintf(stderr, "\nERROR!! | in file:%s @line:%d | unable to read in HDF dataset \"%s\" in input file [%s], check input filename and HDF5 parameters.\n", __FILE__, __LINE__, "Modes", file_info->input_file_name);		
		exit(1);				
	}
	//////////////////////////////////////
	//-------------- MODES ------------ //
	//////////////////////////////////////
	#endif

	#ifdef __PHASE_ORDER
	///////////////////////////////////////
	//-------------- PHASES ------------ //
	///////////////////////////////////////
	// Check if data exists
	if( (H5Lexists(file_info->input_file_handle, "Phases", H5P_DEFAULT)) < 0 ) {
		fprintf(stderr, "\nERROR!! | in file:%s @line:%d | unable to find HDF dataset \"%s\" in input file [%s], check input filename and HDF5 parameters.\n", __FILE__, __LINE__, "Phases", file_info->input_file_name);		
		exit(1);				
	}

	if( (H5LTget_dataset_info(file_info->input_file_handle, "Phases", D2dims, NULL, NULL)) < 0) {
		fprintf(stderr, "\nERROR!! | in file:%s @line:%d | unable to get dataset info from dataset \"%s\" in input file [%s], check input filename and HDF5 parameters.\n", __FILE__, __LINE__, "Phases", file_info->input_file_name);		
		exit(1);				
	}
	// Get the number of timestesp
	sys_vars->num_tsteps = D2dims[0];

	// Check if data dimensions match input dimensions
	if(D2dims[1] != sys_vars->NUM_OSC) {
		fprintf(stderr, "\nERROR!! | in file:%s @line:%d | Dimensions of Dataset \"%s\" in input file [%s] are incorrect, check file and parameters.\n", __FILE__, __LINE__, "Phases", file_info->input_file_name);		
		exit(1);				
	}

	// Read In data
	run_data->phi = malloc(sizeof(double) * sys_vars->num_tsteps * sys_vars->NUM_OSC);
	if( (H5LTread_dataset(file_info->input_file_handle, "Phases", H5T_NATIVE_DOUBLE, run_data->phi)) < 0) {
		fprintf(stderr, "\nERROR!! | in file:%s @line:%d | unable to read in HDF dataset \"%s\" in input file [%s], check input filename and HDF5 parameters.\n", __FILE__, __LINE__, "Phases", file_info->input_file_name);		
		exit(1);				
	}
	///////////////////////////////////////
	//-------------- PHASES ------------ //
	///////////////////////////////////////
	
	///////////////////////////////////////
	//--------------- AMPS ------------- //
	///////////////////////////////////////
	// Check if data exists
	if( (H5Lexists(file_info->input_file_handle, "Amps", H5P_DEFAULT)) < 0 ) {
		fprintf(stderr, "\nERROR!! | in file:%s @line:%d | unable to find HDF dataset \"%s\" in input file [%s], check input filename and HDF5 parameters.\n", __FILE__, __LINE__, "Amps", file_info->input_file_name);		
		exit(1);				
	}

	if( (H5LTget_dataset_info(file_info->input_file_handle, "Amps", D2dims, NULL, NULL)) < 0) {
		fprintf(stderr, "\nERROR!! | in file:%s @line:%d | unable to get dataset info from dataset \"%s\" in input file [%s], check input filename and HDF5 parameters.\n", __FILE__, __LINE__, "Amps", file_info->input_file_name);		
		exit(1);				
	}

	// Check if data dimensions match input dimensions
	if(D2dims[1] != sys_vars->NUM_OSC) {
		fprintf(stderr, "\nERROR!! | in file:%s @line:%d | Dimensions of Dataset \"%s\" in input file [%s] are incorrect, check file and parameters.\n", __FILE__, __LINE__, "Amps", file_info->input_file_name);		
		exit(1);				
	}

	// Read In data
	run_data->amp = malloc(sizeof(double) * 1 * sys_vars->NUM_OSC);
	if( (H5LTread_dataset(file_info->input_file_handle, "Amps", H5T_NATIVE_DOUBLE, run_data->amp)) < 0) {
		fprintf(stderr, "\nERROR!! | in file:%s @line:%d | unable to read in HDF dataset \"%s\" in input file [%s], check input filename and HDF5 parameters.\n", __FILE__, __LINE__, "Amps", file_info->input_file_name);		
		exit(1);				
	}
	///////////////////////////////////////
	//--------------- AMPS ------------- //
	///////////////////////////////////////

	#ifdef __TRIADS
	///////////////////////////////////////
	//-------------- TRIADS ------------ //
	///////////////////////////////////////
	// Check if data exists
	if( (H5Lexists(file_info->input_file_handle, "Triads", H5P_DEFAULT)) < 0 ) {
		fprintf(stderr, "\nERROR!! | in file:%s @line:%d | unable to find HDF dataset \"%s\" in input file [%s], check input filename and HDF5 parameters.\n", __FILE__, __LINE__, "Triads", file_info->input_file_name);		
		exit(1);				
	}

	// Get attributes containing k1 and k3 ranges !!!!!!---> for some reason this doesn't read in the full attribute, only the first dimension - so I have to set the second dim manually
	if((H5LTget_attribute(file_info->input_file_handle, "Triads", "Triad_Dims", H5T_NATIVE_INT, D2dims)) < 0) {
		fprintf(stderr, "\nERROR!! | in file:%s @line:%d | unable to get attribute info from dataset \"%s\" in input file [%s], check input filename and HDF5 parameters.\n", __FILE__, __LINE__, "Triads", file_info->input_file_name);		
		exit(1);				
	}
	sys_vars->k3_range = D2dims[0];
	sys_vars->k1_range = (int)sys_vars->k3_range / 2;
	
	if( (H5LTget_dataset_info(file_info->input_file_handle, "Triads", D2dims, NULL, NULL)) < 0) {
		fprintf(stderr, "\nERROR!! | in file:%s @line:%d | unable to get dataset info from dataset \"%s\" in input file [%s], check input filename and HDF5 parameters.\n", __FILE__, __LINE__, "Triads", file_info->input_file_name);		
		exit(1);				
	}

	if(D2dims[1] != sys_vars->k3_range * sys_vars->k1_range) {
		fprintf(stderr, "\nERROR!! | in file:%s @line:%d | Dimensions of Dataset \"%s\" in input file [%s] are incorrect, check file and parameters.\n", __FILE__, __LINE__, "Triads", file_info->input_file_name);		
		exit(1);				
	}

	// Read In data
	run_data->triads = malloc(sizeof(double) * sys_vars->num_tsteps * sys_vars->k3_range * sys_vars->k1_range);
	if( (H5LTread_dataset(file_info->input_file_handle, "Triads", H5T_NATIVE_DOUBLE, run_data->triads)) < 0) {
		fprintf(stderr, "\nERROR!! | in file:%s @line:%d | unable to read in HDF dataset \"%s\" in input file [%s], check input filename and HDF5 parameters.\n", __FILE__, __LINE__, "Triads", file_info->input_file_name);		
		exit(1);				
	}
	///////////////////////////////////////
	//-------------- TRIADS ------------ //
	///////////////////////////////////////
	#endif
	#endif


	// Create wavenumber list
	run_data->kx = (int* )malloc(sizeof(int) * sys_vars->NUM_OSC);

	for (int i = 0; i < sys_vars->NUM_OSC; ++i) 
	{	
		run_data->kx[i] = i;
	}
	
	
}
// ---------------------------------------------------------------------
//  End of File
// ---------------------------------------------------------------------