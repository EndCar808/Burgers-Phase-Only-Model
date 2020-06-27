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

	if(argc != 7) {
		fprintf(stderr, "ERROR: Not enough input parameters specified.\nLooking for:\nN\nk0\nAlpha\nBeta\nSaving Iters\nTransient Iters\n");
		exit(1);
	} 
	
	// Read Input
	sys_vars->N  = atoi(argv[1]);
	sys_vars->k0 = atoi(argv[2]);
	sys_vars->alpha    = atof(argv[3]);
	sys_vars->beta     = atof(argv[4]);
	sys_vars->post_trans_iters = atoi(argv[5]);
	sys_vars->trans_iters     = atoi(argv[6]);

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
// ---------------------------------------------------------------------
//  End of File
// ---------------------------------------------------------------------