// Enda Carroll
// June 2020
// Header file for the data types used for post-processing 

// ---------------------------------------------------------------------
//  Standard Libraries and Headers
// ---------------------------------------------------------------------
#include <hdf5.h>

#include <gsl/gsl_histogram.h> /**< include the header file for GSL Library for 1D histograms */ 
#include <gsl/gsl_statistics.h>
#include <gsl/gsl_rstat.h>
#include <gsl/gsl_errno.h>
// ---------------------------------------------------------------------
//  User Libraries and Headers
// ---------------------------------------------------------------------


// ---------------------------------------------------------------------
//  Analysis Control Variables
// ---------------------------------------------------------------------
// These definitions control which analysis is to be performed
#define __STATS
#define __TRIADS
#define __PHASE_ORDER
#define __DERIVATIVE


// ---------------------------------------------------------------------
//  Datatype Flag Definitions
// ---------------------------------------------------------------------
#define NBINS_VELINC 100
#define NUM_STR_P 5
#define NUM_VEL_INC 5
#define NBINS_TRIAD_LOCAL 100
#define NBINS_TRIAD_GLOBAL 10000

// ---------------------------------------------------------------------
//  Global Variable Definitions
// ---------------------------------------------------------------------
// These are the definitions of the global variables above
// System variables struct 
typedef struct system_vars {
	// Global variables
	int N;
	int M;
	int NUM_OSC;
	// Spectrum variables
	int k0;
	int kmin;
	int kmax;
	int k3_range;
	int k1_range;
	double alpha;
	double beta;
	// Integration variables
	int num_tsteps;
	int post_trans_iters;
	int trans_iters;
	double t0;
	double T;
	double dt;
	// Initial condition variables
	char u0[128];
} system_vars_struct;

// Runtime data struct
typedef struct runtime_data {
	int* kx;
	double* x;
	double* u;
	double* u_pad;
	fftw_complex* u_z;
	fftw_complex* u_z_pad;
	double* phi;
	double* amp;
	double* triads;
} runtime_data_struct;

// Running Stats Variables
typedef struct runstat_vars {
	double* du_r_small;
	double* du_r_mid;
	double* du_r_large;
	double* struc_p[NUM_VEL_INC][NUM_STR_P];
	double* dudx;
	double* VelInc_Stats[NUM_VEL_INC][5];
	gsl_rstat_workspace* RunVelStats;
} runstat_vars_struct;

// Phase Order Variables
typedef struct phaseorder_vars {
	gsl_histogram* TriadGlobal;
	gsl_histogram* TriadLocal;
} phaseorder_vars_struct;

// HDF File infor struct 
typedef struct HDF_file_info {
	char input_file_name[512];
	char output_file_name[512];
	hid_t input_file_handle;
	hid_t output_file_handle;

	// Complex datatype param
	hid_t C_dtype;
} HDF_file_info_struct;

// Create compound datatype for complex numbers
typedef struct complex_type {
	double re;   // real part 
	double im;   // imaginary part 
} complex_type_struct;

// ---------------------------------------------------------------------
//  Global Variables Declarations
// ---------------------------------------------------------------------
// These global variables are pointers to structs that contain the necessary data for analysis
extern system_vars_struct   	*sys_vars;               // Global pointer to system parameters struct 
extern runtime_data_struct  	*run_data;               // Global pointer to system runtime data struct 
extern runstat_vars_struct 	   *stat_vars;	             // Global pointer to statistics analysis parameters struct
extern phaseorder_vars_struct  *phase_vars;			     // Global pointer to the phase order analysis parameters struct
extern HDF_file_info_struct    *file_info;	             // Global pointer to the input/output hdf5 file info


// ---------------------------------------------------------------------
//  End of File
// ---------------------------------------------------------------------