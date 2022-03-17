// Enda Carroll
// May 2020
// Datatypes header file for the pseudospectral solver

// ---------------------------------------------------------------------
//  Standard Libraries and Headers
// ---------------------------------------------------------------------





// ---------------------------------------------------------------------
//  Compile Time Macros and Definitions
// ---------------------------------------------------------------------
#define checkError(x) ({int __val = (x); __val == -1 ? \
	({fprintf(stderr, "ERROR ("__FILE__":%d) -- %s\n", __LINE__, strerror(errno)); \
	exit(-1);-1;}) : __val; })


// ---------------------------------------------------------------------
//  Code Functions
// ---------------------------------------------------------------------
// #define __FXD_PT_SEARCH__

// ---------------------------------------------------------------------
//  Datasets
// ---------------------------------------------------------------------
#define __AMPS
#define __TIME
// #define __PHASES
// #define __TRIADS
// #define __TRIAD_STATS
#define __TRIAD_ORDER
#if defined(__TRIAD_ORDER)
// The in time qunatities
// #define __ADLER_PHASE_ORDER
#define __SCALE_PHASE_ORDER
// #define __THETA_TIME_PHASE_ORDER
// #define __OMEGA_K
#define __SIN_THETA_K
#define __PHI_K_DOT
// The over time qunatities
#define __SIN_THETA_K_AVG
#define __P_K_AVG
#define __R_K_AVG
// #define __ALT_ORDER_PARAMS
#endif
// #define __MODES
// #define __RHS
// #define __REALSPACE
// #define __GRAD
// #define __REALSPACE_STATS
// #define __STR_FUNCS
#define __TRANSIENTS
// #define __STATS



// ---------------------------------------------------------------------
//  Global Variables
// ---------------------------------------------------------------------
#define SAVE_DATA_STEP 100	// Parameter to determine after how many integrations steps data should be saved to output
#define NBIN_VELINC 10000
#define BIN_LIM 45.0 


// #ifndef M_PI
//     #define M_PI 3.14159265358979323846
// #endif

// ---------------------------------------------------------------------
//  End of File
// ---------------------------------------------------------------------
