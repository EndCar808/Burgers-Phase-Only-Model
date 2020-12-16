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
#define __PHASES
// #define __TRIADS
#define __TRIAD_STATS
// #define __MODES
// #define __RHS
// #define __REALSPACE
// #define __GRAD
#define __REALSPACE_STATS
#define __TRANSIENTS
// #define __STATS



// ---------------------------------------------------------------------
//  Global Variables
// ---------------------------------------------------------------------
#define SAVE_DATA_STEP 10	// Parameter to determine after how many integrations steps data should be saved to output
#ifdef __REALSPACE_STATS
#define NBIN_VELINC 1000
#define BIN_LIM 40.0 
#else
#define BIN_LIM 0.0 
#endif

// ---------------------------------------------------------------------
//  End of File
// ---------------------------------------------------------------------
