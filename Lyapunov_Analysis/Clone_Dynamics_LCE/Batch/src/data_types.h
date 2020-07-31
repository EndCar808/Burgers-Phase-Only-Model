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
//  Datasets
// ---------------------------------------------------------------------
// #define __TRIADS
// #define __MODES
// #define __REALSPACE
// #define __TRANSIENTS
// #define __STATS



// ---------------------------------------------------------------------
//  Global Variables
// ---------------------------------------------------------------------
#define SAVE_DATA_STEP 1	// Parameter to determine after how many integrations steps data should be saved to output
#define SAVE_LCE_STEP 1	    // Parameter to determine after how many algorithm steps lce data should be saved to output



// ---------------------------------------------------------------------
//  End of File
// ---------------------------------------------------------------------
