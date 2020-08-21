// Enda Carroll
// May 2020
// Header file for global definitions, global parameters etc




// ---------------------------------------------------------------------
//  Output Datasets
// ---------------------------------------------------------------------
// #define __TRIADS
// #define __MODES
// #define __REALSPACE
#define __RNORM
// #define __TRANSIENTS			// Turn on transient iterations - these iterations are ignored in the calculation



// ---------------------------------------------------------------------
//  Parameters & Constants
// ---------------------------------------------------------------------
#define SAVE_DATA_STEP 100	// Parameter to determine after how many integrations steps data should be saved to output
#define SAVE_LCE_STEP  10   // Parameter to determine how often to save LCE data
// #ifdef __TRANSIENTS
// #define TRANS_STEPS 0.1     // Set the % (of the total iterations) of transient iterations to perform
// #endif

#ifndef M_PI
    #define M_PI 3.14159265358979323846
#endif