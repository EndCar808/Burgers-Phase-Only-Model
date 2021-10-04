// Enda Carroll
// May 2020
// Header file for global definitions, global parameters etc




// ---------------------------------------------------------------------
//  Output Datasets
// ---------------------------------------------------------------------
// #define __PHASES				// Save the phases
#define __LCE_LAST 		    	// Save only the last state of the LCEs
// #define __LCE_ALL   			// Save each state of the the LCE computation
// #define __TRIAD_ORDER			// Save the scale dependent kuramoto order parameters
// #define __TRIADS				// Save the triads
// #define __MODES                 // Save the modes
// #define __REALSPACE			    // Save the real space solution
// #define __GRAD                  // Save the gradient
// #define __RNORM			        // Save the diagonals of the R matrix
#define __TRANSIENTS			// Turn on transient iterations - these iterations are ignored in the calculation
// #define __CLVs					// Compute the CLVs
#define __CLV_MAX				// Save the maximal CLV only
#define __CLV_STATS             // Compute and save CLV stats such as centroid, entropy etc
// #define __ANGLES     		    // Compute the angles between CLVs


// ---------------------------------------------------------------------
//  Parameters & Constants
// ---------------------------------------------------------------------
#define PRINT_SCREEN		 // Turn on printing to screen
// #define PRINT_LCEs			 // Turn on printing update of the LCEs
#define SAVE_DATA_STEP 100  	 // Parameter to determine after how many integrations steps data should be saved to output
#define SAVE_LCE_STEP 10      // Parameter to determine how often to save LCE data
#define SAVE_CLV_STEP 10      // Parameter to determine how oftern to save the CLV data
// #ifdef __TRANSIENTS
// #define TRANS_STEPS 0.01     // Set the % (of the total iterations) of transient iterations to perform
// #endif

#ifndef M_PI
    #define M_PI 3.14159265358979323846
#endif

// ---------------------------------------------------------------------
//  End of File
// ---------------------------------------------------------------------