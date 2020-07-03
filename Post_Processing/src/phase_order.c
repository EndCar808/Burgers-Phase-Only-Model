// Enda Carroll
// June 2020
// File including functions to perform post-processing phase orrder analysis


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
#include "utils.h"




// ---------------------------------------------------------------------
//  Function Definitions
// ---------------------------------------------------------------------
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
// ---------------------------------------------------------------------
//  End of File
// ---------------------------------------------------------------------