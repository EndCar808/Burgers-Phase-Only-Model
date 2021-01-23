// Enda Carroll
// Jan 2021
// File containing functions to compute stats of the phase only equation

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
#include <gsl/gsl_histogram.h>
#include <gsl/gsl_rstat.h>
#include <sys/types.h>
#include <sys/stat.h>

// ---------------------------------------------------------------------
//  User Libraries and Headers
// ---------------------------------------------------------------------
#include "data_types.h"
#include "solver.h"
#include "utils.h"


// ---------------------------------------------------------------------
//  Function Definitions
// ---------------------------------------------------------------------
void linspace(double* arr, double a, double b, int n_points) {

	// Fill first element
	arr[0] = a;

	// Loop through and fill array
	for (int i = 1; i < n_points - 1; ++i) {
		arr[i] = a + (b - a) * ((double) i) / (n_points - 1);
	}

	// Fill last element
	arr[n_points - 1] = b;
}

void histogram(double* counts, double* data, double* bins, int num_bins, int num_data) {

	// Temp varaiabels
	double tmp_sample;

	// Loop through data
	for (int i = 0; i < num_data; ++i) {
		// Select current data sample
		tmp_sample = data[i];

		// Loop through bins to see which bin the current sample falls in
		for (int j = 0; j < num_bins - 1; ++j) {
			if(tmp_sample >= bins[j] && tmp_sample < bins[j + 1]) {
				counts[j] += 1;
			}
			else {
				continue;
			}
		}
	}
}

void set_vel_inc_hist_bin_ranges(double* bins, double* u, int num_osc, int r) {

	// Initialize variables
	int N_osc     = num_osc - 1;
	double vel_inc;
	double dx = 0.5 / (double)N_osc;
	double tmp_incr;
	double norm;

	/////////////////////////
	// Compute Std Devs
	/////////////////////////
	// Compute std dev of velocity increment
	tmp_incr = 0.0;
	for (int i = 0; i < 2 * N_osc; ++i) {
		vel_inc = u[(i + r) % (2 * N_osc)] - u[i];

		tmp_incr += pow(vel_inc, 2);
	}
	// Calculate the std dev
	norm = sqrt(tmp_incr * dx);

	/////////////////////////
	// Compute Bin Edges
	/////////////////////////
	linspace(bins, -BIN_LIM * norm, BIN_LIM * norm, NBIN_VELINC + 1);

}

void compute_real_space_stats(double* small_counts, double* small_bins, double* large_counts, double* large_bins, double* grad_counts, double* grad_bins, double* u, double* u_grad, int num_osc) {

	// Initialize variables
	int r;
	int N_osc     = num_osc - 1;
	int num_r_inc = 2;
	int r_inc[num_r_inc];
	r_inc[0] = 1;
	r_inc[1] = N_osc;
	double vel_inc;
	double* vel_inc_small = (double* )malloc(sizeof(double) * 2 * N_osc);
	double* vel_inc_large = (double* )malloc(sizeof(double) * 2 * N_osc);


	////////////////////////
	// Compute Increments
	////////////////////////
	// Compute velocity increments
	for (int r_indx = 0; r_indx < num_r_inc; ++r_indx) {
		// Get current incr
		r = r_inc[r_indx]; 
	
		for (int i = 0; i < 2 * N_osc; ++i) {
			// Get current increment
			vel_inc = u[(i + r) % (2 * N_osc)] - u[i];

			// Store increments
			if (r == 1) {
				vel_inc_small[i] = vel_inc;
			}
			else if(r == N_osc) {
				vel_inc_large[i] = vel_inc;
			}
		}
	}

	////////////////////////
	// Compute Histograms
	////////////////////////
	// Compute small scale histogram
	histogram(small_counts, vel_inc_small, small_bins, NBIN_VELINC + 1, 2 * N_osc);
	
	// Compute large scale histogram
	histogram(large_counts, vel_inc_large, large_bins, NBIN_VELINC + 1, 2 * N_osc);
	
	// Compute Gradient histogram
	for (int i = 0; i < 2 * N_osc; ++i) {
		u_grad[i] *= M_PI / (double) N_osc;
	}
	histogram(grad_counts, u_grad, grad_bins, NBIN_VELINC + 1, 2 * N_osc);


	// Free tmp memory
    free(vel_inc_small);
    free(vel_inc_large);
}


void gsl_set_vel_inc_hist_bin_ranges(gsl_histogram** hist_incr, double* u, double* u_grad, double vel_sec_mnt, double grad_sec_mnt, int num_osc) {

	// Initialize variables
	int r;
	int N_osc     = num_osc - 1;
	int num_r_inc = 2;
	int r_inc[num_r_inc];
	r_inc[0] = 1;
	r_inc[1] = N_osc;
	double vel_inc;
	double dx = 0.5 / (double)N_osc;
	double std_dev;


	// Initialize running stats workspace - used to find min & max bin edges
	gsl_rstat_workspace* vel_inc_stats[num_r_inc + 1];

	////////////////////////
	// Compute Increments
	////////////////////////
	// Compute velocity increments for Std Dev
	for (int r_indx = 0; r_indx < num_r_inc; ++r_indx) {
		// Get current incr
		r = r_inc[r_indx]; 
	
		// Initialize stats accumulator
		vel_inc_stats[r_indx] = gsl_rstat_alloc();
		for (int i = 0; i < 2 * N_osc; ++i) {
			// Get current increment
			vel_inc = u[(i + r) % (2 * N_osc)] - u[i];

			// Add incr to accumulator
			gsl_rstat_add(vel_inc, vel_inc_stats[r_indx]);
		}
	}

	// Set accumulator for the gradient
	vel_inc_stats[num_r_inc] = gsl_rstat_alloc();
	for (int i = 0; i < 2 * N_osc; ++i)
	{
		// Add next gradient value to accum
		gsl_rstat_add(u_grad[i], vel_inc_stats[num_r_inc]);
	}

	//////////////////////
	// Set Bin Edges
	//////////////////////
	// Bin ranges will be set to +-BIN_LIM*Std Dev
	for (int i = 0; i < num_r_inc; ++i) {
		// Get the std dev of the smallest incr
		std_dev = gsl_rstat_rms(vel_inc_stats[i]);
		if ( (gsl_histogram_set_ranges_uniform(hist_incr[i], -BIN_LIM * std_dev, BIN_LIM * std_dev)) != 0 ) {
			fprintf(stderr, "ERROR: unable to set ranges for the GSL histogram: Hist_Incrment[%d]\n", i);
			exit(1);						
		}
	}

	// Get the std dev of the gradient
	std_dev = grad_sec_mnt * M_PI / (double) N_osc;
	if ( (gsl_histogram_set_ranges_uniform(hist_incr[num_r_inc], -BIN_LIM * std_dev, BIN_LIM * std_dev)) != 0 ) {
		fprintf(stderr, "ERROR: unable to set ranges for the GSL histogram: %s\n", "VelocityGradient");
		exit(1);						
	}	

	// Free memory
	for (int i = 0; i < num_r_inc + 1; ++i) {
		gsl_rstat_free(vel_inc_stats[i]);
	}
	
}


void gsl_compute_real_space_stats(gsl_histogram** hist_incr, gsl_rstat_workspace** incr_stat, double* str_func, double* u, double* u_grad, double vel_sec_mnt, double grad_sec_mnt, int num_osc, int max_p) {

	// Initialize variables
	int r;
	int N_osc     = num_osc - 1;
	int num_r_inc = 2;
	int r_inc[num_r_inc];
	r_inc[0] = 1;
	r_inc[1] = N_osc;
	double vel_inc;
	double vel_inc_abs;
	double dx = 0.5 / (double)N_osc;


	////////////////////////
	// Compute Increments
	////////////////////////
	// Compute Velocity Incrments
	for (int r_indx = 0; r_indx < num_r_inc; ++r_indx) {
		// Get current incr
		r = r_inc[r_indx]; 
		
		for (int i = 0; i < 2 * N_osc; ++i) {
			// Get current increment
			vel_inc = u[(i + r) % (2* N_osc)] - u[i];

			// Add current vel inc to appropriate bin
			gsl_histogram_increment(hist_incr[r_indx], vel_inc);

			// Add current vel inc to stats accumulator
			gsl_rstat_add(vel_inc, incr_stat[r_indx]);
		}
	}
	
	// Compute the Gradient Histogram
	for (int i = 0; i < 2 * N_osc; ++i)	{
		// Add current gradient to appropriate bin
		gsl_histogram_increment(hist_incr[num_r_inc], u_grad[i] * M_PI / N_osc);
		
		// Add current gradient to stats accumulator
		gsl_rstat_add(u_grad[i] * M_PI / N_osc, incr_stat[num_r_inc]);
	}

	///////////////////////////////
	// Compute Structure Functions
	///////////////////////////////
	#ifdef __STR_FUNCS
	for (int p = 2; p <= max_p; ++p) {
		for (int r = 1; r <= N_osc; ++r) {		
			vel_inc = 0.0;
 			for (int i = 0; i < 2 * N_osc; ++i) {
				// Get current increment
				vel_inc += pow(u[(i + r) % (2 * N_osc)] - u[i], p);
			}
			// Update structure func
			str_func[(p - 2) * N_osc + (r - 1)] += vel_inc * dx;
		}
	}
	#endif

}
// ---------------------------------------------------------------------
//  End of file
// ---------------------------------------------------------------------