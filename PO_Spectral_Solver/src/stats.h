// Enda Carroll
// Jan 2021
// Header file to acompany the stats.c file


// ---------------------------------------------------------------------
//  Standard Libraries and Headers
// ---------------------------------------------------------------------
#include <gsl/gsl_histogram.h>
#include <gsl/gsl_rstat.h>

// ---------------------------------------------------------------------
//  User Libraries and Headers
// ---------------------------------------------------------------------



// ---------------------------------------------------------------------
//  Function Prototypes
// ---------------------------------------------------------------------
void linspace(double* arr, double a, double b, int n_points);
void histogram(double* counts, double* data, double* bins, int num_bins, int num_data);
void set_vel_inc_hist_bin_ranges(double* bins, double* u, int num_osc, int r);
void compute_real_space_stats(double* small_counts, double* small_bins, double* large_counts, double* large_bins, double* grad_counts, double* grad_bins, double* u, double* u_grad, int num_osc);

void gsl_set_vel_inc_hist_bin_ranges(gsl_histogram** hist_incr, double* u, double* u_grad, double vel_sec_mnt, double grad_sec_mnt, int num_osc);
void gsl_compute_real_space_stats(gsl_histogram** hist_incr, gsl_rstat_workspace** incr_stat, double* str_func, double* u, double* u_grad, double vel_sec_mnt, double grad_sec_mnt, int num_osc, int max_p);
// ---------------------------------------------------------------------
//  End of File
// ---------------------------------------------------------------------