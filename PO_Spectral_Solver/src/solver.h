// Enda Carroll
// May 2020
// Header file to acompany the solver.c file


// ---------------------------------------------------------------------
//  Standard Libraries and Headers
// ---------------------------------------------------------------------



// ---------------------------------------------------------------------
//  User Libraries and Headers
// ---------------------------------------------------------------------




// ---------------------------------------------------------------------
//  Function Prototypes
// ---------------------------------------------------------------------

void initial_condition(double* phi, double* amp, int* kx, int num_osc, int k0, int cutoff, double a, double b);
void max(double* a, int n, int k0, double* max_val);
void min(double* a, int n, int k0, double* min_val);

double get_timestep(double* amps, fftw_plan plan_c2r, fftw_plan plan_r2c, int* kx, int n, int num_osc, int k0);
int get_transient_iters(double* amps, fftw_plan plan_c2r, fftw_plan plan_r2c, int* kx, int n, int num_osc, int k0);

void conv_2N_pad(fftw_complex* convo, fftw_complex* uz, fftw_plan *fftw_plan_r2c_ptr, fftw_plan *fftw_plan_c2r_ptr, int n, int num_osc, int k0);
void conv_23(fftw_complex* convo, fftw_complex* uz, fftw_plan *fftw_plan_r2c_ptr_23, fftw_plan *fftw_plan_c2r_ptr_23, int n, int kmax, int k0);
void conv_direct(fftw_complex* convo, fftw_complex* u_z, int n, int k0);

void po_rhs(double* rhs, fftw_complex* u_z, fftw_plan *plan_c2r_pad, fftw_plan *plan_r2c_pad, int* kx, int n, int num_osc, int k0);
void po_rhs_extended(double* rhs, double* rhs_ext, fftw_complex* u_z, double* pert, fftw_plan *plan_c2r_pad, fftw_plan *plan_r2c_pad, int* kx, int n, int num_osc, int k0);

void jacobian(double* jac, fftw_complex* u_z, int n, int num_osc, int k0);
double trace(fftw_complex* u_z, int n, int num_osc, int k0);

void triad_phases(double* triads, fftw_complex* phase_order, double* phi, int kmin, int kmax);

double system_energy(fftw_complex* u_z, int N);
double system_enstrophy(fftw_complex* u_z, int* k, int N);