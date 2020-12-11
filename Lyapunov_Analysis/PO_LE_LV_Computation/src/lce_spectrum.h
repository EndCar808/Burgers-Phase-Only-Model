// Enda Carroll
// May 2020
// Header file to lce_spectrum.c file which computes the Lyapunov spectrum
// of the Phase Only Burgers equation


// ---------------------------------------------------------------------
//  Standard Libraries and Headers
// ---------------------------------------------------------------------



// ---------------------------------------------------------------------
//  User Libraries and Headers
// ---------------------------------------------------------------------




// ---------------------------------------------------------------------
//  Function Prototypes
// ---------------------------------------------------------------------
void po_rhs_extended(double* rhs, double* rhs_ext, fftw_complex* u_z, double* pert, fftw_plan *plan_c2r_pad, fftw_plan *plan_r2c_pad, int* kx, int n, int num_osc, int k0, int numLEs);

void jacobian(double* jac, fftw_complex* u_z, int num_osc, int k0);
double trace(fftw_complex* u_z, int num_osc, int k0);

void orthonormalize(double* pert, double* R_tmp, int num_osc, int kmin, int numLEs);
void modified_gs(double* q, double* r, int num_osc, int kmin);
void compute_CLVs(hid_t* file_space, hid_t* data_set, hid_t* mem_space, double* R, double* GS, int DOF, int numLEs, int m_rev_iters, int m_rev_trans);

void compute_lce_spectrum_clvs(int N, double a, double b, char* u0, int k0, int m_end, int m_iter, int numLEs);
// ---------------------------------------------------------------------
//  End of File
// ---------------------------------------------------------------------