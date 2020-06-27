// Enda Carroll
// May 2020
// Header file to acompany the lce_spectrum.c file


// ---------------------------------------------------------------------
//  Standard Libraries and Headers
// ---------------------------------------------------------------------



// ---------------------------------------------------------------------
//  User Libraries and Headers
// ---------------------------------------------------------------------



// ---------------------------------------------------------------------
//  Function Prototypes
// ---------------------------------------------------------------------
void pert_initial_condition(double* phi, double* amp, int* kx, int num_osc, int k0, double a, double b, double pert, int pert_dir);
double get_global_timestep(double a, double b, int n, int k0);

void open_output_create_slabbed_datasets_lce(hid_t* file_handle, char* output_file_name, hid_t* file_space, hid_t* data_set, hid_t* mem_space, int num_t_steps, int num_m_steps, int num_osc, int k_range, int k1_range, int kmin);

void compute_spectrum(int N, int k0, double a, double b, int m_end, int m_iter, double pert);
double* pert_solver(double* phi, double* amp, int* kx, fftw_plan fftw_plan_c2r, fftw_plan fftw_plan_r2c, int N, int k0,  int iters, int m, double pert, int pert_dir);

void orthonormalize(double* pert, double* znorm, int num_osc, int kmin);
// ---------------------------------------------------------------------
//  End of File
// ---------------------------------------------------------------------