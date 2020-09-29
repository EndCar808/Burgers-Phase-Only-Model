// Enda Carroll
// Sept 2020
// Header file to lce_spectrum.c file which computes the Lyapunov spectrum
// and CLVs of the Lorenz


// ---------------------------------------------------------------------
//  Standard Libraries and Headers
// ---------------------------------------------------------------------



// ---------------------------------------------------------------------
//  User Libraries and Headers
// ---------------------------------------------------------------------




// ---------------------------------------------------------------------
//  Function Prototypes
// ---------------------------------------------------------------------
void mem_chk (void *arr_ptr, char *name);

void get_output_file_name(char* output_file_name, double dt, int m_iters, int m_trans, int m_avg);

void open_output_create_slabbed_datasets(hid_t* file_handle, char* output_file_name, hid_t* file_space, hid_t* data_set, hid_t* mem_space, int num_t_steps, int num_m_steps, int num_clv_steps, int N, int numLEs);

void write_hyperslab_data_d(hid_t file_space, hid_t data_set, hid_t mem_space, double* data, char* data_name, int n, int index);

void initial_conditions_lce(double* pert, double* x, int N, int numLEs);

void rhs_extended(double* rhs, double* rhs_ext, double* x_tmp, double* x_pert, int n, int numLEs);

void orthonormalize(double* pert, double* R_tmp, int N, int numLEs);

void compute_angles(double* angles, double* CLV, int DOF, int numLEs);

void compute_CLVs(hid_t file_space, hid_t data_set, hid_t mem_space, double* R, double* GS, int N, int numLEs, int m_rev_iters, int m_rev_trans);

void compute_lce_spectrum(int N, int numLEs, int m_trans, int m_rev_trans, int m_end, int m_iter);
// ---------------------------------------------------------------------
//  End of File
// ---------------------------------------------------------------------