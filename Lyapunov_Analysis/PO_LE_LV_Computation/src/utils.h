// Enda Carroll
// May 2020
// Header file to utils.c file which computes the Lyapunov spectrum and 
// vectors of the Phase Only Burgers equation


// ---------------------------------------------------------------------
//  Standard Libraries and Headers
// ---------------------------------------------------------------------



// ---------------------------------------------------------------------
//  User Libraries and Headers
// ---------------------------------------------------------------------




// ---------------------------------------------------------------------
//  Function Prototypes
// ---------------------------------------------------------------------
void initial_conditions_lce(double* pert, double* phi, double* amp, fftw_complex* u_z, int* kx, int num_osc, int k0, int kmin, double a, double b, char* IC, int numLEs);
void max(double* a, int n, int k0, double* max_val);
void min(double* a, int n, int k0, double* min_val);

void mem_chk (void *arr_ptr, char *name);

double get_timestep(double* amps, fftw_plan plan_c2r, fftw_plan plan_r2c, int* kx, int n, int num_osc, int k0);
int get_transient_iters(double* amps, fftw_plan plan_c2r, fftw_plan plan_r2c, int* kx, int n, int num_osc, int k0);


void conv_2N_pad(fftw_complex* convo, fftw_complex* uz, fftw_plan *fftw_plan_r2c_ptr, fftw_plan *fftw_plan_c2r_ptr, int n, int num_osc, int k0);
void conv_direct(fftw_complex* convo, fftw_complex* u_z, int num_osc, int k0);

void po_rhs(double* rhs, fftw_complex* u_z, fftw_plan *plan_c2r_pad, fftw_plan *plan_r2c_pad, int* kx, int n, int num_osc, int k0);

void triad_phases(double* triads, fftw_complex* phase_order, double* phi, int kmin, int kmax);

void get_output_file_name(char* output_file_name, int N, int k0, double a, double b, char* u0, int ntsteps, int m_end, int m_iter, int trans_iters, int numLEs);
void create_hdf5_slabbed_dset(hid_t* file_handle, char* dset_name, hid_t* file_space, hid_t* data_set, hid_t* mem_space, hid_t dtype, hid_t* dset_dims, hid_t* dset_max_dims, hid_t* dset_chunk_dims, const int num_dims);
hid_t create_complex_datatype();

void open_output_create_slabbed_datasets_lce(hid_t* file_handle, char* output_file_name, hid_t* file_space, hid_t* data_set, hid_t* mem_space, hid_t dtype, int num_t_steps, int num_m_steps, int num_clv_steps, int num_osc, int k_range, int k1_range, int kmin, int numLEs);
void write_hyperslab_data(hid_t file_space, hid_t data_set, hid_t mem_space, hid_t dtype, double* data, char* data_name, int n, int index);
// ---------------------------------------------------------------------
//  End of File
// ---------------------------------------------------------------------
