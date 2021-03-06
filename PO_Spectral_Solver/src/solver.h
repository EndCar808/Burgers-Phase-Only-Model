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
void initial_condition(double* phi, double* amp, fftw_complex* u_z, int* kx, int num_osc, int k0, double a, double b, char* IC);
void max(double* a, int n, int k0, double* max_val);
void min(double* a, int n, int k0, double* min_val);

void get_output_file_name(char* output_file_name, int N, int k0, double a, double b, char* u0, int ntsteps, int trans_iters);
hid_t create_complex_datatype();

void open_output_create_slabbed_datasets(hid_t* file_handle, char* output_file_name, hid_t* file_space, hid_t* data_set, hid_t* mem_space, hid_t dtype, int num_t_steps, int num_osc, int k_range, int k1_range);
void create_hdf5_slabbed_dset(hid_t* file_handle, char* dset_name, hid_t* file_space, hid_t* data_set, hid_t* mem_space, hid_t dtype, hid_t* dset_dims, hid_t* dset_max_dims, hid_t* dset_chunk_dims, const int num_dims);
void write_hyperslab_data(hid_t file_space, hid_t data_set, hid_t mem_space, hid_t dtype, double* data, char* data_name, int n, int index);
void write_fixed_point(double* phi, double b, char* u0, int N, int num_osc, int k0, int kdim, hid_t* data_set, int save_data_indx);

void fixed_point_search(int N, int Nmax, int k0, double a, double b, int iters, int save_step, char* u0);
int solver(int N, int k0, double a, double b, int iters, int save_step, char* u0);

double get_timestep(double* amps, fftw_plan plan_c2r, fftw_plan plan_r2c, int* kx, int n, int num_osc, int k0);
int get_transient_iters(double* amps, fftw_plan plan_c2r, fftw_plan plan_r2c, int* kx, int n, int num_osc, int k0);

void conv_2N_pad(fftw_complex* convo, fftw_complex* uz, fftw_plan *fftw_plan_r2c_ptr, fftw_plan *fftw_plan_c2r_ptr, int n, int num_osc, int k0);
void conv_23(fftw_complex* convo, fftw_complex* uz, fftw_plan *fftw_plan_r2c_ptr_23, fftw_plan *fftw_plan_c2r_ptr_23, int n, int kmax, int k0);
void conv_direct(fftw_complex* convo, fftw_complex* u_z, int n, int k0);

void po_rhs(double* rhs, fftw_complex* u_z, fftw_plan *plan_c2r_pad, fftw_plan *plan_r2c_pad, int* kx, int n, int num_osc, int k0);

void triad_phases(double* triads, fftw_complex* phase_order, double* phi, int kmin, int kmax);
void amp_normalize(double* norm, double* amp, int num_osc, int k0);
void sync_phase(fftw_complex* ordered, fftw_complex* heaviside, fftw_complex* heavi_ordered, double* phi, double* amps, int kmin, int kmax);

double gradient_energy(double* a_k, int* k, int num_osc);
double theoretical_energy(double* a_k, int num_osc);
double system_energy(fftw_complex* u_z, int N);
double system_enstrophy(fftw_complex* u_z, int* k, int N);
// ---------------------------------------------------------------------
//  End of File
// ---------------------------------------------------------------------