// Enda Carroll
// Sept 2019
// Header file for utility functions in utils.c

// ---------------------------------------------------------------------
//  Standard Libraries and Headers
// ---------------------------------------------------------------------



// ---------------------------------------------------------------------
//  User Libraries and Headers
// ---------------------------------------------------------------------




// ---------------------------------------------------------------------
//  Function Prototypes
// ---------------------------------------------------------------------

void initial_condition(double* u, fftw_complex* u_z, fftw_plan real2compl, double dx, int N);

int max_indx_d(double* array, int n);

void write_array(double *A, int n, char *filename);



double system_energy(double complex* u_z, int N);
double system_enstrophy(double complex* u_z, int* k, int N);

void deriv(double complex* u_z, double complex* dudt_z, double nu, int* k, fftw_plan *real2compl_ptr, fftw_plan *compl2real_ptr, int N);