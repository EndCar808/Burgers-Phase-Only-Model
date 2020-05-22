// Enda Carroll
// Sept 2019
// Programme to run through some examples using FFTW

// ---------------------------------------------------------------------
//  Standard Libraries and Headers
// ---------------------------------------------------------------------
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <complex.h> // having this before <fftw3.h> means fftw_complex is the native double-precision complex
					 // type and you can manipulate it with ordinary arithmetic. Otherwise FFTW defines its own complex type

#include <fftw3.h>
#include <hdf5.h>
#include <hdf5_hl.h>

// ---------------------------------------------------------------------
//  User Libraries and Headers
// ---------------------------------------------------------------------
#include <fftw3.h> // include the latest FFTW library


 
// ---------------------------------
//  Main Programme
// ---------------------------------
int main(int argc, char** argv) {

	// size of array
	int N = pow(3, 6); 

	// Allocate memory
	double* in_arr;	
	in_arr  = (double* )malloc(N*sizeof(double)); 


	fftw_complex* out_arr;
	out_arr = (double complex* )fftw_malloc(N*sizeof(double complex ));

	// create fftw plan to store the type of FFT we want to perform
	fftw_plan fftw_plan_r2c, fftw_plan_c2r;



	printf("ok\n");
	
	




	// create plans - ensure no overwriting - fill arrays after
	fftw_plan_r2c = fftw_plan_dft_r2c_1d(N, in_arr, out_arr, FFTW_PRESERVE_INPUT); 
	fftw_plan_c2r = fftw_plan_dft_c2r_1d(N, out_arr, in_arr, FFTW_PRESERVE_INPUT);



	printf("ok 1\n");

	// fill array
	double dx = 2*M_PI/N;
	double x;
	for (int i = 0; i < N; ++i)	{
		x = (double)i*dx;

		in_arr[i] = sin(x);

		out_arr[i] = 0;
	}


	printf("ok 2\n");

	// execute plan - essentially performs the dft outlined in the plan
	fftw_execute(fftw_plan_r2c); // can perform this multiple times


	printf("ok3\n");


	// write output to file
	// create hdf5 file identifier handle
	hid_t HDF_file_handle;

	// define filename - const because it doesnt change
	const char* output_file_name = "./Test_Data.h5";

	// create datafile - H5F_ACC_TRUNC overwrites file if it exists already
	HDF_file_handle = H5Fcreate(output_file_name, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);



	printf("ok 4\n");


	/* use hdf5_hl library to create datasets */
	// define dimensions and dimension sizes
	hsize_t HDF_D1ndims = 1;
	hsize_t D1dims[HDF_D1ndims];	


	// create the C stuct to store the complex datatype
	typedef struct compound_complex {
		double re;
		double im;
	} compound_complex;

	// declare and intialize the new complex datatpye
	compound_complex complex_data;
	complex_data.re = 0.0;
	complex_data.im = 0.0;

	// create the new HDF5 compound datatpye using the new complex datatype above
	// use this id to write/read the complex modes to/from file later
	hid_t comp_id; 
	comp_id = H5Tcreate(H5T_COMPOUND, sizeof(complex_data));



	// insert each of the members of the complex datatype struct into the new HDF5 compound datatype
	// using the offsetof() function to find the offset in bytes of the field of the complex struct type
	H5Tinsert(comp_id, "r", offsetof(compound_complex, re), H5T_NATIVE_DOUBLE);
	H5Tinsert(comp_id, "i", offsetof(compound_complex, im), H5T_NATIVE_DOUBLE);


	D1dims[0] = N;

	for (int i = 0; i < N; ++i) {
		printf("Output[%d]: %lf %lfi\n", i, creal(out_arr[i]), cimag(out_arr[i]));
	}
	H5LTmake_dataset(HDF_file_handle, "Sinewave", HDF_D1ndims, D1dims, H5T_NATIVE_DOUBLE, in_arr);
	H5LTmake_dataset(HDF_file_handle, "FTSinewave", HDF_D1ndims, D1dims, comp_id, out_arr);


	fftw_execute(fftw_plan_c2r);

	for (int i = 0; i < N; ++i) {
		in_arr[i] /= N;
	}

	H5LTmake_dataset(HDF_file_handle, "IFTSinewave", HDF_D1ndims, D1dims, H5T_NATIVE_DOUBLE, in_arr);




	// Destroy plan
	fftw_destroy_plan(fftw_plan_r2c);
	fftw_destroy_plan(fftw_plan_c2r);

	fftw_free(in_arr);
	fftw_free(out_arr);


	// must close HDf5 handle otherwise seg fault
	H5Fclose(HDF_file_handle);

	return 0;
}