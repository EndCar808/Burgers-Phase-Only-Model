// Enda Carroll
// Sept 2019
// Programme to run through some examples using FFTW on complex 2d and 3d arrays

// ---------------------------------------------------------------------
//  Standard Libraries and Headers
// ---------------------------------------------------------------------
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <complex.h> // having this before <fftw3.h> means fftw_complex is the native double-precision complex
					 // type and you can manipulate it with ordinary arithmetic. Otherwise FFTW defines its own complex type


// ---------------------------------------------------------------------
//  User Libraries and Headers
// ---------------------------------------------------------------------
#include <fftw3.h> // include the latest FFTW library


 
// ---------------------------------
//  Main Programme
// ---------------------------------
int main(int argc, char** argv) {

	// size of arrays
	int X = 5; 
	int Y = 5;
	int Z = 2;

	// Allocate 2d memory - FFTW trasnforms operate on contiguous C-style(row major) order.
	fftw_complex* in2d;
	fftw_complex* out2d;	
	in2d  = (fftw_complex* )fftw_malloc(X*Y*sizeof(fftw_complex)); // must cast as (ffftw_complex* )
	out2d = (fftw_complex* )fftw_malloc(X*Y*sizeof(fftw_complex));

	// Allocate 3d memory - FFTW trasnforms operate on contiguous C-style(row major) order.
	fftw_complex* in3d;
	fftw_complex* out3d;	
	in3d  = (fftw_complex* )fftw_malloc(X*Y*Z*sizeof(fftw_complex)); // must cast as (ffftw_complex* )
	out3d = (fftw_complex* )fftw_malloc(X*Y*Z*sizeof(fftw_complex));


	// create fftw plan to store the type of FFT we want to perform 
	fftw_plan my_plan_2d, my_plan_3d;



	// Specify the type of plan we want to perform
	/*
	Function inputs:
		X, Y, Z - dimension size of the arrays
		inXd - input array
		outXd - output array
				  These arrays can be equal. If so this indicates a 'transform in place' and the input array is overwritten with the output
		FFTW_FORWARD - determines to sign of the exponent of the DFT - FFTW_FORWARD = -1; FFTW_BACKWARD = +1;
		FFTW_ESTIMATE - flag to determine optimization - is typically FFTW_ESTIMATE or FFTW_MEASURE(can be others); FFTW_MEASURE performs a check
		by running different dft's to find the fastest out for the given N - this is takes a few seconds but is useful if you need to 
		perform many dft's throughout the programme. FFTW_ESTIMATE does not perform any check
	NOTE: You should always create plans before initializing arrays - FFTW_MEASURE will overwrite these arrays 
	 */
	my_plan_2d = fftw_plan_dft_2d(X, Y, in2d, out2d, FFTW_FORWARD, FFTW_ESTIMATE);
	my_plan_3d = fftw_plan_dft_3d(X, Y, Z, in3d, out3d, FFTW_FORWARD, FFTW_ESTIMATE);


	// fill 2d array
	int temp, index;
	printf("\nx_2d:\n\n");
	for(int i = 0; i < X; ++i) {
		temp = i*Y;
		for (int j = 0; j < Y; ++j) {
			index = temp + j;
			in2d[index] = exp(-index);
			printf("%d %11.7f %11.7f ", index, creal(in2d[index]), cimag(in2d[index]));
		}	
		printf("\n");	
	}

	// // fill 3d array
	// int temp1, temp2, index;
	// printf("\nx_k:\n\n");
	// for(int i = 0; i < X; ++i) {
	// 	temp1 = i*Y;
	// 	for (int j = 0; j < Y; ++j) {
	// 		temp2 = temp1 + j;
	// 		for (int k = 0; k < Z; ++k)
	// 		{	
	// 			index
	// 			in3d[index] = exp(-index);
	// 		printf("%d %11.7f %11.7f\n", index, creal(in[index]), cimag(in[index]));
	// 		}
			
	// 	}		
	// }



	// execute plan - essentially performs the dft outlined in the plan
	fftw_execute(my_plan_2d); // can perform this multiple times
	fftw_execute(my_plan_3d);

	printf("\nx_2d:\n\n");
	for(int i = 0; i < X; ++i) {
		temp = i*Y;
		for (int j = 0; j < Y; ++j) {
			index = temp + j;
			
			printf("%d %11.7f %11.7f ", index, creal(out2d[index]), cimag(out2d[index]));
		}	
		printf("\n");	
	}





	///////////
	/// Can also perform multidimensional DFTs or achieve the same as above
	/// using the general complex dft function. We need to know the number of dimensions 
	/// and an array of their sizes.
	/// 
	/// Here we perform the same tranforms as above
	int rank = 2;
	int dim[rank];
	dim[0] = X; 
	dim[1] = Y;


	fftw_plan my_plan_nd;


	my_plan_nd = fftw_plan_dft(rank, dim, in2d, out2d, FFTW_FORWARD, FFTW_ESTIMATE);

	fftw_execute(my_plan_nd);

	printf("\nx_nd:\n\n");
	for(int i = 0; i < X; ++i) {
		temp = i*Y;
		for (int j = 0; j < Y; ++j) {
			index = temp + j;
			
			printf("%d %11.7f %11.7f ", index, creal(out2d[index]), cimag(out2d[index]));
		}	
		printf("\n");	
	}

	// Destroy plan
	fftw_destroy_plan(my_plan_2d);
	fftw_destroy_plan(my_plan_3d);

	fftw_free(in2d);
	fftw_free(out2d);	
	fftw_free(in3d);
	fftw_free(out3d);

	return 0;
}