/*---------------------------------------------------------------------------------
EXAMPLE4_product.c, v1.0 March 23, 2010.
Author: Jordi-Lluis Figueras

This is an example code associated to the tutorial 
"A short tutorial on the basic usage of the package \textbf{FFTW3}.", 
the author of which is Jordi-Lluis Figueras.

Copyright (C) 2010 Jordi-Lluis Figueras 

The author, who retains the copyright of this software, permits the distribution of 
this software if it will be used for research or educational profit. 

The distribution to third parties and modification of this software is permitted 
if the copyright stated in this header file is also copied and the authorship of the
whole or part of the code, used from this source, is atributed to the author of this 
file.

This code and information are provided "as is" without any warranty. The author does
not claim nor guarantee the suitability of it. Any possible damage produced by this
software can not be charged to the author.

Barcelona, March 2010.

Contact information:
	Jordi-Lluis Figueras
	Departament de Matematica Aplicada i Analisi
	Universitat de Barcelona
	Gran Via 585
	08007 Barcelona, Spain
	E-mail: figueras@maia.ub.es
	Web: http://www.maia.ub.es/~figueras

The author of this sotware kindly aknowledge FFTW team and its authors. 
Visit 
	www.fftw.org 
or read 
	Matteo Frigo and Steven G. Johnson, “The design and implementation of FFTW3,” 
	Proc. IEEE 93 (2), 216–231 (2005). 
for more details.

---------------------------------------------------------------------------------*/


#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<complex.h>
#include<time.h>
#include<fftw3.h>


int main(int argc, char* argv[])
{
	if(argc < 2)
	{
		printf("Usage:\n\t[1]Number of equidistant points\n");
		return 1;
	}
	int Npointspol = atoi(argv[1]);
	int Npoints = 2*Npointspol-1;
	int i;
	fftw_complex *in, *out;
	fftw_complex *in1, *out1;
	fftw_complex *in2, *out2;
	fftw_plan planback;
	fftw_plan planfor;

	in = (fftw_complex*) fftw_malloc(sizeof(fftw_complex)*Npoints);			//pay attention
	out = (fftw_complex*) fftw_malloc(sizeof(fftw_complex)*Npoints);		//pay attention
	in1 = (fftw_complex*) fftw_malloc(sizeof(fftw_complex)*Npoints);		//pay attention
	out1 = (fftw_complex*) fftw_malloc(sizeof(fftw_complex)*Npoints);		//pay attention
	in2 = (fftw_complex*) fftw_malloc(sizeof(fftw_complex)*Npoints);		//pay attention
	out2 = (fftw_complex*) fftw_malloc(sizeof(fftw_complex)*Npoints);		//pay attention
	planback = fftw_plan_dft_1d(Npoints, in, out, FFTW_BACKWARD, FFTW_ESTIMATE); 	//Here we set which kind of transformation we want to perform
	planfor = fftw_plan_dft_1d(Npoints, in, out, FFTW_FORWARD, FFTW_ESTIMATE); 	//Here we set which kind of transformation we want to perform

	srand(time(NULL));
	for(i = 0; i < Npointspol; i++)
	{
		in1[i] = rand()/(RAND_MAX+1.);
		in2[i] = rand()/(RAND_MAX+1.);
	}
	printf("Polynomial  coefficients:\n");
	for(i = 0; i < Npointspol; i++)
	{
		printf("%2d %11.7f %11.7f\n", i, creal(in1[i]), creal(in2[i]));
	}
	for(i = Npointspol; i < Npoints; i++)
	{
		in1[i] = 0.;
		in2[i] = 0.;
	}

	for(i = 0; i < Npoints; i++)
	{
		in[i] = in1[i];
	}
	fftw_execute(planback);								//Execution of FFT
	for(i = 0; i < Npoints; i++)
	{
		out1[i] = out[i];
		in[i] = in2[i];
	}
	fftw_execute(planback); 							//Execution of FFT
	for(i = 0; i < Npoints; i++)
	{
		out2[i] = out[i];
	}

	for(i = 0; i < Npoints; i++)
	{
		in[i] = out1[i]*out2[i];
	}
	fftw_execute(planfor); 								//Execution of FFT

	printf("\nPolynomial coefficients of the product:\n");
	for(i = 0; i < Npoints; i++)
	{
		printf("%2d %11.7f\n", i, creal(out[i])/(double)Npoints);
	}
	
	fftw_destroy_plan(planback);							//Free memory
	fftw_destroy_plan(planfor);							//Free memory
	fftw_free(in);									//Free memory
	fftw_free(in1);									//Free memory
	fftw_free(in2);									//Free memory
	fftw_free(out);									//Free memory
	fftw_free(out1);								//Free memory
	fftw_free(out2);								//Free memory
	fftw_cleanup();									//Free all the remaining memory
	return 0;
}

