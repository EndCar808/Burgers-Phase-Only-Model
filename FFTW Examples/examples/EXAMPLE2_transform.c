/*---------------------------------------------------------------------------------
EXAMPLE2_transform.c, v1.0 March 23, 2010.
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
#include<math.h>
#include<complex.h>									//This library is declared before fftw3.h
#include<fftw3.h>

int main(void)
{
	int i;
	int Npoints = 10;
	fftw_complex *in, *out;
	fftw_plan plan;

	in = (fftw_complex*) fftw_malloc(sizeof(fftw_complex)*Npoints);			//allocating memory
	out = (fftw_complex*) fftw_malloc(sizeof(fftw_complex)*Npoints);		//allocating memory
	plan = fftw_plan_dft_1d(Npoints, in, out, FFTW_FORWARD, FFTW_ESTIMATE); 	//Here we set which kind of transformation we want to perform

	printf("\nCoefficcients of the expansion:\n\n");
	for(i = 0; i < Npoints; i++)
	{
		in[i] = (i+1.)+(3.*i-1.)*I;
		printf("%d %11.7f %11.7f\n", i, creal(in[i]), cimag(in[i]));		//creal and cimag are functions of complex.h 
	}
	printf("\n");

	fftw_execute(plan); 								//Execution of FFT

	printf("Output:\n\n");
	for(i = 0; i < Npoints; i++)
	{
		printf("%d %11.7f %11.7f\n", i, creal(out[i]), cimag(out[i]));
	}

	printf("\nCoefficcients of the expansion:\n\n");
	for(i = 0; i < Npoints; i++)
	{
		in[i] = exp(-i);
		printf("%d %11.7f %11.7f\n", i, creal(in[i]), cimag(in[i]));		//creal and cimag are functions of complex.h 
	}
	printf("\n");

	fftw_execute(plan); 								//Execution of FFT

	printf("Output:\n\n");
	for(i = 0; i < Npoints; i++)
	{
		printf("%d %11.7f %11.7f\n", i, creal(out[i]), cimag(out[i]));
	}

	
	fftw_destroy_plan(plan);							//Destroy plan
	fftw_free(in);			 						//Free memory
	fftw_free(out);			 						//Free memory
	return 0;
}
