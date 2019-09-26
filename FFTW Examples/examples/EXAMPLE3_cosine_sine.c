/*---------------------------------------------------------------------------------
EXAMPLE3_cosine_sine.c, v1.0 March 22, 2010.
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

inline double function1(double, double*);

int main(int argc, char* argv[])
{
	if(argc < 2)
	{
		printf("Usage:\n\t[1]Number of equidistant points\n");
		return 1;
	}
	int Npoints = atoi(argv[1]);
	double *coef;
	double *xpoints;
	double t, delta;
	int i;

	srand(time(NULL));
	coef = (double*) malloc(5*sizeof(double));
	xpoints = (double*) malloc(Npoints*sizeof(double));

	printf("\nCoefficcients of the expansion\n    cosine\tsine\n");
	coef[0] = 2.*rand()/(RAND_MAX+1.)-1.;
	printf("%d %11.7f %11.7f\n", 0, coef[0], 0.);
	coef[1] = 2.*rand()/(RAND_MAX+1.)-1.;
	printf("%d %11.7f ", 1, coef[1]);
	coef[2] = 2.*rand()/(RAND_MAX+1.)-1.;
	printf("%11.7f \n", coef[2]);
	coef[3] = 2.*rand()/(RAND_MAX+1.)-1.;
	printf("%d %11.7f ", 3, coef[3]);
	coef[4] = 2.*rand()/(RAND_MAX+1.)-1.;
	printf("%11.7f \n", coef[4]);
	
	printf("\n");
	delta = 1./(double)Npoints;
	t = 0.;
	for(i = 0; i < Npoints; i++)
	{
		xpoints[i] = function1(t, coef);
		t += delta;
	}

	fftw_complex *in, *out;
	fftw_plan plan;

	in = (fftw_complex*) fftw_malloc(sizeof(fftw_complex)*Npoints);			//pay attention
	out = (fftw_complex*) fftw_malloc(sizeof(fftw_complex)*Npoints);		//pay attention
	plan = fftw_plan_dft_1d(Npoints, in, out, FFTW_FORWARD, FFTW_ESTIMATE); 	//Here we set which kind of transformation we want to perform

	for(i = 0; i < Npoints; i++)
	{
		in[i] = xpoints[i];
	}
	fftw_execute(plan); //Execution of FFT

	printf("\nCosine-Sine coefficients:\n    cosine\tsine\n");
	printf("0 %11.7f %11.7f\n", creal(out[0])/Npoints, 0.);
	for(i = 1; i < Npoints; i++)
	{
		printf("%d %11.7f %11.7f\n", i, creal(out[i]+out[Npoints-i])/Npoints, cimag(out[Npoints-i]-out[i])/Npoints); 
	}
	printf("\nExponential coefficients:\n");
	for(i = 0; i < Npoints; i++)
	{
		printf("%d %11.7f %11.7f\n", i, creal(out[i])/Npoints, cimag(out[i])/Npoints);
	}

	
	fftw_destroy_plan(plan);	 //Free memory
	fftw_free(in);			 //Free memory
	fftw_free(out);			 //Free memory
	free(coef);
	free(xpoints);
	return 0;
}

inline double function1(double t, double *coef)
{
	return (coef[0]+coef[1]*cos(2.*M_PI*t)+coef[2]*sin(2.*M_PI*t)+coef[3]*cos(4.*M_PI*t)+coef[4]*sin(4.*M_PI*t));
}
