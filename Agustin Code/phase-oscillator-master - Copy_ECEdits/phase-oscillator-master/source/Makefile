all: 
	gfortran -fopenmp -I/usr/include -L/usr/include mod_kind.f90 mod_utilities.f90 mod_functions.f90 mod_oscillators.f90 mod_statistics.f90 main.f90 -o main.exe -lfftw3_omp -lfftw3 -lm

clean: 
	rm *.exe
	rm *.mod
	rm *.smod
