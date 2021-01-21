HIST := $(shell mkdir -p histograms)
INST := $(shell mkdir -p instances)
PARA := $(shell mkdir -p parameters)
PROG := $(shell mkdir -p progress)
REAL := $(shell mkdir -p realspace)
STAT := $(shell mkdir -p statistics)


all: 
	gfortran -O3 -fdefault-real-8 -fopenmp -I/usr/include -L/usr/include mod_kind.f90 mod_utilities.f90 mod_functions.f90 mod_oscillators.f90 mod_statistics.f90 main.f90 -o main.exe -lfftw3_omp -lfftw3 -lm


clean: 
	rm *.exe
	rm *.mod
	rm *.smod
	rm -rf histograms
	rm -rf instances
	rm -rf parameters
	rm -rf progress
	rm -rf realspace
	rm -rf statistics