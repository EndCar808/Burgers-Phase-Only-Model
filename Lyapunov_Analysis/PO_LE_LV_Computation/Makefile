# Enda Carroll
# May 2020
# Makfile to compile programmes for computing the Lyapunov Spectrum and vectors
# of the Phase Only Burgers equation

 
# ---------------------------------------------------------------------
#  Compiler flags
# ---------------------------------------------------------------------
OS = $(shell lsb_release -si)

ifeq ($(OS),Ubuntu)
	# CHIRP
	GCC := gcc
	CCFLAG := -O3 -W -Wall
	LIBFLAG := -fopenmp -lpthread -lfftw3_omp -lfftw3 -lm -lhdf5 -lhdf5_hl
	GSLFLAG := -lgsl -lgslcblas
	LAPACKFLAG := -llapacke -llapack -lblas -lgfortran
	INCDIR = -I/usr/include/hdf5/serial/  -I/usr/include/gsl/ -I../../Libraries/lapack-3.9.0/CBLAS/include/ -I../../Libraries/lapack-3.9.0/LAPACKE/include/
	LIBDIR = -L/usr/lib/x86_64-linux-gnu/hdf5/serial/ -L../../Libraries/lapack-3.9.0/
else ifeq ($(OS), RedHatEnterpriseServer)
	# SONIC
	GCC := gcc
	CCFLAG := -O3 -W -Wall -std=c11
	LIBFLAG := -fopenmp -lpthread -lfftw3_threads -lfftw3 -lm -lhdf5 -lhdf5_hl
	GSLFLAG := -lgsl -lgslcblas
	LAPACKFLAG := -llapacke -llapack -lblas -lgfortran
	INCDIR = -I/opt/software/fftw/3.3.8/include/ -I/opt/software/intel/2019u5/compilers_and_libraries_2019.5.281/linux/mpi/intel64/include -I/opt/software/gsl/include/gsl/ -I/opt/software/hdf5/1.10.5/include/ -I/home/people/16203944/Burgers/burgers-code/Libraries/lapack-3.9.0/CBLAS/include -I/home/people/16203944/Burgers/burgers-code/Libraries/lapack-3.9.0/LAPACKE/include
	LIBDIR = -L/home/people/16203944/Burgers/burgers-code/Libraries/lapack-3.9.0/ -L/opt/software/fftw/3.3.8/lib/ -L/opt/software/hdf5/1.10.5/lib/
else ifeq ($(OS), CentOS)
	# KAY (ICHEC)
	GCC := icc
	CCFLAG := -O3 -W -Wall -std=c11
	LIBFLAG := -fopenmp -lpthread -lfftw3_omp -lfftw3 -lm -lhdf5 -lhdf5_hl
	GSLFLAG := -lgsl -lgslcblas
	LAPACKFLAG := -llapacke -llapack -lblas -lgfortran
	INCDIR = -I/ichec/packages/fftw/3.3.8/intel/double/include -I/ichec/packages/hdf5/intel/1.10.1/include -I/ichec/packages/lapack/3.8.0/gcc/static/include/ 
	LIBDIR = -L/ichec/packages/lapack/3.8.0/gcc/static/lib64 -L/ichec/packages/fftw/3.3.8/intel/double/lib -L/ichec/packages/hdf5/intel/1.10.1/lib
else
	$(error Unknown OS found, please check OS and add appropriate flags in Makefile)
endif

DEPS = $(wildcard *.h) 


# ---------------------------------------------------------------------
#  Directory creation
# ---------------------------------------------------------------------
OBJBIN := obj
OBJDIR := $(shell mkdir -p $(OBJBIN))

BIN := bin
BINDIR := $(shell mkdir -p $(BIN))

OUT := output
OUTDIR := $(shell mkdir -p $(OUT))

SRCDIR := src
SRCFILE := $(wildcard $(SRCDIR)/*.c)

# Construct appropritate object string list from filtered src file list
SRCOBJ_LCE := $(patsubst $(SRCDIR)/%.c, $(OBJBIN)/%.o, $(SRCFILE))
# ---------------------------------------------------------------------
#  Builds
# ---------------------------------------------------------------------
############
.PHONY: all
############


all: $(BIN)/main



############
#  LCE
############
lce: $(BIN)/main 

$(BIN)/main: $(SRCOBJ_LCE) $(DEPS)
	$(GCC) $(CCFLAG) -o $@ $^ ${INCDIR} ${LIBDIR}  $(GSLFLAG) $(LAPACKFLAG) $(LIBFLAG) 

$(OBJBIN)/%.o: $(SRCDIR)/%.c $(DEPS)
	$(GCC) $(CCFLAG) -c $^ -o $@ ${INCDIR} ${LIBDIR}  $(GSLFLAG) $(LAPACKFLAG) $(LIBFLAG) 






# ---------------------------------------------------------------------
#  Clean up
# ---------------------------------------------------------------------
clean:
	rm -rf $(OBJBIN)
	rm -rf $(BIN)
# 	rm -rf $(OUT)