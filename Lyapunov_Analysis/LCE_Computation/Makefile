# Enda Carroll
# May 2020
# Makfile to compile programmes for computing the Lyapunov Spectrum and vectors
# of the Phase Only Burgers equation

 
# ---------------------------------------------------------------------
#  Compiler flags
# ---------------------------------------------------------------------
GCC := gcc
CCFLAG := -O3 -W -Wall
LIBFLAG := -fopenmp -lpthread -lfftw3_omp -lfftw3 -lm -lhdf5 -lhdf5_hl
GSLFLAG := -lgsl -lgslcblas
LAPACKFLAG := -llapacke -llapack -lblas -lgfortran
INCDIR = -I/usr/include/hdf5/serial/  -I/usr/include/gsl/ -I../Libraries/lapack-3.9.0/CBLAS/include/ -I../Libraries/lapack-3.9.0/LAPACKE/include/
LIBDIR = -L/usr/lib/x86_64-linux-gnu/hdf5/serial/ -L../Libraries/lapack-3.9.0/


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
	rm -rf $(OUT)