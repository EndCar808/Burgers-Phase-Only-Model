# Enda Carroll
# Sept 2019
# Makfile to compile solver programme for Burgers equation

 
# ---------------------------------------------------------------------
#  Compiler flags
# ---------------------------------------------------------------------
GCC := gcc
CCFLAG := -O3
LIBFLAG := -lfftw3 -lm -lhdf5 -lhdf5_hl
INCDIR = -I/usr/include/hdf5/serial/  -I/usr/include/gsl/	## these are the direcotry additions necessary when using software on UBUNTU desktop
LIBDIR = -L/usr/lib/x86_64-linux-gnu/hdf5/serial/

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
SRCOBJ := $(patsubst $(SRCDIR)/%.c, $(OBJBIN)/%.o, $(SRCFILE))


# ---------------------------------------------------------------------
#  Builds
# ---------------------------------------------------------------------
############
.PHONY: all
############


all: $(BIN)/main



$(BIN)/main: $(SRCOBJ)
	$(GCC) $(CCFLAG) -o $@ $^ ${INCDIR} ${LIBDIR} $(LIBFLAG)

$(OBJBIN)/%.o: $(SRCDIR)/%.c
	$(GCC) $(CCFLAG) -c $^ -o $@ ${INCDIR} ${LIBDIR} $(LIBFLAG)

# $(OBJBIN)/main.o: $(SRCDIR)/main.c 
# 	$(GCC) -c $^ -o $@ $(LIBFLAG)

# $(OBJBIN)/utils.o: src/utils.c
# 	$(GCC) -c $^ -o $@ $(LIBFLAG)


# ---------------------------------------------------------------------
#  Clean up
# ---------------------------------------------------------------------
clean:
	rm -rf $(OBJBIN)
	rm -rf $(BIN)
	rm -rf $(OUT)