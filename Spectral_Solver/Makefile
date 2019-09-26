# Enda Carroll
# Sept 2019
# Makfile to compile solver programme for Burgers equation

 
# ---------------------------------------------------------------------
#  Compiler flags
# ---------------------------------------------------------------------
GCC := gcc
LIBFLAG := -lfftw3 -lm 
# OPTFLAG := -03



# ---------------------------------------------------------------------
#  Directory creation
# ---------------------------------------------------------------------
OBJBIN := obj
OBJDIR := $(shell mkdir -p $(OBJBIN))

BIN := bin
BINDIR := $(shell mkdir -p $(BIN))

OUT := output
OUTDIR := $(shell mkdir -p $(OUT))

SRCDIF := src
SRCFILE := $(wildcard $(SRCDIF)/*.c)
SRCOBJ := $(patsubst $(SRCDIF)/%.c, $(OBJBIN)/%.o, $(SRCFILE))



# ---------------------------------------------------------------------
#  Builds
# ---------------------------------------------------------------------
all: main


# main: 
# 	$(GCC) -o bin/PDE_Burgers src/main.c $(LIBFLAG) $(OPTFLAG)

main: $(SRCOBJ)
	$(GCC) -o bin/main $^ $(LIBFLAG) 

$(OBJBIN)/%.o: $(SRCDIF)/%.c
	$(GCC) -o $@ $< $(LIBFLAG) 



# ---------------------------------------------------------------------
#  Clean up
# ---------------------------------------------------------------------
clean:
	rm -rf $(OBJBIN)
	rm -rf $(BIN)
	rm -rf $(OUT)