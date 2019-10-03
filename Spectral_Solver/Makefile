# Enda Carroll
# Sept 2019
# Makfile to compile solver programme for Burgers equation

 
# ---------------------------------------------------------------------
#  Compiler flags
# ---------------------------------------------------------------------
GCC := gcc
LIBFLAG := -lfftw3 -lm 
# OPTFLAG := -03
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
	$(GCC) -o $@ $^ $(LIBFLAG)

$(OBJBIN)/%.o: $(SRCDIR)/%.c
	$(GCC) -c $^ -o $@ $(LIBFLAG)

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