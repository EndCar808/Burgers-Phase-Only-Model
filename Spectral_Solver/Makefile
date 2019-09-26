# Enda Carroll
# Sept 2019
# Makfile to compile solver programme for Burgers equation

 
# ---------------------------------------------------------------------
#  Compiler flags
# ---------------------------------------------------------------------
GCC := gcc



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
all: PDE_Burgers


PDE_Burgers: $(SRCOBJ)
	$(GCC) -o bin/PDE_Burgers $^ 

$(OBJBIN)/%.o: $(SRCDIF)/%.c
	$(GCC) -o $@ $<



# ---------------------------------------------------------------------
#  Clean up
# ---------------------------------------------------------------------
clean:
	rm -rf $(OBJBIN)
	rm -rf $(BIN)
	rm -rf $(OUT)