# PROJECT = saxpy

# SRCS = main.cc

# OPTS ?= -n1024

# include ../common/common.mk

include ../common/make.config
TYPE = GPU
CC = clang++
CC_FLAGS = -g -O2 -std=c++11
CC_FLAGS += -I.
EXE = vecadd.out

.PHONY: $(EXE)
vecadd.out: 
	$(CC) $(CC_FLAGS) vecadd.c -o $(EXE) -I$(OPENCL_INC) -L$(OPENCL_LIB) -lOpenCL -Wno-unused-result

%.o: %.[ch]
	$(CC) $(CC_FLAGS) $< -c

clean:
	rm -f *.o *~ *.out *.linkinfo