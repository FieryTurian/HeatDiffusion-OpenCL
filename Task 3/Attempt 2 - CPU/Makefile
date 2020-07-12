# Makefile in order to make an executable called relax

OPENCL        := /opt/AMDAPPSDK-3.0
#OPENCL        := /opt/intel/system_studio_2020/opencl/SDK

# C flags with strictest warnings.
CFLAGS        += -O3 -Wall -g -Wextra -I$(OPENCL)/include -std=c99 -D_GNU_SOURCE

# Linker flags.
LDFLAGS += -L$(OPENCL)/lib/x86_64/sdk -L$(OPENCL)/lib64 -l OpenCL -lrt


all: relax

# Build a binary from C source.
simple.o: simple.c
	$(CC) $(CFLAGS) -std=c99 -c $^

relax: relax.c simple.o
	$(CC) $(CFLAGS) -std=c99 -o $@ $^ $(LDFLAGS)

# Remove the binary.
clean:
	$(RM) relax simple.o

