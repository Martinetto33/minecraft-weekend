UNAME_S = $(shell uname -s)

# You can change this by running `make CHUNKS_SIZE=N', where N is a positive integer.
# For some values of N (such as values greater than 64), the program may go
# in segmentation fault due to too many chunks being generated.
# You need to run `make clean' before being able to change this variable.
CHUNKS_SIZE ?= 16

C_COMPILER = clang
CUDA_COMPILER = nvcc
CFLAGS = -std=c11 -O3 -g -Wall -Wextra -Wpedantic -Wstrict-aliasing
CFLAGS += -Wno-pointer-arith -Wno-newline-eof -Wno-unused-parameter -Wno-gnu-statement-expression
CFLAGS += -Wno-gnu-compound-literal-initializer -Wno-gnu-zero-variadic-macro-arguments
CFLAGS += -Ilib/cglm/include -Ilib/glad/include -Ilib/glfw/include -Ilib/stb -Ilib/noise -fbracket-depth=1024
CFLAGS += -Wno-error=implicit-function-declaration
# Adding the environmental variable ALIN_CHUNKS_SIZE as a compilation flag; it's used in the world/world.c file
CFLAGS += -DALIN_CHUNKS_SIZE=$(CHUNKS_SIZE)
LDFLAGS = lib/glad/src/glad.o lib/cglm/libcglm.a lib/glfw/src/libglfw3.a lib/noise/libnoise.a -lm

CUDA_FLAGS = -g -G
CUDA_FLAGS += -Ilib/cglm/include -Ilib/glad/include -Ilib/glfw/include -Ilib/stb -Ilib/noise #-fbracket-depth=1024
CUDA_FLAGS += -DALIN_CHUNKS_SIZE=$(CHUNKS_SIZE)
# GLFW required frameworks on OSX
ifeq ($(UNAME_S), Darwin)
	LDFLAGS += -framework OpenGL -framework IOKit -framework CoreVideo -framework Cocoa
endif

ifeq ($(UNAME_S), Linux)
	LDFLAGS += -ldl -lpthread
endif

SRC  = $(wildcard src/**/*.c) $(wildcard src/*.c) $(wildcard src/**/**/*.c) $(wildcard src/**/**/**/*.c)
OBJ  = $(SRC:.c=.o)
CUDA_SRC = $(wildcard src/cuda/*.cu) $(wildcard src/cuda/noise/*.cu)
CUDA_OBJ = $(CUDA_SRC:.cu=.o)
OBJ += $(CUDA_OBJ)
BIN = bin

.PHONY: all clean

all: dirs libs game

libs:
	cd lib/cglm && cmake . -DCGLM_STATIC=ON && make
	cd lib/glad && $(C_COMPILER) -o src/glad.o -Iinclude -c src/glad.c
	cd lib/glfw && cmake . && make
	cd lib/noise && make

dirs:
	mkdir -p ./$(BIN)

run: all
	$(BIN)/game

game: $(OBJ) gpuCode.o
	$(CUDA_COMPILER) -o $(BIN)/game $^ $(LDFLAGS)

# Rule to link separately the gpu-code
gpuCode.o: $(CUDA_OBJ)
	$(CUDA_COMPILER) -arch=sm_75 -dlink $(CUDA_OBJ) -o gpuCode.o -L/opt/cuda/nvvm/lib64 -lcudart

# Rule for compiling CUDA files with nvcc; the -x option specifies the language to use for compilation (cu means CUDA)
%.o: %.cu
	$(CUDA_COMPILER) -x cu -dc $< -o $@ $(CUDA_FLAGS)

%.o: %.c
	$(C_COMPILER) -o $@ -c $< $(CFLAGS)

clean:
	rm -rf $(BIN) $(OBJ)
