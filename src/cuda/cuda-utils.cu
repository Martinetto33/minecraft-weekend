#include "hpc.h"
#include "cuda-utils.h"
#include <stdio.h>
#include <stdlib.h>


#ifdef __cplusplus
extern "C" {
#endif

    // See https://stackoverflow.com/questions/18501081/generating-random-number-within-cuda-kernel-in-a-varying-range
    // Call this function once per iteration to initialise random generation
    __global__ void init_random_kernel(unsigned long world_seed, long hash, curandState *states) {
        const int my_id = blockIdx.x * blockDim.x + threadIdx.x;
        curand_init(world_seed + hash, my_id, 0, &states[my_id]);
    }

    __global__ void generate_kernel(curandState *states, unsigned long *random_longs, int output_length) {
        const int my_id = blockIdx.x * blockDim.x + threadIdx.x;
        // Always check for out of bound accesses
        if (my_id < output_length) {
          random_longs[my_id] = curand(&states[my_id]);
        }
    }

    curandState* init_random(unsigned long world_seed, long hash, int number_of_blocks, int threads_per_block) {
        curandState *d_States;
        const int total_threads = threads_per_block * number_of_blocks;
        cudaSafeCall(cudaMalloc((void**)&d_States, total_threads * sizeof(curandState)));
        init_random_kernel<<<number_of_blocks, threads_per_block>>>(world_seed, hash, d_States);
        return d_States;
    }

    void test_random() {
      unsigned long world_seed = 12345;
      long hash = 15;
      curandState* states = init_random(world_seed, hash, 10, 10);
      unsigned long *h_random_longs, *d_random_longs;
      h_random_longs = (unsigned long*)malloc(sizeof(unsigned long) * 10 * 10);
      cudaSafeCall(cudaMalloc((void**)&d_random_longs, 10 * 10 * sizeof(unsigned long)));
      generate_kernel<<<10, 10>>>(states, d_random_longs, 10 * 10);
      cudaSafeCall(cudaMemcpy(h_random_longs, d_random_longs, 10 * 10 * sizeof(unsigned long), cudaMemcpyDeviceToHost));
      for (int i = 0; i < 10 * 10; i++) {
        printf("h_random_longs[%d] = %lu\n", i, h_random_longs[i]);
      }
    }

#ifdef __cplusplus
}
#endif
