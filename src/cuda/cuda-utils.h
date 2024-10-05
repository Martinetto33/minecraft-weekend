#ifndef CUDA_UTILS_H
#define CUDA_UTILS_H

#include <curand.h>
#include <curand_kernel.h>

#define BLKDIM 1024

/* These files are only going to contain stuff that is strictly related
* to working with CUDA, and that does not regard the domain specifically. */

#ifdef __cplusplus
extern "C" {
#endif
    // Here 'blocks' referes to blocks in CUDA GPUs architecture
    curandState* init_random(unsigned long world_seed, long hash, int number_of_blocks, int threads_per_block);
    int calculate_gpu_blocks(int number_of_blocks_to_generate);

#ifdef __cplusplus
}
#endif

#endif //CUDA_UTILS_H
