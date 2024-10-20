#include <cassert>

#include "hpc.h"
#include "cuda-worldgen.h"
#include "cuda-utils.h"
#include <curand.h>
#include "noise/cuda-noise.cuh"

#define BLKDIM 512

#ifdef __cplusplus
extern "C" {
#endif
#ifndef EPSILON
#define EPSILON 0.000001f // for float comparison
#endif
#define N_H 0
#define N_M 1
#define N_T 2
#define N_R 3
#define N_N 4
#define N_P 5
#define WATER_LEVEL 0
#ifndef safe_expf
#define safe_expf(_x, _e) ({ __typeof__(_x) __x = (_x); __typeof__(_e) __e = (_e); sign(__x) * fabsf(powf(fabsf(__x), __e)); })
#endif
#ifndef clamp
#define clamp(x, mn, mx) ({\
__typeof__ (x) _x = (x); \
__typeof__ (mn) _mn = (mn); \
__typeof__ (mx) _mx = (mx); \
max(_mn, min(_mx, _x)); })
#endif

    bool was_environment_initialised = false;
    CudaBasic **bs = nullptr;
    CudaOctave **os = nullptr;
    CudaCombined **cs = nullptr;
    CudaExpScale **expscales = nullptr;

    __device__ static int sign(const float n) {
        return (n > EPSILON) - (n < -EPSILON);
    }

#define BIOME_LAST MOUNTAIN
    enum CudaBiome {
        OCEAN,
        RIVER,
        BEACH,
        DESERT,
        SAVANNA,
        JUNGLE,
        GRASSLAND,
        WOODLAND,
        FOREST,
        RAINFOREST,
        TAIGA,
        TUNDRA,
        ICE,
        MOUNTAIN
    };

#define MAX_DECORATIONS 8
    struct CudaDecoration {
        //FWGDecorate f;
        float chance;
    };

    struct CudaBiomeData {
        CudaBlockId top_block, bottom_block;
        float roughness, scale, exp;
        CudaDecoration decorations[MAX_DECORATIONS];
    };

    CudaBiomeData CUDA_BIOME_DATA[BIOME_LAST + 1] = {
    [OCEAN] = {
        .top_block = CUDA_SAND,
        .bottom_block = CUDA_SAND,
        .roughness = 1.0f,
        .scale = 1.0f,
        .exp = 1.0f
    },
    [RIVER] = {
        .top_block = CUDA_SAND,
        .bottom_block = CUDA_SAND,
        .roughness = 1.0f,
        .scale = 1.0f,
        .exp = 1.0f
    },
    [BEACH] = {
        .top_block = CUDA_SAND,
        .bottom_block = CUDA_SAND,
        .roughness = 0.2f,
        .scale = 0.8f,
        .exp = 1.3f
    },
    [DESERT] = {
        .top_block = CUDA_SAND,
        .bottom_block = CUDA_SAND,
        .roughness = 0.6f,
        .scale = 0.6f,
        .exp = 1.2f,
        .decorations = {
            { /*.f = worldgen_shrub,*/ .chance = 0.005f }
        }
    },
    [SAVANNA] = {
        .top_block = CUDA_GRASS,
        .bottom_block = CUDA_DIRT,
        .roughness = 1.0f,
        .scale = 1.0f,
        .exp = 1.0f,
        .decorations = {
            { /*.f = worldgen_tree,*/ .chance = 0.001f },
            { /*.f = worldgen_flowers,*/ .chance = 0.001f },
            { /*.f = worldgen_grass,*/ .chance = 0.005f },
        }
    },
    [JUNGLE] = {
        .top_block = CUDA_GRASS,
        .bottom_block = CUDA_DIRT,
        .roughness = 1.0f,
        .scale = 1.0f,
        .exp = 1.0f,
        .decorations = {
            { /*.f = worldgen_tree,*/ .chance = 0.01f },
            { /*.f = worldgen_flowers,*/ .chance = 0.001f },
            { /*.f = worldgen_grass,*/ .chance = 0.01f },
        }
    },
    [GRASSLAND] = {
        .top_block = CUDA_GRASS,
        .bottom_block = CUDA_DIRT,
        .roughness = 1.0f,
        .scale = 1.0f,
        .exp = 1.0f,
        .decorations = {
            { /*.f = worldgen_tree,*/ .chance = 0.0005f },
            { /*.f = worldgen_flowers,*/ .chance = 0.003f },
            { /*.f = worldgen_grass,*/ .chance = 0.02f },
        }
    },
    [WOODLAND] = {
        .top_block = CUDA_GRASS,
        .bottom_block = CUDA_DIRT,
        .roughness = 1.0f,
        .scale = 1.0f,
        .exp = 1.0f,
        .decorations = {
            { /*.f = worldgen_tree,*/ .chance = 0.007f },
            { /*.f = worldgen_flowers,*/ .chance = 0.003f },
            { /*.f = worldgen_grass,*/ .chance = 0.008f },
        }
    },
    [FOREST] = {
        .top_block = CUDA_GRASS,
        .bottom_block = CUDA_DIRT,
        .roughness = 1.0f,
        .scale = 1.0f,
        .exp = 1.0f,
        .decorations = {
            { /*.f = worldgen_tree,*/ .chance = 0.009f },
            { /*.f = worldgen_flowers,*/ .chance = 0.003f },
            { /*.f = worldgen_grass,*/ .chance = 0.008f },
        }
    },
    [RAINFOREST] = {
        .top_block = CUDA_GRASS,
        .bottom_block = CUDA_DIRT,
        .roughness = 1.0f,
        .scale = 1.0f,
        .exp = 1.0f,
        .decorations = {
            { /*.f = worldgen_tree,*/ .chance = 0.009f },
            { /*.f = worldgen_flowers,*/ .chance = 0.003f },
            { /*.f = worldgen_grass,*/ .chance = 0.008f },
        }
    },
    [TAIGA] = {
        .top_block = CUDA_PODZOL,
        .bottom_block = CUDA_DIRT,
        .roughness = 1.0f,
        .scale = 1.0f,
        .exp = 1.0f,
        .decorations = {
            { /*.f = worldgen_pine,*/ .chance = 0.006f },
            { /*.f = worldgen_flowers,*/ .chance = 0.001f },
            { /*.f = worldgen_grass,*/ .chance = 0.008f },
        }
    },
    [TUNDRA] = {
        .top_block = CUDA_SNOW,
        .bottom_block = CUDA_STONE,
        .roughness = 1.0f,
        .scale = 1.0f,
        .exp = 1.0f,
        .decorations = {
            { /*.f = worldgen_pine,*/ .chance = 0.0005f }
        }
    },
    [ICE] = {
        .top_block = CUDA_SNOW,
        .bottom_block = CUDA_STONE,
        .roughness = 1.0f,
        .scale = 1.0f,
        .exp = 1.0f
    },
    [MOUNTAIN] = {
        .top_block = CUDA_SNOW,
        .bottom_block = CUDA_STONE,
        .roughness = 2.0f,
        .scale = 1.2f,
        .exp = 1.0f
    },
};

    const CudaBiome BIOME_TABLE[6][6] = {
        { ICE, TUNDRA, GRASSLAND,   DESERT,     DESERT,     DESERT },
        { ICE, TUNDRA, GRASSLAND,   GRASSLAND,  DESERT,     DESERT },
        { ICE, TUNDRA, WOODLAND,    WOODLAND,   SAVANNA,    SAVANNA },
        { ICE, TUNDRA, TAIGA,       WOODLAND,   SAVANNA,    SAVANNA },
        { ICE, TUNDRA, TAIGA,       FOREST,     JUNGLE,     JUNGLE },
        { ICE, TUNDRA, TAIGA,       TAIGA,      JUNGLE,     JUNGLE }
    };

    const float HEAT_MAP[] = {
        0.05f,
        0.18f,
        0.4f,
        0.6f,
        0.8f
    };

    const float MOISTURE_MAP[] = {
        0.2f,
        0.3f,
        0.5f,
        0.6f,
        0.7f
    };

    /* Tables: */
	__device__ __constant__ CudaBiome device_biome_table[6 * 6];
    __device__ __constant__ CudaBiomeData device_biome_data[BIOME_LAST + 1];
    __device__ __constant__ float device_heat_map[5];
    __device__ __constant__ float device_moisture_map[5];

    __device__ CudaBiome get_biome(const float h, const float m, const float t, const float n, const float i) {
        if (h <= 0.0f || n <= 0.0f) {
            return OCEAN;
        }

        if (h <= 0.005f) {
            return BEACH;
        }

        if (n >= 0.1f && i >= 0.2f) {
            return MOUNTAIN;
        }

        size_t t_i = 0, m_i = 0;

        for (; t_i < 4; t_i++) {
            if (t <= device_heat_map[t_i]) {
                break;
            }
        }

        for (; m_i < 4; m_i++) {
            if (m <= device_moisture_map[m_i]) {
                break;
            }
        }

        const int device_biome_table_index = m_i * 6 + t_i; // 6 is the table size, m_i represents the row number, t_i the column
        return device_biome_table[device_biome_table_index];
    }

    __global__ void generate_noise(CudaBasic **bs, CudaOctave **os, CudaCombined **cs, CudaExpScale **expscales) {
        bs[0] = new CudaBasic(1);
        bs[1] = new CudaBasic(2);
        bs[2] = new CudaBasic(3);
        bs[3] = new CudaBasic(4);

        os[0] = new CudaOctave(5, 0);
        os[1] = new CudaOctave(5, 1);
        os[2] = new CudaOctave(5, 2);
        os[3] = new CudaOctave(5, 3);
        os[4] = new CudaOctave(5, 4);
        os[5] = new CudaOctave(5, 5);

        cs[0] = new CudaCombined(bs[0], bs[1]);
        cs[1] = new CudaCombined(bs[2], bs[3]);
        cs[2] = new CudaCombined(os[3], os[4]);
        cs[3] = new CudaCombined(os[1], os[2]);
        cs[4] = new CudaCombined(os[1], os[3]);

        expscales[N_H] = new CudaExpScale(os[0], 1.3f, 1.0f / 128.0f); // n_h
        expscales[N_M] = new CudaExpScale(cs[0], 1.0f, 1.0f / 512.0f); // n_m
        expscales[N_T] = new CudaExpScale(cs[1], 1.0f, 1.0f / 512.0f); // n_t
        expscales[N_R] = new CudaExpScale(cs[2], 1.0f, 1.0f / 16.0f); // n_r
        expscales[N_N] = new CudaExpScale(cs[3], 3.0f, 1.0f / 512.0f); // n_n
        expscales[N_P] = new CudaExpScale(cs[4], 3.0f, 1.0f / 512.0f); // n_p
    }



    void initialise_tables() {
        cudaSafeCall(cudaMemcpyToSymbol(device_biome_table, BIOME_TABLE, 6 * 6 * sizeof(CudaBiome), 0));
        cudaSafeCall(cudaMemcpyToSymbol(device_biome_data, CUDA_BIOME_DATA, (BIOME_LAST + 1) * sizeof(CudaBiomeData), 0));
        cudaSafeCall(cudaMemcpyToSymbol(device_heat_map, HEAT_MAP, 5 * sizeof(float), 0));
        cudaSafeCall(cudaMemcpyToSymbol(device_moisture_map, MOISTURE_MAP, 5 * sizeof(float), 0));
    }

    void destroy_noise_arrays(CudaBasic *bs, CudaOctave *os, CudaCombined *cs, CudaExpScale **expscales) {
      cudaFree(bs);
      cudaFree(os);
      cudaFree(cs);
      cudaFree(expscales);
    }

    __device__ __forceinline__ int get_index_from_coordinates(unsigned int x, unsigned int z, int chunk_size_x, int chunk_size_z) {
        return clamp(x, 0, chunk_size_x - 1) * chunk_size_x + clamp(z, 0, chunk_size_z - 1);
    }

    __device__ __forceinline__ void check_neighbour(const int heightmap_index) {
        const unsigned int first_index_in_next_block = (blockIdx.x + 1) * blockDim.x;
        const unsigned int first_index_in_this_block = blockIdx.x * blockDim.x;
        if (heightmap_index < first_index_in_this_block || heightmap_index >= first_index_in_next_block) {
            printf("[Thread %u] Heightmap index is out of bounds: first_index_in_this_block = %u, first_index_in_next_block = %u, received index: %d\n",
                blockIdx.x * blockDim.x + threadIdx.x, first_index_in_this_block, first_index_in_next_block, heightmap_index);
        }
    }

    __global__ void compute_worldgen_data_gpu(CUDA_WORLDGEN_DATA *data,
                                          const int chunk_size_x, const int chunk_size_z,
                                          const int number_of_chunk_columns,
                                          const int chunk_world_position_x, const int chunk_world_position_z,
                                          unsigned long world_seed,
                                          CudaExpScale **expscales) {
        if (const unsigned int my_id = blockIdx.x * blockDim.x + threadIdx.x; my_id < number_of_chunk_columns) {
            const long long wx = chunk_world_position_x + (static_cast<long long>(my_id) % chunk_size_x);
            const long long wz = chunk_world_position_z + (static_cast<long long>(my_id) / chunk_size_z);
            const float wx_f = __ll2float_rz(wx); // rounding long long int to 0 in CUDA way
            const float wz_f = __ll2float_rz(wz);
            const float world_seed_f = __ull2float_rz(world_seed);

            // TODO: REMOVE THIS DEBUG PRINT
            //printf("Compute Worldgen Data GPU: [Thread %d] wx = %f, wz = %f\n", my_id, wx_f, wz_f);

            float h = expscales[N_H]->compute(world_seed_f, wx_f, wz_f),
                  m = expscales[N_M]->compute(world_seed_f, wx_f, wz_f) * 0.5f + 0.5f,
                  t = expscales[N_T]->compute(world_seed_f, wx_f, wz_f) * 0.5f + 0.5f,
                  r = expscales[N_R]->compute(world_seed_f, wx_f, wz_f),
                  n = expscales[N_N]->compute(world_seed_f, wx_f, wz_f),
                  p = expscales[N_P]->compute(world_seed_f, wx_f, wz_f);


            // add 'peak' noise to mountain noise
            n += safe_expf(p, (1.0f - n) * 3.0f);

            // decrease temperature with height
            t -= 0.4f * n;
            t = clamp(t, 0.0f, 1.0f);

            CudaBiome biome_id = get_biome(h, m, t, n, n + h);
            CudaBiomeData biome = device_biome_data[biome_id];

            h = sign(h) * fabsf(powf(fabsf(h), biome.exp));

            //printf("[Thread %u] h = %f, m = %f, t = %f, r = %f, n = %f, p = %f\n", my_id, h, m, t, r, n, p);
            //printf("[Thread %u] biome_id = %d\n", my_id, biome_id);

            data[my_id].h_b = ((h * 32.0f) + (n * 256.0f)) * biome.scale + (biome.roughness * r * 2.0f);
            data[my_id].b = biome_id;
            __syncthreads(); // barrier synchronisation needed to avoid race conditions
            // TODO: CHECK THESE POTENTIALLY DANGEROUS LINES, CAST FROM INT TO UNSIGNED INT
            const unsigned int my_x = my_id % chunk_size_x;
            const unsigned int my_z = my_id / chunk_size_x;
            assert(my_z < chunk_size_z);
            const int down_left = get_index_from_coordinates(my_x - 1, my_z - 1, chunk_size_x, chunk_size_z);
            const int down_right = get_index_from_coordinates(my_x + 1, my_z - 1, chunk_size_x, chunk_size_z);
            const int up_left = get_index_from_coordinates(my_x - 1, my_z + 1, chunk_size_x, chunk_size_z);
            const int up_right = get_index_from_coordinates(my_x + 1, my_z + 1, chunk_size_x, chunk_size_z);
            check_neighbour(down_left);
            check_neighbour(down_right);
            check_neighbour(up_left);
            check_neighbour(up_right);
            /*printf("[Thread %u]: down_left = %d, down_right = %d, up_left = %d, up_right = %d\n", my_id, down_left, down_right, up_left, up_right);*/
            float v = 0.0f;
            v += data[get_index_from_coordinates(my_x - 1, my_z - 1, chunk_size_x, chunk_size_z)].h_b;
            v += data[get_index_from_coordinates(my_x + 1, my_z - 1, chunk_size_x, chunk_size_z)].h_b;
            v += data[get_index_from_coordinates(my_x - 1, my_z + 1, chunk_size_x, chunk_size_z)].h_b;
            v += data[get_index_from_coordinates(my_x + 1, my_z + 1, chunk_size_x, chunk_size_z)].h_b;
            v *= 0.25f;
            assert(!isnan(v));
            data[get_index_from_coordinates(my_x, my_z, chunk_size_x, chunk_size_z)].h = __float2ll_rz(v);
            //printf("GPU: Worldgen Data[%u]: h_b = %f, h = %ld, b = %ld \n", my_id, data[my_id].h_b, data[my_id].h, data[my_id].b);
        }
    }

    __global__ void generate_blocks_gpu(
        CUDA_WORLDGEN_DATA *data,
        CudaBlockId *blocks,
        int chunk_size_x, int chunk_size_y, int chunk_size_z,
        int chunk_world_position_y,
        unsigned long total_blocks_number,
        int *array_of_partial_results
    ) {
        /* Each thread of the same block modifies its own value of this shared array;
         * eventually, the threads of this block will compute a partial reduction on this
         * array. */
        __shared__ int local_generated_blocks[BLKDIM];
        const unsigned int lindex = threadIdx.x; // local index
        local_generated_blocks[lindex] = 0;
        const unsigned int global_index = blockIdx.x * blockDim.x + threadIdx.x;

        if (global_index < total_blocks_number) {
            const unsigned int my_x = global_index % chunk_size_x;
            // y theoretically should not be negative, since we're considering coordinates within a chunk
            const unsigned int my_y = global_index / (chunk_size_x * chunk_size_z);
            const unsigned int completed_xz_planes_elements = my_y * chunk_size_x * chunk_size_z;
            const unsigned int my_z = (global_index - completed_xz_planes_elements) / chunk_size_x;
            const unsigned int my_xz = my_z * chunk_size_x + my_x;
            // Data is an array of size chunk_size_x * chunk_size_z;
            // it only contains values for columns, not for all the blocks!
            const CUDA_WORLDGEN_DATA my_data = data[my_xz];
            //printf("[Thread %u] my_xz = %u, worldgen data = { h_b = %f, h = %ld, b = %ld }\n", global_index, my_xz, my_data.h_b, my_data.h, my_data.b);
            const long h = my_data.h;
            const auto biome = static_cast<CudaBiome>(my_data.b);
            if (!(biome >= 0 && biome < BIOME_LAST + 1)) {
                //printf("[Thread %d]: biome = %d\n", global_index, biome);
                //printf("[Thread %u] my_xz = %u, worldgen data = { h_b = %f, h = %ld, b = %ld }\n", global_index, my_xz, my_data.h_b, my_data.h, my_data.b);
                assert(false);
            }
            const CudaBiomeData biome_data = device_biome_data[biome];
            const CudaBlockId top_block = h > 48 ? CUDA_SNOW : biome_data.top_block,
                              under_block = biome_data.bottom_block;

            const long y_w = chunk_world_position_y + static_cast<long>(my_y);
            CudaBlockId block = CUDA_AIR;

            if (y_w > h && y_w <= WATER_LEVEL) {
                block = CUDA_WATER;
            } else if (y_w > h) {
                blocks[global_index] = CUDA_AIR;
                return;
            } else if (y_w == h) {
                block = top_block;
            } else if (y_w >= (h - 3)) {
                block = under_block;
            } else {
                block = CUDA_STONE;
            }


            //printf("[Thread %u] generated block %d\n", global_index, block);
            blocks[global_index] = block;
            local_generated_blocks[lindex]++;
            __syncthreads();
            /* At the end, the threads of each block compute a partial reduction.
             * This algorithm is safe since BLKDIM is a power of 2 (1024 = 2^10).
             * */
            int bsize = blockDim.x / 2;
            while (bsize > 0) {
                if (lindex < bsize) {
                    local_generated_blocks[lindex] += local_generated_blocks[lindex + bsize];
                }
                bsize = bsize / 2;
                __syncthreads();
            }
            if (0 == lindex) {
                array_of_partial_results[blockIdx.x] = local_generated_blocks[lindex];
            }
            // TODO: add decorations
        }
    }

    /*
     * This function generates blocks for a single chunk.
     * parameter_data is an array of CUDA_WORLDGEN_DATA, in case they were already
     * generated. Its size is chunk_size_x * chunk_size_z.
     *
     * WARNING: this function modifies directly the memory area pointed by the
     * parameter_data. This violates good programming practices, but avoids the
     * necessity of allocating a new array and copying manually the elements one
     * by one. Watch out with the free()s!
     * */
    CUDA_RESULT generateBlocks(int chunk_size_x, int chunk_size_y, int chunk_size_z,
                               int chunk_world_position_x, int chunk_world_position_y, int chunk_world_position_z,
                               unsigned long world_seed, long hash,
                               bool must_generate_worldgen_data,
                               CUDA_WORLDGEN_DATA *parameter_data) {
	  // How many world blocks should be generated?
      const int blocks_to_generate = chunk_size_x * chunk_size_y * chunk_size_z;
      // How many GPU blocks should be allocated for the computation?
      const int gpu_blocks = calculate_gpu_blocks(blocks_to_generate);
      if (!was_environment_initialised) {
          init_random(world_seed, hash, gpu_blocks, BLKDIM); // defined in cuda-utils
          // Copying biomes related stuff to device constant memory
          initialise_tables();
          // Initialisation of noise arrays of pointers
          cudaSafeCall(cudaMalloc((void **) &bs, 4 * sizeof(CudaBasic&)));
          cudaSafeCall(cudaMalloc((void **) &os, 6 * sizeof(CudaOctave&)));
          cudaSafeCall(cudaMalloc((void **) &cs, 5 * sizeof(CudaCombined&)));
          cudaSafeCall(cudaMalloc((void **) &expscales, 6 * sizeof(CudaExpScale&)));
          generate_noise<<<1, 1>>>(bs, os, cs, expscales);
          cudaCheckError();
          was_environment_initialised = true;
      }

      CUDA_WORLDGEN_DATA *h_data = parameter_data; // WATCH OUT: FREEING h_data will cause the memory pointed by parameter_data to be freed as well!
      CUDA_WORLDGEN_DATA *d_data;
      cudaSafeCall(cudaMalloc((void**)&d_data, chunk_size_x * chunk_size_z * sizeof(CUDA_WORLDGEN_DATA)));
      //cudaSafeCall(cudaMemset(d_data, 0, chunk_size_x * chunk_size_z * sizeof(CUDA_WORLDGEN_DATA)));
      //cudaSafeCall(cudaDeviceSynchronize());

      if (must_generate_worldgen_data) {
        h_data = (CUDA_WORLDGEN_DATA *)malloc(sizeof(CUDA_WORLDGEN_DATA) * chunk_size_x * chunk_size_z);

        /* Bottom blocks of the chunk (usually 32 * 32) */
        const int bottom_blocks_to_generate = chunk_size_x * chunk_size_z;
        const int gpu_blocks_for_worldgen_data = calculate_gpu_blocks(bottom_blocks_to_generate);
        compute_worldgen_data_gpu<<<gpu_blocks_for_worldgen_data, BLKDIM>>>(
            d_data,
            chunk_size_x, chunk_size_z,
            chunk_size_x * chunk_size_z,
            chunk_world_position_x, chunk_world_position_z,
            world_seed,
            expscales
        );
        cudaCheckError();
        // TEST: remove this
        /*cudaSafeCall(cudaMemcpy(h_data, d_data, sizeof(CUDA_WORLDGEN_DATA) * chunk_size_x * chunk_size_z, cudaMemcpyDeviceToHost));
        for (int i = 0; i < chunk_size_x * chunk_size_z; i++) {
            printf("h_data[%d]: h_b = %f, h = %ld, b = %ld\n", i, h_data[i].h_b, h_data[i].h, h_data[i].b);
        }*/
        cudaSafeCall(cudaMemcpy(h_data, d_data, sizeof(CUDA_WORLDGEN_DATA) * chunk_size_x * chunk_size_z, cudaMemcpyDeviceToHost));
      } else {
          // If the data had to be generated, then they're already in the device memory; otherwise, they need to be copied
          /*printf("No worldgen data need to be generated!\n");
          printf("RECEIVED WORLDGEN DATA: \n");
          for (int i = 0; i < chunk_size_x * chunk_size_z; i+=10) {
              printf("received_h_data[%d]: h_b = %f, h = %ld, b = %ld \n", i, h_data[i].h_b, h_data[i].h, h_data[i].b);
          }*/
          cudaSafeCall(cudaMemcpy(d_data, h_data, sizeof(CUDA_WORLDGEN_DATA) * chunk_size_x * chunk_size_z, cudaMemcpyHostToDevice));
      }

      CudaBlockId *d_blocks;
      const size_t total_blocks_size = sizeof(CudaBlockId) * blocks_to_generate;
      CudaBlockId *h_blocks = (CudaBlockId *) malloc(total_blocks_size);
      cudaSafeCall(cudaMalloc((void **) &d_blocks, total_blocks_size));

      // Arrays to calculate the total number of generated blocks
      int *h_array_of_partial_results = (int *) malloc(gpu_blocks * sizeof(int));
      int *d_array_of_partial_results;
      cudaSafeCall(cudaMalloc((void **) &d_array_of_partial_results, gpu_blocks * sizeof(int)));
      //cudaSafeCall(cudaMemset(d_array_of_partial_results, 0, gpu_blocks * sizeof(int))); // used to avoid uninitialised memory, even though this might not be necessary
      //cudaSafeCall(cudaDeviceSynchronize()); // because memset is asynchronous, unless the device pointer refers to pinned memory;
      // see https://developer.nvidia.com/blog/how-optimize-data-transfers-cuda-cc/ for details on pinned memory

      generate_blocks_gpu<<<gpu_blocks, BLKDIM>>>(
        d_data,
        d_blocks,
        chunk_size_x, chunk_size_y, chunk_size_z,
        chunk_world_position_y,
        blocks_to_generate,
        d_array_of_partial_results
      );
      cudaCheckError();
      cudaSafeCall(cudaMemcpy(h_blocks, d_blocks, total_blocks_size, cudaMemcpyDeviceToHost));
      cudaSafeCall(cudaMemcpy(h_array_of_partial_results, d_array_of_partial_results, gpu_blocks * sizeof(int), cudaMemcpyDeviceToHost));

      int generated_blocks = 0;
      for (int i = 0; i < gpu_blocks; i++) {
          generated_blocks += h_array_of_partial_results[i];
      }

      //printf("IN generateBlocks: generated blocks = %d\n", generated_blocks);

      free(h_array_of_partial_results);
      cudaFree(d_blocks);
      cudaFree(d_array_of_partial_results);
      cudaFree(d_data);

      /*printf("In generateBlocks, generated blocks = %d \n", generated_blocks);
      for (int i = 0; i < generated_blocks; i+=100) {
          printf("h_blocks[%d]: %d\n", i, h_blocks[i]);
      }
      printf("Generated worldgen data: \n");
      for (int i = 0; i < chunk_size_x * chunk_size_z; i+=10) {
          printf("h_data[%d]: h_b = %f, h = %ld, b = %ld \n", i, h_data[i].h_b, h_data[i].h, h_data[i].b);
      }*/

      return CUDA_RESULT {
          .blocks_number = generated_blocks, // try to put only the useful blocks here
          .blocks = h_blocks,
          .data = h_data
      };
    }

#ifdef __cplusplus
}
#endif