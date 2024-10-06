#include "hpc.h"
#include "cuda-worldgen.h"
#include "cuda-utils.h"
#include <curand.h>
#include "noise/cuda-noise.h"

#define BLKDIM 1024

#ifdef __cplusplus
extern "C" {
#endif

__device__ CudaNoise *bs;
__device__ CudaNoise *os;
__device__ CudaNoise *cs;
__device__ CudaNoise *expscales;


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
        .top_block = SAND,
        .bottom_block = SAND,
        .roughness = 1.0f,
        .scale = 1.0f,
        .exp = 1.0f
    },
    [RIVER] = {
        .top_block = SAND,
        .bottom_block = SAND,
        .roughness = 1.0f,
        .scale = 1.0f,
        .exp = 1.0f
    },
    [BEACH] = {
        .top_block = SAND,
        .bottom_block = SAND,
        .roughness = 0.2f,
        .scale = 0.8f,
        .exp = 1.3f
    },
    [DESERT] = {
        .top_block = SAND,
        .bottom_block = SAND,
        .roughness = 0.6f,
        .scale = 0.6f,
        .exp = 1.2f,
        .decorations = {
            { /*.f = worldgen_shrub,*/ .chance = 0.005f }
        }
    },
    [SAVANNA] = {
        .top_block = GRASS,
        .bottom_block = DIRT,
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
        .top_block = GRASS,
        .bottom_block = DIRT,
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
        .top_block = GRASS,
        .bottom_block = DIRT,
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
        .top_block = GRASS,
        .bottom_block = DIRT,
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
        .top_block = GRASS,
        .bottom_block = DIRT,
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
        .top_block = GRASS,
        .bottom_block = DIRT,
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
        .top_block = PODZOL,
        .bottom_block = DIRT,
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
        .top_block = SNOW,
        .bottom_block = STONE,
        .roughness = 1.0f,
        .scale = 1.0f,
        .exp = 1.0f,
        .decorations = {
            { /*.f = worldgen_pine,*/ .chance = 0.0005f }
        }
    },
    [ICE] = {
        .top_block = SNOW,
        .bottom_block = STONE,
        .roughness = 1.0f,
        .scale = 1.0f,
        .exp = 1.0f
    },
    [MOUNTAIN] = {
        .top_block = SNOW,
        .bottom_block = STONE,
        .roughness = 2.0f,
        .scale = 1.2f,
        .exp = 1.0f
    },
};

    const enum CudaBiome BIOME_TABLE[6][6] = {
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
	__device__ __constant__ CudaBiome **device_biome_table;
    __device__ __constant__ float *device_heat_map;
    __device__ __constant__ float *device_moisture_map;

    __device__ CudaBiome get_biome(float h, float m, float t, float n, float i) {
        if (h <= 0.0f || n <= 0.0f) {
            return OCEAN;
        } else if (h <= 0.005f) {
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

        return device_biome_table[m_i][t_i];
    }

    __global__ void generate_noise() {
        bs[0] = cuda_basic(1);
        bs[1] = cuda_basic(2);
        bs[2] = cuda_basic(3);
        bs[3] = cuda_basic(4);

        os[0] = cuda_octave(5, 0);
        os[1] = cuda_octave(5, 1);
        os[2] = cuda_octave(5, 2);
        os[3] = cuda_octave(5, 3);
        os[4] = cuda_octave(5, 4);
        os[5] = cuda_octave(5, 5);

        cs[0] = cuda_combined(&bs[0], &bs[1]);
        cs[1] = cuda_combined(&bs[2], &bs[3]);
        cs[2] = cuda_combined(&os[3], &os[4]);
        cs[3] = cuda_combined(&os[1], &os[2]);
        cs[4] = cuda_combined(&os[1], &os[3]);

        expscales[0] = cuda_expscale(&os[0], 1.3f, 1.0f / 128.0f); // n_h
        expscales[1] = cuda_expscale(&cs[0], 1.0f, 1.0f / 512.0f); // n_m
        expscales[2] = cuda_expscale(&cs[1], 1.0f, 1.0f / 512.0f); // n_t
        expscales[3] = cuda_expscale(&cs[2], 1.0f, 1.0f / 16.0f); // n_r
        expscales[4] = cuda_expscale(&cs[3], 3.0f, 1.0f / 512.0f); // n_n
        expscales[5] = cuda_expscale(&cs[4], 3.0f, 1.0f / 512.0f); // n_p
    }

    void initialise_tables() {
      // First allocating an array of pointers
      cudaSafeCall(cudaMalloc((void**)&device_biome_table, 6 * sizeof(enum CudaBiome*)));
      // Now for each pointer, we allocate space for 6 enum elements
      for (int i = 0; i < 6; i++) {
        cudaSafeCall(cudaMalloc((void**)&device_biome_table[i], 6 * sizeof(enum CudaBiome)));
      }
      cudaSafeCall(cudaMalloc((void**)&device_heat_map, 6 * sizeof(float)));
      cudaSafeCall(cudaMalloc((void**)&device_moisture_map, 6 * sizeof(float)));

      // Now copying data from host to device
	  for (int i = 0; i < 6; i++) {
        cudaSafeCall(cudaMemcpyToSymbol((*(&device_biome_table[i])), BIOME_TABLE[i], 6 * sizeof(enum CudaBiome), 0));
	  }
      cudaSafeCall(cudaMemcpyToSymbol((*(&device_heat_map)), HEAT_MAP, 5 * sizeof(float), 0));
      cudaSafeCall(cudaMemcpyToSymbol((*(&device_moisture_map)), MOISTURE_MAP, 5 * sizeof(float), 0));
    }

    void destroy_tables() {
      for (int i = 0; i < 6; i++) {
        cudaFree(device_biome_table[i]);
      }
      cudaFree(device_biome_table);
      cudaFree(device_heat_map);
      cudaFree(device_moisture_map);
    }

    void initialise_noise_arrays() {
      cudaSafeCall(cudaMalloc((void**)&bs, 4 * sizeof(struct CudaNoise)));
      cudaSafeCall(cudaMalloc((void**)&os, 6 * sizeof(struct CudaNoise)));
      cudaSafeCall(cudaMalloc((void**)&cs, 5 * sizeof(struct CudaNoise)));
      cudaSafeCall(cudaMalloc((void**)&expscales, 6 * sizeof(struct CudaNoise)));
      generate_noise<<<1,1>>>();
      cudaCheckError();
    }

    void destroy_noise_arrays() {
      cudaFree(bs);
      cudaFree(os);
      cudaFree(cs);
      cudaFree(expscales);
    }

    /*
     * This function generates blocks for a single chunk.
     * */
    CUDA_RESULT generateBlocks(int chunk_size_x, int chunk_size_y, int chunk_size_z,
                               int chunk_world_position_x, int chunk_world_position_y, int chunk_world_position_z,
                               unsigned long world_seed, long hash,
                               bool must_generate_heightmap) {
      // TODO
	  // How many world blocks should be generated?
      const int blocks_to_generate = chunk_size_x * chunk_size_y * chunk_size_z;
      // How many GPU blocks should be allocated for the computation?
      const int gpu_blocks = calculate_gpu_blocks(blocks_to_generate);
      init_random(world_seed, hash, gpu_blocks, BLKDIM); // defined in cuda-utils

      int *heightmap = (int *)malloc(chunk_size_x * chunk_size_z * sizeof(int)); // generating a y column height for each position (x, z)

      if (must_generate_heightmap) {
        initialise_tables();
        initialise_noise_arrays();



        destroy_tables();
        destroy_noise_arrays();
      }

      return CUDA_RESULT {
        .blocks_number = 0, // try to put only the useful blocks here
        .blocks = nullptr,
        .heightmap = heightmap
      };
    }

#ifdef __cplusplus
}
#endif