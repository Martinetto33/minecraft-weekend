#ifndef CUDA_WORLDGEN_H
#define CUDA_WORLDGEN_H

#ifdef __cplusplus
extern "C" {
#endif
#include <stdbool.h>
    /* Essentially a type alias that is identical to BlockId, defined
    * in worldgen.c by the original author. Needed for struct definitions
    * below. */
    enum CudaBlockId {
        CUDA_AIR = 0,
        CUDA_GRASS = 1,
        CUDA_DIRT = 2,
        CUDA_STONE = 3,
        CUDA_SAND = 4,
        CUDA_WATER = 5,
        CUDA_GLASS = 6,
        CUDA_LOG = 7,
        CUDA_LEAVES = 8,
        CUDA_ROSE = 9,
        CUDA_BUTTERCUP = 10,
        CUDA_COAL = 11,
        CUDA_COPPER = 12,
        CUDA_LAVA = 13,
        CUDA_CLAY = 14,
        CUDA_GRAVEL = 15,
        CUDA_PLANKS = 16,
        CUDA_TORCH = 17,
        CUDA_COBBLESTONE = 18,
        CUDA_SNOW = 19,
        CUDA_PODZOL = 20,
        CUDA_SHRUB = 21,
        CUDA_TALLGRASS = 22,
        CUDA_PINE_LOG = 23,
        CUDA_PINE_LEAVES = 24
    };

    typedef struct CudaWorldgenData {
        float h_b;
        long h, b;
    } CUDA_WORLDGEN_DATA;

    /*
     * A structure containing the heightmap values (stored as a simple array of all the y
     * heights in this chunk) and the blocks calculated by the GPU.
     * blocks_number specifies the size of the array of blocks, while
     * the heightmap will likely have a constant size determine by CHUNKS_SIZE.x * CHUNKS_SIZE.z,
     * which is 32x32 = 1024 in the original configuration of the project.
     * */
    typedef struct cudaChunkResult {
        unsigned long blocks_number;
        enum CudaBlockId *blocks;
        CUDA_WORLDGEN_DATA *data;
    } CUDA_RESULT;

    /*
     * A structure containing an array of CUDA_RESULTs, one for each Chunk.
     * The Chunk structures are not memorised in this struct, to allow the maximum
     * available decoupling between the CUDA and the C code.
     * This may be useful for optimisation purposes, because this way I may be able to only
     * transfer data to and from the GPU once per frame, by generating all the chunks in a
     * single go.
     * */
    typedef struct cudaFullResult {
      int chunks_number;
      CUDA_RESULT *results;
    } CUDA_FULL_RESULT;

    /**
    * Hash is used in the random seed generator. Sum it to the world seed.
     */
    CUDA_FULL_RESULT generateChunks(int chunks_number, int chunk_size_x, int chunk_size_y, int chunk_size_z,
                                    int chunk_world_position_x, int chunk_world_position_y, int chunk_world_position_z,
                                    unsigned long world_seed, long hash,
                                    bool must_generate_heightmap);
    CUDA_RESULT generateBlocks(int chunk_size_x, int chunk_size_y, int chunk_size_z,
                               int chunk_world_position_x, int chunk_world_position_y, int chunk_world_position_z,
                               unsigned long world_seed, long hash,
                               bool must_generate_worldgen_data,
                               CUDA_WORLDGEN_DATA *parameter_data);
#ifdef __cplusplus
}
#endif

#endif //CUDA_WORLDGEN_H
