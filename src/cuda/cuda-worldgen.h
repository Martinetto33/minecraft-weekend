#ifndef CUDA_WORLDGEN_H
#define CUDA_WORLDGEN_H

#ifdef __cplusplus
extern "C" {
#endif
    /* Essentially a type alias that is identical to BlockId, defined
    * in worldgen.c by the original author. Needed for struct definitions
    * below. */
    enum CudaBlockId {
        AIR = 0,
        GRASS = 1,
        DIRT = 2,
        STONE = 3,
        SAND = 4,
        WATER = 5,
        GLASS = 6,
        LOG = 7,
        LEAVES = 8,
        ROSE = 9,
        BUTTERCUP = 10,
        COAL = 11,
        COPPER = 12,
        LAVA = 13,
        CLAY = 14,
        GRAVEL = 15,
        PLANKS = 16,
        TORCH = 17,
        COBBLESTONE = 18,
        SNOW = 19,
        PODZOL = 20,
        SHRUB = 21,
        TALLGRASS = 22,
        PINE_LOG = 23,
        PINE_LEAVES = 24
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
        int blocks_number;
        CudaBlockId *blocks;
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
                               bool must_generate_heightmap);
#ifdef __cplusplus
}
#endif

#endif //CUDA_WORLDGEN_H
