#include "hpc.h"
#include "cuda-worldgen.h"

#define BLKDIM 1024

#ifdef __cplusplus
extern "C" {
#endif

#define BIOME_LAST MOUNTAIN
    enum Biome {
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
    struct Decoration {
        //FWGDecorate f;
        float chance;
    };

    struct BiomeData {
        enum CudaBlockId top_block, bottom_block;
        float roughness, scale, exp;
        struct Decoration decorations[MAX_DECORATIONS];
    };

    struct BiomeData BIOME_DATA[BIOME_LAST + 1] = {
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

    const enum Biome BIOME_TABLE[6][6] = {
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

    __device__ enum Biome get_biome(float h, float m, float t, float n, float i) {
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
            if (t <= HEAT_MAP[t_i]) {
                break;
            }
        }

        for (; m_i < 4; m_i++) {
            if (m <= MOISTURE_MAP[m_i]) {
                break;
            }
        }

        return BIOME_TABLE[m_i][t_i];
    }

#ifdef __cplusplus
}
#endif