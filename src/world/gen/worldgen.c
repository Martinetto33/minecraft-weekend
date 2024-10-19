#include "../../cuda/cuda-worldgen.h"
#include "worldgen.h"
#include "noise.h"
#include "../chunk.h"
#include "../world.h"

#define RADIAL2I(c, r, v)\
    (glms_vec2_norm(glms_vec2_sub(IVEC2S2V((c)), IVEC2S2V((v)))) / glms_vec2_norm(IVEC2S2V((r))))

#define RADIAL3I(c, r, v)\
    (glms_vec3_norm(glms_vec3_sub(IVEC3S2V((c)), IVEC3S2V((v)))) / glms_vec3_norm(IVEC3S2V((r))))

#define WATER_LEVEL 0

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
    FWGDecorate f;
    f32 chance;
};

struct BiomeData {
    enum BlockId top_block, bottom_block;
    f32 roughness, scale, exp;
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
            { .f = worldgen_shrub, .chance = 0.005f }
        }
    },
    [SAVANNA] = {
        .top_block = GRASS,
        .bottom_block = DIRT,
        .roughness = 1.0f,
        .scale = 1.0f,
        .exp = 1.0f,
        .decorations = {
            { .f = worldgen_tree, .chance = 0.001f },
            { .f = worldgen_flowers, .chance = 0.001f },
            { .f = worldgen_grass, .chance = 0.005f },
        }
    },
    [JUNGLE] = {
        .top_block = GRASS,
        .bottom_block = DIRT,
        .roughness = 1.0f,
        .scale = 1.0f,
        .exp = 1.0f,
        .decorations = {
            { .f = worldgen_tree, .chance = 0.01f },
            { .f = worldgen_flowers, .chance = 0.001f },
            { .f = worldgen_grass, .chance = 0.01f },
        }
    },
    [GRASSLAND] = {
        .top_block = GRASS,
        .bottom_block = DIRT,
        .roughness = 1.0f,
        .scale = 1.0f,
        .exp = 1.0f,
        .decorations = {
            { .f = worldgen_tree, .chance = 0.0005f },
            { .f = worldgen_flowers, .chance = 0.003f },
            { .f = worldgen_grass, .chance = 0.02f },
        }
    },
    [WOODLAND] = {
        .top_block = GRASS,
        .bottom_block = DIRT,
        .roughness = 1.0f,
        .scale = 1.0f,
        .exp = 1.0f,
        .decorations = {
            { .f = worldgen_tree, .chance = 0.007f },
            { .f = worldgen_flowers, .chance = 0.003f },
            { .f = worldgen_grass, .chance = 0.008f },
        }
    },
    [FOREST] = {
        .top_block = GRASS,
        .bottom_block = DIRT,
        .roughness = 1.0f,
        .scale = 1.0f,
        .exp = 1.0f,
        .decorations = {
            { .f = worldgen_tree, .chance = 0.009f },
            { .f = worldgen_flowers, .chance = 0.003f },
            { .f = worldgen_grass, .chance = 0.008f },
        }
    },
    [RAINFOREST] = {
        .top_block = GRASS,
        .bottom_block = DIRT,
        .roughness = 1.0f,
        .scale = 1.0f,
        .exp = 1.0f,
        .decorations = {
            { .f = worldgen_tree, .chance = 0.009f },
            { .f = worldgen_flowers, .chance = 0.003f },
            { .f = worldgen_grass, .chance = 0.008f },
        }
    },
    [TAIGA] = {
        .top_block = PODZOL,
        .bottom_block = DIRT,
        .roughness = 1.0f,
        .scale = 1.0f,
        .exp = 1.0f,
        .decorations = {
            { .f = worldgen_pine, .chance = 0.006f },
            { .f = worldgen_flowers, .chance = 0.001f },
            { .f = worldgen_grass, .chance = 0.008f },
        }
    },
    [TUNDRA] = {
        .top_block = SNOW,
        .bottom_block = STONE,
        .roughness = 1.0f,
        .scale = 1.0f,
        .exp = 1.0f,
        .decorations = {
            { .f = worldgen_pine, .chance = 0.0005f }
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

const f32 HEAT_MAP[] = {
    0.05f,
    0.18f,
    0.4f,
    0.6f,
    0.8f
};

const f32 MOISTURE_MAP[] = {
    0.2f,
    0.3f,
    0.5f,
    0.6f,
    0.7f
};

// h = height, [-1, 1]
// m = moisture, [0, 1]
// t = temperature [0, 1]
// n = mountain noise [0, 1]
// i = modified heightmap noise [0, 1]
static enum Biome get_biome(f32 h, f32 m, f32 t, f32 n, f32 i) {
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

static enum BlockId _get(struct Chunk *chunk, s32 x, s32 y, s32 z) {
    ivec3s p = (ivec3s) {{ x, y, z }};
    if (chunk_in_bounds(p)) {
        return chunk_get_block(chunk, p);
    } else {
        return world_get_block(chunk->world, glms_ivec3_add(chunk->position, p));
    }
}

static void _set(struct Chunk *chunk, s32 x, s32 y, s32 z, u32 d) {
    ivec3s p = (ivec3s){{x, y, z}};
    if (chunk_in_bounds(p)) {
        chunk_set_block(chunk, p, d);
    }
    
    ivec3s p_w = glms_ivec3_add(chunk->position, p);
    if (!world_contains(chunk->world, p_w)) {
        world_append_unloaded_block(chunk->world, p_w, d);
    } else {
        world_set_block(chunk->world, p_w, d);
    }
}


void worldgen_generate(struct Chunk *chunk) {
    //SRAND(chunk->world->seed + ivec3shash(chunk->offset));

    struct Heightmap *heightmap = chunk_get_heightmap(chunk);

    // generate worldgen data if it doesn't exist for this chunk column yet
    if (!heightmap->flags.generated) {
        heightmap->flags.generated = true;

        struct Noise bs[] = {
            basic(1), basic(2),
            basic(3), basic(4)
        };

        struct Noise os[] = {
            octave(5, 0), octave(5, 1),
            octave(5, 2), octave(5, 3),
            octave(5, 4), octave(5, 5),
        };

        struct Noise cs[] = {
            combined(&bs[0], &bs[1]),
            combined(&bs[2], &bs[3]),
            combined(&os[3], &os[4]),
            combined(&os[1], &os[2]),
            combined(&os[1], &os[3])
        };

        struct Noise
            n_h = expscale(&os[0], 1.3f, 1.0f / 128.0f),
            n_m = expscale(&cs[0], 1.0f, 1.0f / 512.0f),
            n_t = expscale(&cs[1], 1.0f, 1.0f / 512.0f),
            n_r = expscale(&cs[2], 1.0f, 1.0f / 16.0f),
            n_n = expscale(&cs[3], 3.0f, 1.0f / 512.0f),
            n_p = expscale(&cs[4], 3.0f, 1.0f / 512.0f);

        for (s64 x = 0; x < CHUNK_SIZE.x; x++) {
            for (s64 z = 0; z < CHUNK_SIZE.z; z++) {
                s64 wx = chunk->position.x + x, wz = chunk->position.z + z;

                f32 h = n_h.compute(&n_h.params, chunk->world->seed, wx, wz),
                    m = n_m.compute(&n_m.params, chunk->world->seed, wx, wz) * 0.5f + 0.5f,
                    t = n_t.compute(&n_t.params, chunk->world->seed, wx, wz) * 0.5f + 0.5f,
                    r = n_r.compute(&n_r.params, chunk->world->seed, wx, wz),
                    n = n_n.compute(&n_n.params, chunk->world->seed, wx, wz),
                    p = n_p.compute(&n_p.params, chunk->world->seed, wx, wz);

                // add 'peak' noise to mountain noise
                n += safe_expf(p, (1.0f - n) * 3.0f);

                // decrease moisture with distance from ocean
                // m += 0.05f * n;
                
                // decrease temperature with height
                t -= 0.4f * n;
                t = clamp(t, 0.0f, 1.0f);

                enum Biome biome_id = get_biome(h, m, t, n, n + h);
                struct BiomeData biome = BIOME_DATA[biome_id];

                h = sign(h) * fabsf(powf(fabsf(h), biome.exp));

                heightmap->worldgen_data[x * CHUNK_SIZE.x + z] = (struct WorldgenData) {
                    .h_b = ((h * 32.0f) + (n * 256.0f)) * biome.scale + (biome.roughness * r * 2.0f),
                    .b = biome_id
                };
            }
        }

#define WG_GET_H(_x, _z)\
    heightmap->worldgen_data[\
        clamp((_x), 0, CHUNK_SIZE.x - 1) * CHUNK_SIZE.x +\
        clamp((_z), 0, CHUNK_SIZE.z - 1)]

        // smooth heightmap
        for (s64 x = 0; x < CHUNK_SIZE.x; x++) {
            for (s64 z = 0; z < CHUNK_SIZE.z; z++) {
                f32 v = 0.0f;
                v += (WG_GET_H(x - 1, z - 1)).h_b;
                v += (WG_GET_H(x + 1, z - 1)).h_b;
                v += (WG_GET_H(x - 1, z + 1)).h_b;
                v += (WG_GET_H(x + 1, z + 1)).h_b;
                v *= 0.25f;
                WG_GET_H(x, z).h = v;
            }
        }
    }

    for (s64 x = 0; x < CHUNK_SIZE.x; x++) {
        for (s64 z = 0; z < CHUNK_SIZE.z; z++) {
            struct WorldgenData data = heightmap->worldgen_data[x * CHUNK_SIZE.x + z];
            s64 h = data.h;
            enum Biome biome = data.b;
            struct BiomeData biome_data = BIOME_DATA[biome]; 

            enum BlockId top_block = h > 48 ? SNOW : biome_data.top_block,
                under_block = biome_data.bottom_block;

            for (s64 y = 0; y < CHUNK_SIZE.y; y++) {
                s64 y_w = chunk->position.y + y;

                enum BlockId block = AIR;

                if (y_w > h && y_w <= WATER_LEVEL) {
                    block = WATER;
                } else if (y_w > h) {
                    continue;
                } else if (y_w == h) {
                    block = top_block;
                } else if (y_w >= (h - 3)) {
                    block = under_block;
                } else {
                    block = STONE;
                }

                chunk_set_block(chunk, (ivec3s) {{ x, y, z }}, block);

                /*if (y_w == h) {
                    // decorate
                    for (size_t i = 0; i < MAX_DECORATIONS; i++) {
                        if (biome_data.decorations[i].f == NULL) {
                            break;
                        }

                        if (RANDCHANCE(biome_data.decorations[i].chance)) {
                            biome_data.decorations[i].f(chunk, _get, _set, x, y, z);
                        }
                    }
                }*/
            }
        }
    }
}

// TODO: REMOVE THIS FUNCTION
void print_binary_representation(enum CudaBlockId cuda_id, enum BlockId block) {
    int cuda_bits = sizeof(cuda_id) * 8, jdah_bits = sizeof(block) * 8;
    printf("Binary representation of cuda_id %d: ", cuda_id);
    for (int i = cuda_bits - 1; i >= 0; i--) {
        int bit = (cuda_id >> i) & 1;
        printf("%d", bit);
    }
    printf("\n");

    printf("Binary representation of block %d: ", block);
    for (int i = jdah_bits - 1; i >= 0; i--) {
        int bit = (block >> i) & 1;
        printf("%d", bit);
    }
    printf("\n");
}

void cuda_worldgen_generate(struct Chunk *chunk) {
    printf("Now processing chunk with world coordinates: x = %d, y = %d, z = %d\n", chunk->position.x, chunk->position.y, chunk->position.z);
    struct Heightmap *heightmap = chunk_get_heightmap(chunk);
    CUDA_RESULT result;
    const int bottom_blocks = CHUNK_SIZE.x * CHUNK_SIZE.z;
    if (!heightmap->flags.generated) {
        result = generateBlocks(
            CHUNK_SIZE.x, CHUNK_SIZE.y, CHUNK_SIZE.z,
            chunk->position.x, chunk->position.y, chunk->position.z,
            chunk->world->seed, ivec3shash(chunk->offset),
            true,
            NULL
        );
        heightmap->flags.generated = true;
        // Extracting the heightmap from the result
        for (int i = 0; i < bottom_blocks; i++) {
            const CUDA_WORLDGEN_DATA cuda_worldgen_datum = result.data[i];
            heightmap->worldgen_data[i] = (struct WorldgenData) {
                .h = cuda_worldgen_datum.h,
                .b = cuda_worldgen_datum.b,
                .h_b = cuda_worldgen_datum.h_b
            };
        }
    } else {
        // Worldgen data exist, so they must be converted to CUDA_WORLDGEN_DATA. A cast proved not to be enough.
        CUDA_WORLDGEN_DATA *data_to_send = (CUDA_WORLDGEN_DATA *)malloc(sizeof(CUDA_WORLDGEN_DATA) * bottom_blocks);
        for (int i = 0; i < bottom_blocks; i++) {
            CUDA_WORLDGEN_DATA data;
            data.h_b = heightmap->worldgen_data[i].h_b;
            data.b = heightmap->worldgen_data[i].b;
            data.h = heightmap->worldgen_data[i].h;
            data_to_send[i] = data;
        }
        result = generateBlocks(
            CHUNK_SIZE.x, CHUNK_SIZE.y, CHUNK_SIZE.z,
            chunk->position.x, chunk->position.y, chunk->position.z,
            chunk->world->seed, ivec3shash(chunk->offset),
            false,
            data_to_send
        );
        // No free is needed, because data_to_send will be returned in the result by generateBlocks(),
        // and a free is performed at the end of this function.
        // free(data_to_send);
    }

    printf("\n[Chunk: x = %d, y = %d, z = %d] Blocks number in CUDA_RESULT: %d\n", chunk->position.x, chunk->position.y, chunk->position.z, result.blocks_number);
    int non_air_placed_blocks = 0;
    const int all_blocks = CHUNK_SIZE.x * CHUNK_SIZE.y * CHUNK_SIZE.z;
    for (int i = 0; i < all_blocks; i++) {
        if (non_air_placed_blocks >= result.blocks_number) {
            break;
        }
        if (result.blocks[i] != CUDA_AIR) {
            const int x = i % CHUNK_SIZE.x;
            const int y = i / bottom_blocks;
            const int completed_xz_planes_elements = y * bottom_blocks;
            const int z = (i - completed_xz_planes_elements) / CHUNK_SIZE.x;
            chunk_set_block(chunk, (ivec3s) {{ x, y, z }}, result.blocks[i]);
            non_air_placed_blocks++;
        }
    }

    //print_binary_representation(CUDA_AIR, AIR);

    // Free the memory
    free(result.blocks);
    free(result.data);
}
