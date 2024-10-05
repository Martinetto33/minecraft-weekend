#include "cuda-noise.h"
/* This file is basically a merge of jdah's noise library and libnoise found under lib/noise/ */

#ifdef __cplusplus
extern "C" {
#endif
#include "../../../lib/noise/noise1234.h"
#include <string.h> // for memcpy
#define EPSILON 0.000001f // for float comparison

    static int sign(const float n) {
        return (n > EPSILON) - (n < -EPSILON);
    }

    float cuda_octave_compute(const Octave *p, const float seed, const float x, const float z) {
        float u = 1.0f, v = 0.0f;
        for (int i = 0; i < p->n; i++) {
            v += (1.0f / u) * noise3((x / 1.01f) * u, (z / 1.01f) * u, seed + (p->o * 32));
            u *= 2.0f;
        }
        return v;
    }

    Noise cuda_octave(int n, int o) {
        Noise result = {.compute = reinterpret_cast<FNoise>(cuda_octave_compute)};
        const Octave params = {n, o};
        memcpy(&result.params, &params, sizeof(Octave));
        return result;
    }

    float cuda_combined_compute(const Combined *p, const float seed, const float x, const float z) {
        return p->n->compute(&p->n->params, seed, x + p->m->compute(&p->m->params, seed, x, z), z);
    }

    Noise cuda_combined(Noise *n, Noise *m) {
        Noise result = {.compute = reinterpret_cast<FNoise>(cuda_combined_compute)};
        const Combined params = {n, m};
        memcpy(&result.params, &params, sizeof(Combined));
        return result;
    }

    float cuda_noise_compute(const Basic *b, const float seed, const float x, const float z) {
        return noise3(x, z, seed + (b->o * 32.0f));
    }

    Noise cuda_basic(const int o) {
        Noise result = {.compute = reinterpret_cast<FNoise>(cuda_noise_compute) };
        const Basic params = { .o = o };
        memcpy(&result.params, &params, sizeof(Basic));
        return result;
    }

    float cuda_expscale_compute(const ExpScale *e, const float seed, const float x, const float z) {
        const float n = e->n->compute(&e->n->params, seed, x * e->scale, z * e->scale);
        return sign(n) * powf(fabsf(n), e->exp);
    }

    Noise cuda_expscale(Noise *n, float exp, float scale) {
        Noise result = {.compute = reinterpret_cast<FNoise>(cuda_expscale_compute) };
        const ExpScale params = { .n = n, .exp = exp, .scale = scale };
        memcpy(&result.params, &params, sizeof(ExpScale));
        return result;
    }

#ifdef __cplusplus
}
#endif
