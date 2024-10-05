#include "cuda-noise.h"
/* This file is basically a merge of jdah's noise library and libnoise found under lib/noise/ */

#ifdef __cplusplus
extern "C" {
#endif
#include "../../../lib/noise/noise1234.h"

    float cuda_octave_compute(struct Octave *p, float seed, float x, float z) {
        float u = 1.0f, v = 0.0f;
        for (int i = 0; i < p->n; i++) {
            v += (1.0f / u) * noise3((x / 1.01f) * u, (z / 1.01f) * u, seed + (p->o * 32));
            u *= 2.0f;
        }
        return v;
    }

#ifdef __cplusplus
}
#endif
