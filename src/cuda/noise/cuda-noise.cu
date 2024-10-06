#include "cuda-noise.h"
/* This file is basically a merge of jdah's noise library and libnoise found under lib/noise/ */

#ifdef __cplusplus
extern "C" {
#endif
#include "../../../lib/noise/noise1234.h"
#include <string.h> // for memcpy
#define EPSILON 0.000001f // for float comparison

// This is the new and improved, C(2) continuous interpolant
#define FADE(t) ( t * t * t * ( t * ( t * 6 - 15 ) + 10 ) )

#define FASTFLOOR(x) ( ((int)(x)<(x)) ? ((int)x) : ((int)x-1 ) )
#define LERP(t, a, b) ((a) + (t)*((b)-(a)))

    __device__ unsigned char perm[] = {151,160,137,91,90,15,
  131,13,201,95,96,53,194,233,7,225,140,36,103,30,69,142,8,99,37,240,21,10,23,
  190, 6,148,247,120,234,75,0,26,197,62,94,252,219,203,117,35,11,32,57,177,33,
  88,237,149,56,87,174,20,125,136,171,168, 68,175,74,165,71,134,139,48,27,166,
  77,146,158,231,83,111,229,122,60,211,133,230,220,105,92,41,55,46,245,40,244,
  102,143,54, 65,25,63,161, 1,216,80,73,209,76,132,187,208, 89,18,169,200,196,
  135,130,116,188,159,86,164,100,109,198,173,186, 3,64,52,217,226,250,124,123,
  5,202,38,147,118,126,255,82,85,212,207,206,59,227,47,16,58,17,182,189,28,42,
  223,183,170,213,119,248,152, 2,44,154,163, 70,221,153,101,155,167, 43,172,9,
  129,22,39,253, 19,98,108,110,79,113,224,232,178,185, 112,104,218,246,97,228,
  251,34,242,193,238,210,144,12,191,179,162,241, 81,51,145,235,249,14,239,107,
  49,192,214, 31,181,199,106,157,184, 84,204,176,115,121,50,45,127, 4,150,254,
  138,236,205,93,222,114,67,29,24,72,243,141,128,195,78,66,215,61,156,180,
  151,160,137,91,90,15,
  131,13,201,95,96,53,194,233,7,225,140,36,103,30,69,142,8,99,37,240,21,10,23,
  190, 6,148,247,120,234,75,0,26,197,62,94,252,219,203,117,35,11,32,57,177,33,
  88,237,149,56,87,174,20,125,136,171,168, 68,175,74,165,71,134,139,48,27,166,
  77,146,158,231,83,111,229,122,60,211,133,230,220,105,92,41,55,46,245,40,244,
  102,143,54, 65,25,63,161, 1,216,80,73,209,76,132,187,208, 89,18,169,200,196,
  135,130,116,188,159,86,164,100,109,198,173,186, 3,64,52,217,226,250,124,123,
  5,202,38,147,118,126,255,82,85,212,207,206,59,227,47,16,58,17,182,189,28,42,
  223,183,170,213,119,248,152, 2,44,154,163, 70,221,153,101,155,167, 43,172,9,
  129,22,39,253, 19,98,108,110,79,113,224,232,178,185, 112,104,218,246,97,228,
  251,34,242,193,238,210,144,12,191,179,162,241, 81,51,145,235,249,14,239,107,
  49,192,214, 31,181,199,106,157,184, 84,204,176,115,121,50,45,127, 4,150,254,
  138,236,205,93,222,114,67,29,24,72,243,141,128,195,78,66,215,61,156,180
};

    __device__ float grad3( int hash, float x, float y , float z ) {
        int h = hash & 15;     // Convert low 4 bits of hash code into 12 simple
        float u = h<8 ? x : y; // gradient directions, and compute dot product.
        float v = h<4 ? y : h==12||h==14 ? x : z; // Fix repeats at h = 12 to 15
        return ((h&1)? -u : u) + ((h&2)? -v : v);
    }

    //---------------------------------------------------------------------
    /** 3D float Perlin noise.
     */
    __device__ float noise3( float x, float y, float z )
    {
        int ix0, iy0, ix1, iy1, iz0, iz1;
        float fx0, fy0, fz0, fx1, fy1, fz1;
        float s, t, r;
        float nxy0, nxy1, nx0, nx1, n0, n1;

        ix0 = FASTFLOOR( x ); // Integer part of x
        iy0 = FASTFLOOR( y ); // Integer part of y
        iz0 = FASTFLOOR( z ); // Integer part of z
        fx0 = x - ix0;        // Fractional part of x
        fy0 = y - iy0;        // Fractional part of y
        fz0 = z - iz0;        // Fractional part of z
        fx1 = fx0 - 1.0f;
        fy1 = fy0 - 1.0f;
        fz1 = fz0 - 1.0f;
        ix1 = ( ix0 + 1 ) & 0xff; // Wrap to 0..255
        iy1 = ( iy0 + 1 ) & 0xff;
        iz1 = ( iz0 + 1 ) & 0xff;
        ix0 = ix0 & 0xff;
        iy0 = iy0 & 0xff;
        iz0 = iz0 & 0xff;

        r = FADE( fz0 );
        t = FADE( fy0 );
        s = FADE( fx0 );

        nxy0 = grad3(perm[ix0 + perm[iy0 + perm[iz0]]], fx0, fy0, fz0);
        nxy1 = grad3(perm[ix0 + perm[iy0 + perm[iz1]]], fx0, fy0, fz1);
        nx0 = LERP( r, nxy0, nxy1 );

        nxy0 = grad3(perm[ix0 + perm[iy1 + perm[iz0]]], fx0, fy1, fz0);
        nxy1 = grad3(perm[ix0 + perm[iy1 + perm[iz1]]], fx0, fy1, fz1);
        nx1 = LERP( r, nxy0, nxy1 );

        n0 = LERP( t, nx0, nx1 );

        nxy0 = grad3(perm[ix1 + perm[iy0 + perm[iz0]]], fx1, fy0, fz0);
        nxy1 = grad3(perm[ix1 + perm[iy0 + perm[iz1]]], fx1, fy0, fz1);
        nx0 = LERP( r, nxy0, nxy1 );

        nxy0 = grad3(perm[ix1 + perm[iy1 + perm[iz0]]], fx1, fy1, fz0);
        nxy1 = grad3(perm[ix1 + perm[iy1 + perm[iz1]]], fx1, fy1, fz1);
        nx1 = LERP( r, nxy0, nxy1 );

        n1 = LERP( t, nx0, nx1 );

        return 0.936f * ( LERP( s, n0, n1 ) );
    }

    //---------------------------------------------------------------------

    __device__ static int sign(const float n) {
        return (n > EPSILON) - (n < -EPSILON);
    }

    __device__ float cuda_octave_compute(const Octave *p, const float seed, const float x, const float z) {
        float u = 1.0f, v = 0.0f;
        for (int i = 0; i < p->n; i++) {
            v += (1.0f / u) * noise3((x / 1.01f) * u, (z / 1.01f) * u, seed + (p->o * 32));
            u *= 2.0f;
        }
        return v;
    }

    __device__ CudaNoise cuda_octave(int n, int o) {
        CudaNoise result = {.compute = reinterpret_cast<FNoise>(cuda_octave_compute)};
        const Octave params = {n, o};
        memcpy(&result.params, &params, sizeof(Octave));
        return result;
    }

    __device__ float cuda_combined_compute(const Combined *p, const float seed, const float x, const float z) {
        return p->n->compute(&p->n->params, seed, x + p->m->compute(&p->m->params, seed, x, z), z);
    }

    __device__ CudaNoise cuda_combined(CudaNoise *n, CudaNoise *m) {
        CudaNoise result = {.compute = reinterpret_cast<FNoise>(cuda_combined_compute)};
        const Combined params = {n, m};
        memcpy(&result.params, &params, sizeof(Combined));
        return result;
    }

    __device__ float cuda_noise_compute(const Basic *b, const float seed, const float x, const float z) {
        return noise3(x, z, seed + (b->o * 32.0f));
    }

    __device__ CudaNoise cuda_basic(const int o) {
        CudaNoise result = {.compute = reinterpret_cast<FNoise>(cuda_noise_compute) };
        const Basic params = { .o = o };
        memcpy(&result.params, &params, sizeof(Basic));
        return result;
    }

    __device__ float cuda_expscale_compute(const ExpScale *e, const float seed, const float x, const float z) {
        const float n = e->n->compute(&e->n->params, seed, x * e->scale, z * e->scale);
        return sign(n) * powf(fabsf(n), e->exp);
    }

    __device__ CudaNoise cuda_expscale(CudaNoise *n, float exp, float scale) {
        CudaNoise result = {.compute = reinterpret_cast<FNoise>(cuda_expscale_compute) };
        const ExpScale params = { .n = n, .exp = exp, .scale = scale };
        memcpy(&result.params, &params, sizeof(ExpScale));
        return result;
    }

#ifdef __cplusplus
}
#endif
