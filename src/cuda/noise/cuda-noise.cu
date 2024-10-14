    #include "cuda-noise.cuh"

    #include <cstdio>
    /* This file is basically a merge of jdah's noise library and libnoise found under lib/noise/ */

    #ifdef __cplusplus
    extern "C" {
    #endif
    #include <stdarg.h>
    #ifndef EPSILON_ALIN
    #define EPSILON_ALIN 0.000001f // for float comparison
    #endif

    // This is the new and improved, C(2) continuous interpolant
    #define CUDA_FADE(t) ( t * t * t * ( t * ( t * 6 - 15 ) + 10 ) )

    #define CUDA_FASTFLOOR(x) ( ((int)(x)<(x)) ? ((int)x) : ((int)x-1 ) )
    #define CUDA_LERP(t, a, b) ((a) + (t)*((b)-(a)))

    #define BASIC 0
    #define OCTAVE 1
    #define COMBINED 2
    #define EXPSCALE 3

        __device__ unsigned char cuda_perm[] = {151,160,137,91,90,15,
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

        __device__ float cuda_grad3( int hash, float x, float y , float z ) {
            int h = hash & 15;     // Convert low 4 bits of hash code into 12 simple
            float u = h<8 ? x : y; // gradient directions, and compute dot product.
            float v = h<4 ? y : h==12||h==14 ? x : z; // Fix repeats at h = 12 to 15
            return ((h&1)? -u : u) + ((h&2)? -v : v);
        }

        //---------------------------------------------------------------------
        /** 3D float Perlin noise.
         */
        __device__ float cuda_noise3( float x, float y, float z )
        {
            int ix0, iy0, ix1, iy1, iz0, iz1;
            float fx0, fy0, fz0, fx1, fy1, fz1;
            float s, t, r;
            float nxy0, nxy1, nx0, nx1, n0, n1;

            ix0 = CUDA_FASTFLOOR( x ); // Integer part of x
            iy0 = CUDA_FASTFLOOR( y ); // Integer part of y
            iz0 = CUDA_FASTFLOOR( z ); // Integer part of z
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

            r = CUDA_FADE( fz0 );
            t = CUDA_FADE( fy0 );
            s = CUDA_FADE( fx0 );

            nxy0 = cuda_grad3(cuda_perm[ix0 + cuda_perm[iy0 + cuda_perm[iz0]]], fx0, fy0, fz0);
            nxy1 = cuda_grad3(cuda_perm[ix0 + cuda_perm[iy0 + cuda_perm[iz1]]], fx0, fy0, fz1);
            nx0 = CUDA_LERP( r, nxy0, nxy1 );

            nxy0 = cuda_grad3(cuda_perm[ix0 + cuda_perm[iy1 + cuda_perm[iz0]]], fx0, fy1, fz0);
            nxy1 = cuda_grad3(cuda_perm[ix0 + cuda_perm[iy1 + cuda_perm[iz1]]], fx0, fy1, fz1);
            nx1 = CUDA_LERP( r, nxy0, nxy1 );

            n0 = CUDA_LERP( t, nx0, nx1 );

            nxy0 = cuda_grad3(cuda_perm[ix1 + cuda_perm[iy0 + cuda_perm[iz0]]], fx1, fy0, fz0);
            nxy1 = cuda_grad3(cuda_perm[ix1 + cuda_perm[iy0 + cuda_perm[iz1]]], fx1, fy0, fz1);
            nx0 = CUDA_LERP( r, nxy0, nxy1 );

            nxy0 = cuda_grad3(cuda_perm[ix1 + cuda_perm[iy1 + cuda_perm[iz0]]], fx1, fy1, fz0);
            nxy1 = cuda_grad3(cuda_perm[ix1 + cuda_perm[iy1 + cuda_perm[iz1]]], fx1, fy1, fz1);
            nx1 = CUDA_LERP( r, nxy0, nxy1 );

            n1 = CUDA_LERP( t, nx0, nx1 );

            return 0.936f * ( CUDA_LERP( s, n0, n1 ) );
        }

        //---------------------------------------------------------------------

        __device__ void check_for_nan(const int num_args, const char *caller, const float seed, const float x, const float z, ...) {
            va_list args;
            va_start(args, num_args);  // Initialize va_list for num_args arguments

            int detected_NaNs = 0;
            // Loop through all arguments
            for (int i = 0; i < num_args; i++) {
                const float arg = va_arg(args, float);  // Get the next argument (assuming it's a double)
                // Check if this argument is NaN
                if (isnan(arg)) {
                    printf("NaN detected! [Thread Id: %u] Caller: %s, NaN argument index = %d\n", blockIdx.x * blockDim.x + threadIdx.x, caller, i);
                    detected_NaNs++;
                }
            }
            if (detected_NaNs > 0) {
                printf("Complete info: [Thread Id: %u] Caller: %s, seed = %f, x = %f, z = %f, detected NaNs = %d\n", blockIdx.x * blockDim.x + threadIdx.x, caller, seed, x, z, detected_NaNs);
            }
            va_end(args);  // Clean up the va_list
        }

        __device__ static int sign(const float n) {
            return (n > EPSILON_ALIN) - (n < -EPSILON_ALIN);
        }

        __device__ CudaNoise::CudaNoise() = default;

        __device__ float CudaOctave::compute(const float seed, const float x, const float z) {
            float u = 1.0f, v = 0.0f;
            for (int i = 0; i < this->n; i++) {
                v += (1.0f / u) * cuda_noise3((x / 1.01f) * u, (z / 1.01f) * u, seed + (this->o * 32));
                u *= 2.0f;
            }
            /* DEBUG INFO; TODO: REMOVE */
            check_for_nan(1, "Octave Compute", seed, x, z, v);
            /* END OF DEBUG INFO; TODO: REMOVE */
            return v;
        }

        __device__ CudaOctave::CudaOctave(const int n, const int o) {
            this->n = n;
            this->o = o;
        }

        __device__ float CudaCombined::compute(const float seed, const float x, const float z) {
            const float result = this->n->compute(seed, x + this->m->compute(seed, x, z), z);
            /* DEBUG INFO; TODO: REMOVE */
            check_for_nan(1, "Combined compute", seed, x, z, result);
            /* END OF DEBUG INFO; TODO: REMOVE */
            return result;
        }

        __device__ CudaCombined::CudaCombined(CudaNoise *n, CudaNoise *m) {
            this->n = n;
            this->m = m;
        }

        __device__ float CudaBasic::compute(const float seed, const float x, const float z) {
            const float result = cuda_noise3(x, z, seed + (this->o * 32.0f));
            /* DEBUG INFO; TODO: REMOVE */
            check_for_nan(1, "Basic Compute", seed, x, z, result);
            /* END OF DEBUG INFO; TODO: REMOVE */
            return result;
        }

        __device__ CudaBasic::CudaBasic(const int o) {
            this->o = o;
        }

        __device__ float CudaExpScale::compute(const float seed, const float x, const float z) {
            const float n = this->n->compute(seed, x * this->scale, z * this->scale);
            const float result = sign(n) * powf(fabsf(n), this->exp);
            /* DEBUG INFO; TODO: REMOVE */
            check_for_nan(2, "Cuda Expscale", seed, x, z, n, result);
            /* END OF DEBUG INFO; TODO: REMOVE */
            return result;
        }

        __device__ CudaExpScale::CudaExpScale(CudaNoise *n, const float exp, const float scale) {
            this->n = n;
            this->exp = exp;
            this->scale = scale;
        }

    #ifdef __cplusplus
    }
    #endif