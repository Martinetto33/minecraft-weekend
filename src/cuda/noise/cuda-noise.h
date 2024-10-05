#ifndef CUDA_NOISE_H
#define CUDA_NOISE_H

#ifdef __cplusplus
extern "C" {
#endif

    typedef float (*FNoise)(void *p, float s, float x, float z);

    struct Noise {
        unsigned char params[512];
        FNoise compute;
    };

    // Octave noise with n octaves and seed offset o
    // Maximum amplitude is 2^0 + 2^1 + 2^2 ... 2^n = 2^(n+1) - 1
    // i.e. for octave 8, values range between [-511, 511]
    struct Octave {
        int n, o;
    };

    // Combined noise where compute(x, z) = n.compute(x + m.compute(x, z), z)
    struct Combined {
        Noise *n, *m;
    };

    struct Basic {
        int o;
    };

    struct ExpScale {
        Noise *n;
        float exp, scale;
    };

    Noise cuda_octave(int n, int o);
    Noise cuda_combined(Noise *n, Noise *m);
    Noise cuda_basic(int o);
    Noise cuda_expscale(Noise *n, float exp, float scale);

#ifdef __cplusplus
}
#endif

#endif //CUDA_NOISE_H
