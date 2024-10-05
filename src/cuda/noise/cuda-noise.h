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
        struct Noise *n, *m;
    };

    struct Basic {
        int o;
    };

    struct ExpScale {
        struct Noise *n;
        int exp, scale;
    };

    struct Noise octave(int n, int o);
    struct Noise combined(struct Noise *n, struct Noise *m);
    struct Noise basic(int o);
    struct Noise expscale(struct Noise *n, float exp, float scale);

#ifdef __cplusplus
}
#endif

#endif //CUDA_NOISE_H
