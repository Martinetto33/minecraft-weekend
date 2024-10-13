#ifndef CUDA_NOISE_H
#define CUDA_NOISE_H

#ifdef __cplusplus
extern "C" {
#endif

    class CudaNoise {
      public:
        __device__ explicit CudaNoise();
        __device__ virtual ~CudaNoise() {}
        __device__ virtual float compute(float seed, float x, float z) = 0;
    };

    // Octave noise with n octaves and seed offset o
    // Maximum amplitude is 2^0 + 2^1 + 2^2 ... 2^n = 2^(n+1) - 1
    // i.e. for octave 8, values range between [-511, 511]
    class CudaOctave final : public CudaNoise {
      public:
        __device__ explicit CudaOctave(int n, int o);
        int n, o;
        __device__ ~CudaOctave() override {}
        __device__ float compute(float seed, float x, float z) override;
    };

    // Combined noise where compute(x, z) = n.compute(x + m.compute(x, z), z)
    class CudaCombined final : public CudaNoise {
      public:
        __device__ explicit CudaCombined(CudaNoise *n, CudaNoise *m);
        CudaNoise *n, *m;
        __device__ ~CudaCombined() override {}
        __device__ float compute(float seed, float x, float z) override;
    };

    class CudaBasic final : public CudaNoise {
      public:
        __device__ explicit CudaBasic(int o);
        int o;
        __device__ ~CudaBasic() override {}
        __device__ float compute(float seed, float x, float z) override;
    };

    class CudaExpScale final : public CudaNoise {
      public:
        __device__ CudaExpScale(CudaNoise *n, float exp, float scale);
        CudaNoise *n;
        float exp, scale;
        __device__ ~CudaExpScale() override {}
        __device__ float compute(float seed, float x, float z) override;
    };

#ifdef __cplusplus
}
#endif

#endif //CUDA_NOISE_H
