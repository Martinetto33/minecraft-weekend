//
// Created by alinb on 9/26/24.
//


#include "hpc.h"
#include "cuda-circles.h"
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>

/* The constant `BLKDIM` has to be equal to the actual number of threads per block of
 * the GPU! This is because it is used inside the kernels to create a __shared__
 * vector that can't be of variable length, i.e. it can't be declared as:
 *
 *  __shared__ my_vector[blockDim.x]
 *
 * due to the following error:
 *
 *   "Variable length array declaration cannot have static storage duration"
 *
 * See https://itecnote.com/tecnote/why-cant-the-size-of-a-static-array-be-made-variable/
 * for more details.
 * */
#define BLKDIM 1024

#ifdef __cplusplus
extern "C" { // used to avoid name mangling, and thus allowing to call this code from plain C files;
// see https://en.wikipedia.org/wiki/Name_mangling#Name_mangling_in_C.2B.2B
#endif

    typedef struct {
        float x, y;   /* coordinates of center */
        float r;      /* radius */
        float dx, dy; /* displacements due to interactions with other circles */
    } circle_t;

    /* These constants can be replaced with #defines if necessary */
    const float XMIN = 0.0;
    const float XMAX = 1000.0;
    const float YMIN = 0.0;
    const float YMAX = 1000.0;
    const float RMIN = 10.0;
    const float RMAX = 100.0;
    __device__ float EPSILON = 1e-5;
    __device__ float K = 1.5;

    int ncircles;
    __device__ int d_ncircles; // a static variable for the GPU
    circle_t *circles = NULL;

    /**
     * Return a random float in [a, b]
     */
    float randab(float a, float b)
    {
        return a + (((float)rand())/RAND_MAX) * (b-a);
    }

    /**
     * Create and populate the array `circles[]` with randomly placed
     * circles.
     *
     * Do NOT parallelize this function.
     */
    void init_circles(int n)
    {
        assert(circles == NULL);
        ncircles = n;
        circles = (circle_t*)malloc(n * sizeof(*circles));
        assert(circles != NULL);
        for (int i=0; i<n; i++) {
            circles[i].x = randab(XMIN, XMAX);
            circles[i].y = randab(YMIN, YMAX);
            circles[i].r = randab(RMIN, RMAX);
            circles[i].dx = circles[i].dy = 0.0;
        }
    }

    /**
     * Set all displacements to zero.
     */
    __global__ void reset_displacements_kernel(circle_t* d_circles)
    {
        const unsigned int my_id = blockIdx.x * blockDim.x + threadIdx.x;
        /* Avoiding accesses beyond array bounds. */
        if (my_id < d_ncircles) {
            d_circles[my_id].dx = d_circles[my_id].dy = 0.0;
        }
    }

    /**
     * Move the circles to a new position according to the forces acting
     * on each one.
     */
    __global__ void move_circles_kernel(circle_t* d_circles)
    {
        const unsigned int my_id = blockIdx.x * blockDim.x + threadIdx.x;
        /* Avoiding accesses beyond array bounds. */
        if (my_id < d_ncircles) {
            d_circles[my_id].x += d_circles[my_id].dx;
            d_circles[my_id].y += d_circles[my_id].dy;
        }
    }

    /**
     * This is the main function responsible for carrying out the heaviest computation.
     * array_of_partial_results will contain the block-specific computations of the total number
     * of intersections found.
     * `i` is the variable used to iterate on the circles array. In order to generate all the pairs
     * of circles (i, j) with j >= i, each thread verifies if its global index is > i and falls within
     * the array bounds.
     * */
    __global__ void compute_forces_kernel(circle_t* d_circles, int i, int* array_of_partial_results, int num_blocks)
    {
        /* Each thread of the same block modifies its own value of this shared array;
         * eventually, the threads of this block will compute a final partial reduction on this
         * array. */
        __shared__ int local_overlaps_array[BLKDIM];
        const unsigned int lindex = threadIdx.x; // local index
        const unsigned int gindex = blockIdx.x * blockDim.x + threadIdx.x; // global index
        local_overlaps_array[lindex] = 0; // each thread sets the number of intersections found to 0, even if it's
                                          // beyond bounds
        /* Avoiding accesses beyond array bounds. */
        if (gindex >= i + 1 && gindex < d_ncircles) {
            /* d_circles[i].x and d_circles[i].y are only read and never modified,
             * so this can be done safely without concurrency issues. */
            const float deltax = d_circles[gindex].x - d_circles[i].x;
            const float deltay = d_circles[gindex].y - d_circles[i].y;
            /* hypotf(x,y) computes sqrtf(x*x + y*y) avoiding
               overflow. This function is defined in <math.h>, and
               should be available also on CUDA. In case of troubles,
               it is ok to use sqrtf(x*x + y*y) instead. */
            const float dist = hypotf(deltax, deltay);
            const float Rsum = d_circles[i].r + d_circles[gindex].r;
            if (dist < Rsum - EPSILON) {
                local_overlaps_array[lindex]++;
                const float overlap = Rsum - dist;
                if (overlap > 0.0) {
                    // avoid division by zero
                    const float overlap_x = overlap / (dist + EPSILON) * deltax;
                    const float overlap_y = overlap / (dist + EPSILON) * deltay;
                    /* The following two accesses need to be synchronized, since all threads
                     * could potentially modify the variable at d_circles[i] concurrently. */
                    atomicAdd(&d_circles[i].dx, -(overlap_x / K));
                    atomicAdd(&d_circles[i].dy, -(overlap_y / K));
                    d_circles[gindex].dx += overlap_x / K;
                    d_circles[gindex].dy += overlap_y / K;
                }
            }
        }
        __syncthreads();
        /* At the end, the threads of each block compute a partial reduction.
         * This algorithm is safe since BLKDIM is a multiple of 2.
         * */
        int bsize = blockDim.x / 2;
        while (bsize > 0) {
            if (lindex < bsize) {
                local_overlaps_array[lindex] += local_overlaps_array[lindex + bsize];
            }
            bsize = bsize / 2;
            __syncthreads();
        }
        if (0 == lindex) {
            array_of_partial_results[(num_blocks * i) + blockIdx.x] = local_overlaps_array[0];
        }
    }

    /**
     * Compute the force acting on each circle; returns the number of
     * overlapping pairs of circles (each overlapping pair must be counted
     * only once).
     *
     * Parameters:
     * d_circles:                  the device array of circles
     * number_of_blocks:           the total number of blocks that will participate
     *                             in the computation
     * d_array_of_partial_results: a single device array of partial sums of the number
     *                             of intersections, that is large enough to contain
     *                             ncircles sets of number_of_blocks partial sums
     * h_array_of_partial_results: the host equivalent of d_array_of_partial_results;
     *                             the host will compute one final sum reduction on the
     *                             elements in this array
     */
    int compute_forces(circle_t* d_circles,
                       int number_of_blocks,
                       int* d_array_of_partial_results,
                       int* h_array_of_partial_results)
    {
        int n_intersections = 0;
        /* Main body of the function. */
        for (int i = 0; i < ncircles; i++) {
            compute_forces_kernel<<<number_of_blocks, BLKDIM>>>
            (d_circles, i,d_array_of_partial_results,number_of_blocks);
            cudaCheckError();
        }
        cudaSafeCall(cudaMemcpy(h_array_of_partial_results,
                                d_array_of_partial_results,
                                number_of_blocks * sizeof(int) * ncircles,
                                cudaMemcpyDeviceToHost));
        for (int j = 0; j < number_of_blocks * ncircles; j++) {
            n_intersections += h_array_of_partial_results[j];
        }
        return n_intersections;
    }

    #ifdef MOVIE
    void retrieve_circles(circle_t* device_array, circle_t* host_array, size_t size)
    {
        cudaSafeCall(cudaMemcpy(host_array, device_array, size, cudaMemcpyDeviceToHost));
    }

    /**
     * Dumps the circles into a text file that can be processed using
     * gnuplot. This function may be used for debugging purposes, or to
     * produce a movie of how the algorithm works.
     *
     * You may want to completely remove this function from the final
     * version.
     */
    void dump_circles( int iterno, circle_t* device_array, size_t size )
    {
        char fname[64];
        snprintf(fname, sizeof(fname), "circles-%05d.gp", iterno);
        FILE *out = fopen(fname, "w");
        const float WIDTH = XMAX - XMIN;
        const float HEIGHT = YMAX - YMIN;
        fprintf(out, "set term png notransparent large\n");
        fprintf(out, "set output \"circles-%05d.png\"\n", iterno);
        fprintf(out, "set xrange [%f:%f]\n", XMIN - WIDTH*.2, XMAX + WIDTH*.2 );
        fprintf(out, "set yrange [%f:%f]\n", YMIN - HEIGHT*.2, YMAX + HEIGHT*.2 );
        fprintf(out, "set size square\n");
        fprintf(out, "plot '-' with circles notitle\n");
        retrieve_circles(device_array, circles, size);
        for (int i=0; i<ncircles; i++) {
            fprintf(out, "%f %f %f\n", circles[i].x, circles[i].y, circles[i].r);
        }
        fprintf(out, "e\n");
        fclose(out);
    }
    #endif

    int cuda_circles_main() {
        int n = 10000; // the number of circles
        int iterations = 20;
        circle_t* d_circles = NULL;

        init_circles(n);
        const double tstart_prog = hpc_gettime();

        /* Here I must allocate the array of circles on the device. */
        const size_t array_size = ncircles * sizeof(circle_t);
        const int n_blocks = (ncircles + BLKDIM - 1) / BLKDIM;
        cudaSafeCall(cudaMalloc((void **) &d_circles, array_size));
        cudaSafeCall(cudaMemcpy(d_circles, circles, array_size,
                                cudaMemcpyHostToDevice));
        /* Initialising the device copy of n. */
        cudaSafeCall(cudaMemcpyToSymbol((*(&d_ncircles)), &ncircles,
                                        sizeof(ncircles)));

        /* Allocation of necessary data to calculate the number of overlaps. */
        int* h_array_of_partial_results = (int *)malloc(n_blocks * sizeof(int) * ncircles);
        assert(h_array_of_partial_results != NULL);
        int* d_array_of_partial_results = NULL;
        cudaSafeCall(cudaMalloc((void **) &d_array_of_partial_results,
                                n_blocks * sizeof(int) * ncircles));

    #ifdef MOVIE
        dump_circles(0, d_circles, array_size);
    #endif
        for (int it=0; it<iterations; it++) {
            const double tstart_iter = hpc_gettime();
            reset_displacements_kernel<<<n_blocks, BLKDIM>>>(d_circles);
            /* cudaCheckError() also calls cudaDeviceSynchronize() */
            cudaCheckError();
            const int n_overlaps = compute_forces(d_circles,
                                                  n_blocks,
                                                  d_array_of_partial_results,
                                                  h_array_of_partial_results);
            move_circles_kernel<<<n_blocks, BLKDIM>>>(d_circles);
            cudaCheckError();
            const double elapsed_iter = hpc_gettime() - tstart_iter;
    #ifdef MOVIE
            dump_circles(it+1, d_circles, array_size);
    #endif
            printf("Iteration %d of %d, %d overlaps (%f s)\n", it+1, iterations, n_overlaps, elapsed_iter);
        }
        const double elapsed_prog = hpc_gettime() - tstart_prog;
        printf("Elapsed time: %f\n", elapsed_prog);

        free(circles);
        free(h_array_of_partial_results);
        cudaFree(d_circles);
        cudaFree(d_array_of_partial_results);

        return EXIT_SUCCESS;
    }
#ifdef __cplusplus
}
#endif
