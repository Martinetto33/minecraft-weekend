/****************************************************************************
 *
 * hpc.h - Miscellaneous utility functions for the HPC course
 *
 * Copyright (C) 2017 by Moreno Marzolla <moreno.marzolla(at)unibo.it>
 * Last modified on 2020-05-23 by Moreno Marzolla
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * --------------------------------------------------------------------------
 *
 * This header file provides a function double hpc_gettime() that
 * returns the elapsed time (in seconds) since "the epoch". The
 * function uses the timing routing of the underlying parallel
 * framework (OpenMP or MPI), if enabled; otherwise, the default is to
 * use the clock_gettime() function.
 *
 * IMPORTANT NOTE: to work reliably this header file must be the FIRST
 * header file that appears in your code.
 *
 ****************************************************************************/

#ifndef HPC_H
#define HPC_H

#if defined(_OPENMP)
#include <omp.h>
/******************************************************************************
 * OpenMP timing routines
 ******************************************************************************/
double hpc_gettime( void )
{
    return omp_get_wtime();
}

#elif defined(MPI_Init)
/******************************************************************************
 * MPI timing routines
 ******************************************************************************/
double hpc_gettime( void )
{
    return MPI_Wtime();
}

#else
/******************************************************************************
 * POSIX-based timing routines
 ******************************************************************************/
#if _XOPEN_SOURCE < 600
#define _XOPEN_SOURCE 600
#endif
#include <time.h>

/* The inline modifier allows this function to be present in multiple translation
 * units (i.e. .c or .cu files) without causing issues at link time. Since the function
 * definition (that is, its body) is directly coded in this file, then multiple definition
 * errors could occur when linking .o files since each file includes all the content of
 * this file when hpc.h is included.
 * This increases a bit the size of the codebase, but at link time there won't be
 * multiple definitions of this function.
 * When specifying a function as inlined, we're suggesting that the compiler should
 * copy directly the code of the function in the places where the function is called.
 * This avoids the overhead of calling a different function and might be suitable for
 * small, frequently called functions, like this one is.
 */
double hpc_gettime();
#endif

#ifdef __CUDACC__

#include <stdio.h>
#include <stdlib.h>

/* from https://gist.github.com/ashwin/2652488 */

#define cudaSafeCall( err ) __cudaSafeCall( err, __FILE__, __LINE__ )
#define cudaCheckError()    __cudaCheckError( __FILE__, __LINE__ )

inline void __cudaSafeCall( cudaError err, const char *file, const int line )
{
#ifndef NO_CUDA_CHECK_ERROR
    if ( cudaSuccess != err ) {
        fprintf( stderr, "cudaSafeCall() failed at %s:%i : %s\n",
                 file, line, cudaGetErrorString( err ) );
        abort();
    }
#endif
}

inline void __cudaCheckError( const char *file, const int line )
{
#ifndef NO_CUDA_CHECK_ERROR
    cudaError err = cudaGetLastError();
    if ( cudaSuccess != err ) {
        fprintf( stderr, "cudaCheckError() failed at %s:%i : %s\n",
                 file, line, cudaGetErrorString( err ) );
        abort();
    }

    /* More careful checking. However, this will affect performance.
       Comment away if needed. */
    err = cudaDeviceSynchronize();
    if( cudaSuccess != err ) {
        fprintf( stderr, "cudaCheckError() with sync failed at %s:%i : %s\n",
                 file, line, cudaGetErrorString( err ) );
        abort();
    }
#endif
}

#endif

#endif
