#ifndef HELPERS_HOST
#define HELPERS_HOST

#include "HelpersKernels.cu.h"

#include <sys/time.h>
#include <time.h>
#include <cassert>

// The maximum number of blocks in one dimension.
#define MAX_BLOCKS 65535

template<class T>
class Add {
public:
    typedef T BaseType;
    static __device__ __host__ inline T identity() {
        return T::identity_addition();
    }
    static __device__ __host__ inline T apply(const T t1, const T t2) {
        return t1 + t2;
    }
};

class MyInt4 {
public:
    int x; int y; int z; int w;
    static const int cardinal = 4;
    
    __device__ __host__ inline MyInt4() {
        x = 0; y = 0; z = 0; w = 0; 
    }
    __device__ __host__ inline MyInt4(const int& a, const int& b, const int& c, const int& d) {
        x = a; y = b; z = c; w = d; 
    }
    __device__ __host__ inline MyInt4(const MyInt4& i4) {
        x = i4.x; y = i4.y; z = i4.z; w = i4.w; 
    }
    __device__ __host__ inline MyInt4(const volatile MyInt4& i4) {
        x = i4.x; y = i4.y; z = i4.z; w = i4.w; 
    }
    volatile __device__ __host__ inline MyInt4& operator=(const MyInt4& i4) volatile {
        x = i4.x; y = i4.y; z = i4.z; w = i4.w; 
        return *this;
    }
    __device__ __host__ friend MyInt4 operator+(const MyInt4 &m1, const MyInt4 &m2) {
        return MyInt4(m1.x+m2.x, m1.y+m2.y, m1.z+m2.z, m1.w+m2.w);
    }
    __device__ __host__ int& operator[](const int i) {
        assert(i >= 0 && i <= 3);
        if (i == 0) return x;
        else if (i == 1) return y;
        else if (i == 2) return z;
        else return w; // i == 3
    }
    static __device__ __host__ MyInt4 identity_addition() {
        return MyInt4(0,0,0,0);
    }
    static __device__ __host__ MyInt4 identity_multiplication() {
        // Does this even make sense? What is the neutral element of a 4-tuple
        // w.r.t multiplication?
        return MyInt4(1,1,1,1);
    }
    __device__ __host__ inline void set(const volatile MyInt4& i4) {
        x = i4.x; y = i4.y; z = i4.z; w = i4.w; 
    }
};

class Mod4 {
public:
    typedef int           InType;
    // The out-type is always int for these discriminators.
    typedef MyInt4        TupleType;
    static __host__ __device__ inline int apply(volatile int x) {
        return x & 3;
    }
};

int nextMultOf(unsigned int x, unsigned int m) {
    if( x % m ) return x - (x % m) + m;
    else        return x;
}

int timeval_subtract(struct timeval *result, struct timeval *t2, struct timeval *t1)
{
    unsigned int resolution=1000000;
    long int diff = (t2->tv_usec + resolution * t2->tv_sec) - (t1->tv_usec + resolution * t1->tv_sec);
    result->tv_sec = diff / resolution;
    result->tv_usec = diff % resolution;
    return (diff<0);
}

/**
 * block_size is the size of the cuda block (must be a multiple 
 *                of 32 less than 1025)
 * d_size     is the size of both the input and output arrays.
 * d_in       is the device array; it is supposably
 *                allocated and holds valid values (input).
 * d_out      is the output GPU array -- if you want 
 *            its data on CPU needs to copy it back to host.
 *
 * OP         class denotes the associative binary operator 
 *                and should have an implementation similar to 
 *                `class Add' in ScanUtil.cu, i.e., exporting
 *                `identity' and `apply' functions.
 * T          denotes the type on which OP operates, 
 *                e.g., float or int. 
 */
template<class OP, class T>
void scanInc(    unsigned int  block_size,
                 unsigned long d_size, 
                 T*            d_in,  // device
                 T*            d_out  // device
) {
    unsigned int num_blocks;
    unsigned int sh_mem_size = block_size * 32; //sizeof(T);

    num_blocks = ( (d_size % block_size) == 0) ?
                    d_size / block_size     :
                    d_size / block_size + 1 ;

    scanIncKernel<OP,T><<< num_blocks, block_size, sh_mem_size >>>(d_in, d_out, d_size);
    cudaThreadSynchronize();
    
    if (block_size >= d_size) { return; }

    /**********************/
    /*** Recursive Case ***/
    /**********************/

    //   1. allocate new device input & output array of size num_blocks
    T *d_rec_in, *d_rec_out;
    cudaMalloc((void**)&d_rec_in , num_blocks*sizeof(T));
    cudaMalloc((void**)&d_rec_out, num_blocks*sizeof(T));

    unsigned int num_blocks_rec = ( (num_blocks % block_size) == 0 ) ?
                                  num_blocks / block_size     :
                                  num_blocks / block_size + 1 ; 

    //   2. copy in the end-of-block results of the previous scan 
    copyEndOfBlockKernel<T><<< num_blocks_rec, block_size >>>(d_out, d_rec_in, num_blocks);
    cudaThreadSynchronize();

    //   3. scan recursively the last elements of each CUDA block
    scanInc<OP,T>( block_size, num_blocks, d_rec_in, d_rec_out );

    //   4. distribute the the corresponding element of the 
    //      recursively scanned data to all elements of the
    //      corresponding original block
    distributeEndBlock<OP,T><<< num_blocks, block_size >>>(d_rec_out, d_out, d_size);
    cudaThreadSynchronize();

    //   5. clean up
    cudaFree(d_rec_in );
    cudaFree(d_rec_out);
}

#endif //HELPERS_HOST
