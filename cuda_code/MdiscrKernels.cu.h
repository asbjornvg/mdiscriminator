#ifndef MDISCR_KERNELS
#define MDISCR_KERNELS

#include <cuda_runtime.h>
#include <cassert>

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

/*
 * Kernel that applies the conditition given by the "apply" operation of the
 * DISCR template parameter class to each element of the input.
 */
template<class DISCR>
__global__ void
mapCondKernel(   typename DISCR::InType*    d_in,
                 int*                       d_out,
                 unsigned int               d_size
    ) {
    const unsigned int gid = blockIdx.x*blockDim.x + threadIdx.x;
    if(gid < d_size) {
        d_out[gid] = DISCR::apply(d_in[gid]);
    }
}

/*
 * Kernel that turns all the elements into tuples. An element with value k
 * is turned into a tuple of all zeros with a 1 in the k'th position.
*/
template<class DISCR>
__global__ void
mapTupleKernel(  int*                        d_in,
                 typename DISCR::TupleType*  d_out,
                 unsigned int                d_size
    ) {
    const unsigned int gid = blockIdx.x*blockDim.x + threadIdx.x;
    if(gid < d_size) {
        // New tuple of zeros.
        typename DISCR::TupleType tuple;
        // Set the entry to 1 that corresponds to the class.
        tuple[d_in[gid]] = 1;
        d_out[gid] = tuple;
    }
}

/*
 * Kernel that extracts the appropriate entry from the scanned columns and adds
 * the corresponding offset.
*/
template<class DISCR>
__global__ void
zipWithKernel(  int*                        d_classes,
                typename DISCR::TupleType*  d_scan_results,
                typename DISCR::TupleType   offsets,
                int*                        d_out,
                unsigned int                d_size
    ) {
    const unsigned int gid = blockIdx.x*blockDim.x + threadIdx.x;
    if(gid < d_size) {
        // The current entries.
        int k = d_classes[gid];
        typename DISCR::TupleType scan_result = d_scan_results[gid];
        
        // Select the k'th entries.
        int scan_result_k = scan_result[k];
        int offset_k = offsets[k];
        // Add offset. Subtract 1 to make it 0-indexed.
        d_out[gid] = scan_result_k + offset_k - 1;
    }
}

/*
 * Kernel that permutes that elements of the input array based on the given
 * indices.
*/
template<class T>
__global__ void
permuteKernel(  T*            d_in,
                int*          indices,
                T*            d_out,
                unsigned int  d_size
    ) {
    const unsigned int gid = blockIdx.x*blockDim.x + threadIdx.x;
    // if ( gid < d_size ) {
    //     int prev = (gid > 0) ? indices[gid-1] : 0;
    //     int curr = indices[gid];
    //     if(prev != curr) d_out[curr-1] = d_in[gid];
    // }
    if ( gid < d_size ) {
        int index = indices[gid];
        d_out[index] = d_in[gid];
    }
}

/***************************************/
/*** Scan Inclusive Helpers & Kernel ***/
/***************************************/
template<class OP, class T>
__device__ inline
T scanIncWarp( volatile T* ptr, const unsigned int idx ) {
    const unsigned int lane = idx & 31;

    // no synchronization needed inside a WARP,
    //   i.e., SIMD execution
    if (lane >= 1)  ptr[idx] = OP::apply(ptr[idx-1],  ptr[idx]);
    if (lane >= 2)  ptr[idx] = OP::apply(ptr[idx-2],  ptr[idx]);
    if (lane >= 4)  ptr[idx] = OP::apply(ptr[idx-4],  ptr[idx]);
    if (lane >= 8)  ptr[idx] = OP::apply(ptr[idx-8],  ptr[idx]);
    if (lane >= 16) ptr[idx] = OP::apply(ptr[idx-16], ptr[idx]);
    return const_cast<T&>(ptr[idx]);
}

template<class OP, class T>
__device__ inline
T scanIncBlock(volatile T* ptr, const unsigned int idx) {
    const unsigned int lane   = idx &  31;
    const unsigned int warpid = idx >> 5;

    T val = scanIncWarp<OP,T>(ptr,idx);
    __syncthreads();

    // place the end-of-warp results in
    //   the first warp. This works because
    //   warp size = 32, and 
    //   max block size = 32^2 = 1024
    if (lane == 31) { ptr[warpid] = const_cast<T&>(ptr[idx]); }
    __syncthreads();

    //
    if (warpid == 0) scanIncWarp<OP,T>(ptr, idx);
    __syncthreads();

    if (warpid > 0) {
        val = OP::apply(ptr[warpid-1], val);
    }

    return val;
}

template<class OP, class T>
__global__ void 
scanIncKernel(T* d_in, T* d_out, unsigned int d_size) {
    extern __shared__ char sh_mem1[];
    volatile T* sh_memT = (volatile T*)sh_mem1;
    const unsigned int tid = threadIdx.x;
    const unsigned int gid = blockIdx.x*blockDim.x + tid;
    T el    = (gid < d_size) ? d_in[gid] : OP::identity();
    sh_memT[tid] = el;
    __syncthreads();
    T res   = scanIncBlock < OP, T >(sh_memT, tid);
    if (gid < d_size) d_out [gid] = res; 
}


/***********************************************************/
/*** Kernels to copy/distribute the end of block results ***/
/***********************************************************/

template<class T>
__global__ void 
copyEndOfBlockKernel(T* d_in, T* d_out, unsigned int d_out_size) {
    const unsigned int gid = blockIdx.x*blockDim.x + threadIdx.x;
    
    if(gid < d_out_size)
        d_out[gid] = d_in[ blockDim.x*(gid+1) - 1];
}

template<class OP, class T>
__global__ void 
distributeEndBlock(T* d_in, T* d_out, unsigned int d_size) {
    const unsigned int gid = blockIdx.x*blockDim.x + threadIdx.x;
    
    if(gid < d_size && blockIdx.x > 0)
        d_out[gid] = OP::apply(d_out[gid],d_in[blockIdx.x-1]);
}

#endif //MDISCR_KERNELS
