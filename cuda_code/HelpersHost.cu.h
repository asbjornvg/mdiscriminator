#ifndef HELPERS_HOST
#define HELPERS_HOST

#include "HelpersCommon.h"
#include "HelpersKernels.cu.h"
//#include "SeqMdiscr.h"

#include <stdio.h>
#include <string>
#include <cassert>

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
       fprintf(stderr,"GPUassert (%d): %s (%s, line %d)\n", code, cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

template<class T>
class Add {
public:
    typedef T BaseType;
    static __device__ __host__ inline T identity() {
        return T(0);
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
    __device__ __host__ inline MyInt4(int init) {
        x = init; y = init; z = init; w = init;
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
    __device__ __host__ inline MyInt4& operator=(const MyInt4& i4) {
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
    __device__ __host__ operator int * () {
        return reinterpret_cast<int *>(this);
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

template<int N>
class MyInt {
public:
    int arr[N];
    static const int cardinal = N;
    
    __device__ __host__ int& operator[](const int i) {
        assert(i >= 0 && i < N);
        return arr[i];
    }
    __device__ __host__ inline MyInt() {
        for (int i = 0; i < N; i++) {
            arr[i] = 0;
        }
    }
    __device__ __host__ inline MyInt(int init) {
        for (int i = 0; i < N; i++) {
            arr[i] = init;
        }
    }
    __device__ __host__ inline MyInt(const MyInt<N>& other) {
        for (int i = 0; i < N; i++) {
            arr[i] = other.arr[i];
        }
    }
    __device__ __host__ inline MyInt(const volatile MyInt<N>& other) {
        for (int i = 0; i < N; i++) {
            arr[i] = other.arr[i];
        }
    }
    volatile __device__ __host__ inline MyInt<N>& operator=(const MyInt<N>& other) volatile {
        for (int i = 0; i < N; i++) {
            arr[i] = other.arr[i];
        }
        return *this;
    }
    __device__ __host__ inline MyInt<N>& operator=(const MyInt<N>& other) {
        for (int i = 0; i < N; i++) {
            arr[i] = other.arr[i];
        }
        return *this;
    }
    __device__ __host__ friend MyInt<N> operator+(const MyInt<N> &m1, const MyInt<N> &m2) {
        MyInt<N> m;
        for (int i = 0; i < N; i++) {
            m.arr[i] = m1.arr[i] + m2.arr[i];
        }
        return m;
    }
    __device__ __host__ operator int * () {
        return reinterpret_cast<int *>(this);
    }
    __device__ __host__ inline void set(const volatile MyInt<N>& other) {
        for (int i = 0; i < N; i++) {
            arr[i] = other[i];
        }
    }
};

template<int N>
class Mod {
public:
    typedef int           InType;
    // The out-type is always int for these discriminators.
    typedef MyInt<N>      TupleType;
    static __host__ __device__ inline int apply(volatile int x) {
        return x % N;
    }
};

// template<class ModN>
// bool validateOneSegment(typename ModN::InType*  in_array,
//                         typename ModN::InType*  out_array,
//                         int*                    sizes_array,
//                         unsigned int            num_elems
//     ) {
    
//     bool success = true;
    
//     // Allocate memory for the comparison (sequential) run.
//     typename ModN::InType* in_array_cmp =
//         (typename ModN::InType*) malloc(num_elems * sizeof(typename ModN::InType));
//     typename ModN::InType* out_array_cmp =
//         (typename ModN::InType*) malloc(num_elems * sizeof(typename ModN::InType));
//     int* sizes_array_cmp = (int*) malloc(num_elems * sizeof(int));
    
//     // Copy the in-array.
//     memcpy(in_array_cmp, in_array, num_elems*sizeof(typename ModN::InType));
    
//     // Run the sequential version.
//     seqMdiscr<ModN, false>(num_elems, in_array_cmp, out_array_cmp, sizes_array_cmp);
    
//     typename ModN::InType out_element, out_element_cmp;
//     int size, size_cmp;
//     for(unsigned int i = 0; i < num_elems; i++) {
        
//         // Compare elements.
//         out_element = out_array[i];
//         out_element_cmp = out_array_cmp[i];
//         if (out_element != out_element_cmp) {
//             success = false;
//             printf("Violation: %d should be %d (i = %d)\n", out_element, out_element_cmp, i);
//         }
        
//         // Compare sizes.
//         size = sizes_array[i];
//         size_cmp = sizes_array_cmp[i];
//         if (size != size_cmp) {
//             success = false;
//             printf("Invalid size: %d should be %d (i = %d)\n", size, size_cmp, i);
//         }
        
//         // Only print the first few violations.
//         if (!success && i > 9) {
//             break;
//         }
//     }
    
//     // Free memory.
//     free(in_array_cmp);
//     free(out_array_cmp);
//     free(sizes_array_cmp);
    
//     return success;
// }

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
    unsigned int sh_mem_size = block_size * sizeof(T);

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

/**
 * block_size is the size of the cuda block (must be a multiple 
 *                of 32 less than 1025)
 * d_size     is the size of both the input and output arrays.
 * d_in       is the device array; it is supposably
 *                allocated and holds valid values (input).
 * flags      is the flag array, in which !=0 indicates 
 *                start of a segment.
 * d_out      is the output GPU array -- if you want 
 *            its data on CPU you need to copy it back to host.
 *
 * OP         class denotes the associative binary operator 
 *                and should have an implementation similar to 
 *                `class Add' in ScanUtil.cu, i.e., exporting
 *                `identity' and `apply' functions.
 * T          denotes the type on which OP operates, 
 *                e.g., float or int. 
 */
template<class OP, class T>
void sgmScanInc( const unsigned int  block_size,
                 const unsigned long d_size,
                 T*            d_in,  //device
                 int*          flags, //device
                 T*            d_out  //device
) {
    unsigned int num_blocks;
    unsigned int val_sh_size = block_size * sizeof(T  );
    unsigned int flg_sh_size = block_size * sizeof(int);
    
    num_blocks = ( (d_size % block_size) == 0) ?
                    d_size / block_size     :
                    d_size / block_size + 1 ;
    
    T     *d_rec_in;
    int   *f_rec_in;
    cudaMalloc((void**)&d_rec_in, num_blocks*sizeof(T  ));
    cudaMalloc((void**)&f_rec_in, num_blocks*sizeof(int));

    sgmScanIncKernel<OP,T> <<< num_blocks, block_size, val_sh_size+flg_sh_size >>>
                    (d_in, flags, d_out, f_rec_in, d_rec_in, d_size);
    cudaThreadSynchronize();
    //cudaError_t err = cudaThreadSynchronize();
    //if( err != cudaSuccess)
    //    printf("cudaThreadSynchronize error: %s\n", cudaGetErrorString(err));

    if (block_size >= d_size) { cudaFree(d_rec_in); cudaFree(f_rec_in); return; }

    //   1. allocate new device input & output array of size num_blocks
    T   *d_rec_out;
    int *f_inds;
    cudaMalloc((void**)&d_rec_out, num_blocks*sizeof(T   ));
    cudaMalloc((void**)&f_inds,    d_size    *sizeof(int ));

    //   2. recursive segmented scan on the last elements of each CUDA block
    sgmScanInc<OP,T>
                ( block_size, num_blocks, d_rec_in, f_rec_in, d_rec_out );

    //   3. create an index array that is non-zero for all elements
    //      that correspond to an open segment that crosses two blocks,
    //      and different than zero otherwise. This is implemented
    //      as a CUDA-block level inclusive scan on the flag array,
    //      i.e., the segment that start the block has zero-flags,
    //      which will be preserved by the inclusive scan. 
    scanIncKernel<Add<int>,int> <<< num_blocks, block_size, flg_sh_size >>>
                ( flags, f_inds, d_size );

    //   4. finally, accumulate the recursive result of segmented scan
    //      to the elements from the first segment of each block (if 
    //      segment is open).
    sgmDistributeEndBlock <OP,T> <<< num_blocks, block_size >>>
                ( d_rec_out, d_out, f_inds, d_size );
    cudaThreadSynchronize();

    //   5. clean up
    cudaFree(d_rec_in );
    cudaFree(d_rec_out);
    cudaFree(f_rec_in );
    cudaFree(f_inds   );
}

template<class OP, class T>
void sgmScanExc( const unsigned int  block_size,
                 const unsigned long d_size,
                 T*            d_in,  //device
                 int*          flags, //device
                 T*            d_out  //device
    ) {
    
    unsigned int num_blocks = ( (d_size % block_size) == 0) ?
        d_size / block_size     :
        d_size / block_size + 1 ;
    
    T* d_intermediate;
    cudaMalloc((void**)&d_intermediate, d_size*sizeof(T));
    
    sgmScanInc<OP,T>(block_size, d_size, d_in, flags, d_intermediate);
    
    sgmShiftRightByOne<T><<<num_blocks, block_size>>>
        (d_intermediate, flags, d_out, OP::identity(), d_size);
    cudaThreadSynchronize();
    
    cudaFree(d_intermediate);
}

#endif //HELPERS_HOST
