#ifndef MDISCR_KERNELS
#define MDISCR_KERNELS

#include <cuda_runtime.h>

/*
 * Kernel that applies the conditition given by the "apply" operation of the
 * DISCR template parameter class to each element of the input.
 */
template<class DISCR>
__global__ void
discrKernel(   typename DISCR::InType*    in_array,
               int*                       classes, //d_out
               unsigned int               d_size
    ) {
    const unsigned int gid = blockIdx.x*blockDim.x + threadIdx.x;
    if(gid < d_size) {
        classes[gid] = DISCR::apply(in_array[gid]);
    }
}

/*
 * Kernel that turns all the elements into tuples. An element with value k
 * is turned into a tuple of all zeros with a 1 in the k'th position.
*/
template<class TUPLETYPE>
__global__ void
tupleKernel(  int*          classes,
              TUPLETYPE*    columns, //d_out
              unsigned int  d_size
    ) {
    const unsigned int gid = blockIdx.x*blockDim.x + threadIdx.x;
    if(gid < d_size) {
        // New tuple of zeros.
        TUPLETYPE tuple;
        // Set the entry to 1 that corresponds to the class.
        tuple[classes[gid]] = 1;
        columns[gid] = tuple;
    }
}

/*
 * Kernel that computes indices based on the scan results and offsets.
*/
template<class TUPLETYPE>
__global__ void
indicesKernel(  int*          classes,
                TUPLETYPE*    scan_results,
                TUPLETYPE     offsets,
                int*          indices, //d_out
                unsigned int  d_size
    ) {
    const unsigned int gid = blockIdx.x*blockDim.x + threadIdx.x;
    if(gid < d_size) {
        // The current entries.
        int k = classes[gid];
        TUPLETYPE scan_result = scan_results[gid];
        
        // Select the k'th entries.
        int scan_result_k = scan_result[k];
        int offset_k = offsets[k];
        // Add offset. Subtract 1 to make it 0-indexed.
        indices[gid] = scan_result_k + offset_k - 1;
    }
}

/*
 * Kernel that permutes that elements of the input array based on the given
 * indices.
*/
template<class T>
__global__ void
permuteKernel(  T*            in_array,
                int*          indices,
                T*            out_array, //d_out
                unsigned int  d_size
    ) {
    const unsigned int gid = blockIdx.x*blockDim.x + threadIdx.x;
    if ( gid < d_size ) {
        int index = indices[gid];
        out_array[index] = in_array[gid];
    }
}

/*
 * Kernel that writes a sizes-array (flag-array), placing the given reduction
 * at the given offsets, and putting zeros at all the other places. Both the
 * offsets and the reduction are tuples of the same size.
 */
template<class TUPLETYPE>
__global__ void
sizesKernel(TUPLETYPE     offsets,
            TUPLETYPE     reduction,
            int*          sizes, //d_out
            unsigned int  d_size
    ) {
    const unsigned int gid = blockIdx.x*blockDim.x + threadIdx.x;
    if(gid < d_size) {
        if (gid < TUPLETYPE::cardinal) {
            int index = offsets[gid];
            int size = reduction[gid];
            sizes[index] = size;
        }
    }
}

#endif //MDISCR_KERNELS
