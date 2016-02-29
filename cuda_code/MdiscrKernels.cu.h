#ifndef MDISCR_KERNELS
#define MDISCR_KERNELS

#include <cuda_runtime.h>

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

#endif //MDISCR_KERNELS
