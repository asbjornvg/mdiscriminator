#ifndef MDISCR_SEGM_KERNELS
#define MDISCR_SEGM_KERNELS

#include <cuda_runtime.h>

/*
 * 
 */
__global__ void
segmentOffsetsKernel(int*          sizes_accumulated,
                     int*          sizes_extended,
                     int*          segment_offsets, //d_out
                     unsigned int  d_size
    ) {
    const unsigned int gid = blockIdx.x*blockDim.x + threadIdx.x;
    if(gid < d_size) {
        segment_offsets[gid] = sizes_accumulated[gid] - sizes_extended[gid];
    }
}

/*
 * 
 */
template<class TUPLETYPE>
__global__ void
distributeReductionsKernel(int*          sizes_extended,
                           int*          segment_offsets,
                           TUPLETYPE*    scan_results,
                           TUPLETYPE*    reductions, //d_out
                           unsigned int  d_size
    ) {
    const unsigned int gid = blockIdx.x*blockDim.x + threadIdx.x;
    if(gid < d_size) {
        int index = sizes_extended[gid] + segment_offsets[gid] - 1;
        reductions[gid] = scan_results[index];
    }
}

/*
 * 
 */
template<class TUPLETYPE>
__global__ void
classOffsetsKernel(TUPLETYPE*    reductions,
                   TUPLETYPE*    class_offsets, //d_out
                   unsigned int  d_size
    ) {
    const unsigned int gid = blockIdx.x*blockDim.x + threadIdx.x;
    if(gid < d_size) {
        TUPLETYPE reduction = reductions[gid];
        TUPLETYPE offset;
        
        // "Exclusive scan" of the reduction tuple.
        unsigned int tmp = 0;
        for(unsigned int k = 0; k < TUPLETYPE::cardinal; k++) {
            offset[k] = tmp;
            tmp += reduction[k];
        }
        
        class_offsets[gid] = offset;
    }
}

/*
 * 
 */
template<class TUPLETYPE>
__global__ void
indicesKernelSegm(int*          classes,
                  TUPLETYPE*    scan_results,
                  TUPLETYPE*    class_offsets,
                  int*          segment_offsets,
                  int*          indices, //d_out
                  unsigned int  d_size
    ) {
    const unsigned int gid = blockIdx.x*blockDim.x + threadIdx.x;
    if(gid < d_size) {
        int k = classes[gid];
        int scan_result_k = scan_results[gid][k];
        int class_offset_k = class_offsets[gid][k];
        int segment_offset = segment_offsets[gid];
        indices[gid] = scan_result_k + class_offset_k + segment_offset - 1;
    }
}

/*
 * 
 */
template<class TUPLETYPE>
__global__ void
sizesKernelSegm(//int*          iot_segm,
                TUPLETYPE*    reductions,
                TUPLETYPE*    class_offsets,
                int*          segment_offsets,
                int*          sizes, //d_out
                unsigned int  d_size
    ) {
    const unsigned int gid = blockIdx.x*blockDim.x + threadIdx.x;
    if(gid < d_size) {
        //int iot = iot_segm[gid];
        int iot = gid - segment_offsets[gid];
        TUPLETYPE reduction = reductions[gid];
        TUPLETYPE class_offset = class_offsets[gid];
        
        sizes[gid] = 0;
        for (unsigned int k = 0; k < TUPLETYPE::cardinal; k++) {
            int reduction_k = reduction[k];
            int class_offset_k = class_offset[k];
            if (iot == class_offset_k && reduction_k > 0) {
                sizes[gid] = reduction_k;
            }
        }
    }
}

#endif //MDISCR_SEGM_KERNELS
