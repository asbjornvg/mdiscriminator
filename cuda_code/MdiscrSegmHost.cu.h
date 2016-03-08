#ifndef MDISCR_SEGM_HOST
#define MDISCR_SEGM_HOST

#include "MdiscrKernels.cu.h"
#include "MdiscrSegmKernels.cu.h"
#include "HelpersHost.cu.h"

#include <stdio.h>

/**
 * num_elems     The size of both the input and output array.
 * in_array      The device input array. It is supposably allocated and holds
 *               valid values.
 * out_array     The device output array. If you want its data on CPU, you need
 *               to copy it back to host.
 * DISCR         Denotes the partitioning function (discriminator) and should
 *               have an implementation similar to 'class Mod4', i.e., exporting
 *               an 'apply' function and types `InType` and `TupleType'. The
 *               out-type of the discriminator is assumed to be int (equivalance
 *               classes).
 */
template<class DISCR>
void mdiscrSegm( const unsigned int       num_elems,
                 typename DISCR::InType*  in_array,      // device
                 int*            segment_sizes, // device
                 typename DISCR::InType*  out_array,     // device
                 int*            new_sizes      // device
    ) {
    
    // Time measurement data structures.
    struct timeval t_start, t_med1, t_med2, t_med3, t_med4, t_med5, t_med6,
        t_med7, t_med8, t_med9, t_med10, t_end, t_diff;
    unsigned long int elapsed;
    
    // Sizes for the kernels.
    unsigned int block_size, num_blocks;
    
    // Intermediate result data structures.
    int *sizes_extended, *sizes_accumulated, *segment_offsets,
        *classes, *indices;
    typename DISCR::TupleType *columns, *scan_results, *reductions,
        *class_offsets;
    
    // Compute the sizes for the kernels.
    block_size = nextMultOf( (num_elems + MAX_BLOCKS - 1) / MAX_BLOCKS, 32 );
    block_size = (block_size < 256) ? 256 : block_size;
    num_blocks = (num_elems + block_size - 1) / block_size;
    
    // Allocate memory for the intermediate results.
    cudaMalloc((void**)&sizes_extended,      num_elems*sizeof(int));
    cudaMalloc((void**)&sizes_accumulated,   num_elems*sizeof(int));
    cudaMalloc((void**)&segment_offsets,     num_elems*sizeof(int));
    cudaMalloc((void**)&classes,             num_elems*sizeof(int));
    cudaMalloc((void**)&indices,             num_elems*sizeof(int));
    cudaMalloc((void**)&columns,             num_elems*sizeof(typename DISCR::TupleType));
    cudaMalloc((void**)&scan_results,        num_elems*sizeof(typename DISCR::TupleType));
    cudaMalloc((void**)&reductions,          num_elems*sizeof(typename DISCR::TupleType));
    cudaMalloc((void**)&class_offsets,       num_elems*sizeof(typename DISCR::TupleType));
    
    gettimeofday(&t_start, NULL);
    sgmScanInc<Add<int>,int>
        (block_size, num_elems, segment_sizes, segment_sizes, sizes_extended);
    cudaThreadSynchronize();
    gettimeofday(&t_med1, NULL);
    
    scanInc<Add<int>,int>
        (block_size, num_elems, segment_sizes, sizes_accumulated);
    cudaThreadSynchronize();
    gettimeofday(&t_med2, NULL);
    
    segmentOffsetsKernel<<<num_blocks, block_size>>>
        (sizes_accumulated, sizes_extended, segment_offsets, num_elems);
    cudaThreadSynchronize();
    gettimeofday(&t_med3, NULL);
    
    // Find the equivalence classes using the discriminator.
    discrKernel<DISCR><<<num_blocks, block_size>>>(in_array, classes, num_elems);
    cudaThreadSynchronize();
    gettimeofday(&t_med4, NULL);
    
    // Turn the elements into tuples of all zeros with a 1 in the k'th position,
    // where k is the class that the elements maps to.
    tupleKernel<typename DISCR::TupleType><<<num_blocks, block_size>>>
        (classes, columns, num_elems);
    cudaThreadSynchronize();
    gettimeofday(&t_med5, NULL);
    
    // Scan the columns (segmented scan).
    sgmScanInc<Add<typename DISCR::TupleType>,typename DISCR::TupleType>
        (block_size, num_elems, columns, segment_sizes, scan_results);
    cudaThreadSynchronize();
    gettimeofday(&t_med6, NULL);
    
    distributeReductionsKernel<typename DISCR::TupleType><<<num_blocks, block_size>>>
        (sizes_extended, segment_offsets, scan_results, reductions, num_elems);
    cudaThreadSynchronize();
    gettimeofday(&t_med7, NULL);
    
    classOffsetsKernel<typename DISCR::TupleType><<<num_blocks, block_size>>>
        (reductions, class_offsets, num_elems);
    cudaThreadSynchronize();
    gettimeofday(&t_med8, NULL);
    
    indicesKernelSegm<typename DISCR::TupleType><<<num_blocks, block_size>>>
        (classes, scan_results, class_offsets, segment_offsets,
         indices, num_elems);
    cudaThreadSynchronize();
    gettimeofday(&t_med9, NULL);
    
    permuteKernel<typename DISCR::InType><<<num_blocks, block_size>>>
        (in_array, indices, out_array, num_elems);
    cudaThreadSynchronize();
    gettimeofday(&t_med10, NULL);
    
    sizesKernelSegm<typename DISCR::TupleType><<<num_blocks, block_size>>>
        (reductions, class_offsets, segment_offsets, new_sizes, num_elems);
        //(iot_segm, reductions, class_offsets, new_sizes, num_elems);
    cudaThreadSynchronize();
    gettimeofday(&t_end, NULL);
    
    timeval_subtract(&t_diff, &t_end, &t_start);
    elapsed = (t_diff.tv_sec*1e6+t_diff.tv_usec); 
    printf("mdiscrSegm total runtime:              %6lu microsecs, from which:\n", elapsed);
    
    timeval_subtract(&t_diff, &t_med1, &t_start);
    elapsed = (t_diff.tv_sec*1e6+t_diff.tv_usec);
    printf("\t%-30s %6lu microsecs\n", "sgmScanInc (sizes_extended):", elapsed);
    
    timeval_subtract(&t_diff, &t_med2, &t_med1);
    elapsed = (t_diff.tv_sec*1e6+t_diff.tv_usec);
    printf("\t%-30s %6lu microsecs\n", "scanInc (sizes_accumulated):", elapsed);
    
    timeval_subtract(&t_diff, &t_med3, &t_med2);
    elapsed = (t_diff.tv_sec*1e6+t_diff.tv_usec);
    printf("\t%-30s %6lu microsecs\n", "segmentOffsetsKernel:", elapsed);
    
    timeval_subtract(&t_diff, &t_med4, &t_med3);
    elapsed = (t_diff.tv_sec*1e6+t_diff.tv_usec);
    printf("\t%-30s %6lu microsecs\n", "discrKernel:", elapsed);
    
    timeval_subtract(&t_diff, &t_med5, &t_med4);
    elapsed = (t_diff.tv_sec*1e6+t_diff.tv_usec);
    printf("\t%-30s %6lu microsecs\n", "tupleKernel:", elapsed);
    
    timeval_subtract(&t_diff, &t_med6, &t_med5);
    elapsed = (t_diff.tv_sec*1e6+t_diff.tv_usec);
    printf("\t%-30s %6lu microsecs\n", "sgmScanInc (scan_results):", elapsed);
    
    timeval_subtract(&t_diff, &t_med7, &t_med6);
    elapsed = (t_diff.tv_sec*1e6+t_diff.tv_usec);
    printf("\t%-30s %6lu microsecs\n", "distributeReductionsKernel:", elapsed);
    
    timeval_subtract(&t_diff, &t_med8, &t_med7);
    elapsed = (t_diff.tv_sec*1e6+t_diff.tv_usec);
    printf("\t%-30s %6lu microsecs\n", "classOffsetsKernel:", elapsed);
    
    timeval_subtract(&t_diff, &t_med9, &t_med8);
    elapsed = (t_diff.tv_sec*1e6+t_diff.tv_usec);
    printf("\t%-30s %6lu microsecs\n", "indicesKernelSegm:", elapsed);
    
    timeval_subtract(&t_diff, &t_med10, &t_med9);
    elapsed = (t_diff.tv_sec*1e6+t_diff.tv_usec);
    printf("\t%-30s %6lu microsecs\n", "permuteKernel:", elapsed);
    
    /* timeval_subtract(&t_diff, &t_med11, &t_med10); */
    /* elapsed = (t_diff.tv_sec*1e6+t_diff.tv_usec); */
    /* printf("\t%-30s %6lu microsecs\n", "onesKernel:", elapsed); */
    
    /* timeval_subtract(&t_diff, &t_med12, &t_med11); */
    /* elapsed = (t_diff.tv_sec*1e6+t_diff.tv_usec); */
    /* printf("\t%-30s %6lu microsecs\n", "sgmScanExc (iot_segm):", elapsed); */
    
    timeval_subtract(&t_diff, &t_end, &t_med10);
    elapsed = (t_diff.tv_sec*1e6+t_diff.tv_usec);
    printf("\t%-30s %6lu microsecs\n", "sizesKernelSegm:", elapsed);
    
    // Free resources.
    cudaFree(sizes_extended);
    cudaFree(sizes_accumulated);
    cudaFree(segment_offsets);
    cudaFree(classes);
    cudaFree(indices);
    cudaFree(columns);
    cudaFree(scan_results);
    cudaFree(reductions);
    cudaFree(class_offsets);
}

#endif //MDISCR_SEGM_HOST
