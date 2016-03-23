#ifndef MDISCR_HOST
#define MDISCR_HOST

#include "MdiscrKernels.cu.h"
#include "HelpersHost.cu.h"

#include <thrust/scan.h>
#include <thrust/execution_policy.h>
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
void mdiscr( const unsigned int       num_elems,
             typename DISCR::InType*  in_array,   // device
             typename DISCR::InType*  out_array,  // device
             int*                     sizes_array // device
    ) {
    
    // Time measurement data structures.
    struct timeval t_start, t_med1, t_med2, t_med3, t_med4, t_med5,
        t_end, t_diff;
    unsigned long int elapsed;
    
    // Sizes for the kernels.
    unsigned int block_size = getBlockSize(num_elems);
    unsigned int num_blocks = getNumBlocks(num_elems, block_size);
    
    // Intermediate result data structures.
    typename DISCR::TupleType reduction, offsets;
    int *classes, *indices;
    typename DISCR::TupleType *columns, *scan_results;
    
    // Allocate memory for the intermediate results.
    cudaMalloc((void**)&classes, num_elems*sizeof(int));
    cudaMalloc((void**)&indices, num_elems*sizeof(int));
    cudaMalloc((void**)&columns, num_elems*sizeof(typename DISCR::TupleType));
    cudaMalloc((void**)&scan_results, num_elems*sizeof(typename DISCR::TupleType));
    
    // Find the equivalence classes using the discriminator.
    gettimeofday(&t_start, NULL);
    discrKernel<DISCR><<<num_blocks, block_size>>>(in_array, classes, num_elems);
    cudaThreadSynchronize();
    gettimeofday(&t_med1, NULL);
    
    // Turn the elements into tuples of all zeros with a 1 in the k'th position,
    // where k is the class that the elements maps to.
    tupleKernel<typename DISCR::TupleType><<<num_blocks, block_size>>>
        (classes, columns, num_elems);
    cudaThreadSynchronize();
    gettimeofday(&t_med2, NULL);
    
    // Scan the columns.
#ifdef THRUST
    thrust::inclusive_scan(thrust::device, columns, columns + num_elems, scan_results);
#else
    scanInc<Add<typename DISCR::TupleType>,typename DISCR::TupleType>
        (block_size, num_elems, columns, scan_results);
#endif
    cudaThreadSynchronize();
    gettimeofday(&t_med3, NULL);
    
    // Now, the last tuple contains the reduction for each class, i.e., the
    // total number of elements belonging to each class.
    cudaMemcpy(&reduction, &scan_results[num_elems-1],
               sizeof(typename DISCR::TupleType), cudaMemcpyDeviceToHost);
    cudaThreadSynchronize();
    
    // "Exclusive scan" of the reduction tuple to produce the offsets.
    unsigned int tmp = 0;
    for(unsigned int k = 0; k < DISCR::TupleType::cardinal; k++) {
        offsets[k] = tmp;
        tmp += reduction[k];
    }
    
    // Extract the appropriate entries from the columns and add the
    // corresponding offset.
    indicesKernel<typename DISCR::TupleType><<<num_blocks, block_size>>>
        (classes, scan_results, offsets, indices, num_elems);
    cudaThreadSynchronize();
    gettimeofday(&t_med4, NULL);
    
    // Permute the elements based on the indices.
    permuteKernel<typename DISCR::InType><<<num_blocks, block_size>>>
        (in_array, indices, out_array, num_elems);
    cudaThreadSynchronize();
    gettimeofday(&t_med5, NULL);
    
    // Set all sizes to zero.
    cudaMemset(sizes_array, 0, num_elems * sizeof(int));
    
    
    sizesKernel<typename DISCR::TupleType><<<num_blocks, block_size>>>
        (offsets, reduction, sizes_array, num_elems);
    cudaThreadSynchronize();
    gettimeofday(&t_end, NULL);
    
    timeval_subtract(&t_diff, &t_end, &t_start);
    elapsed = (t_diff.tv_sec*1e6+t_diff.tv_usec); 
    printf("mdiscr total runtime:   %6lu microsecs, from which:\n", elapsed);
    
    timeval_subtract(&t_diff, &t_med1, &t_start);
    elapsed = (t_diff.tv_sec*1e6+t_diff.tv_usec); 
    printf("\t%-15s %6lu microsecs\n", "discrKernel:", elapsed);
    
    timeval_subtract(&t_diff, &t_med2, &t_med1);
    elapsed = (t_diff.tv_sec*1e6+t_diff.tv_usec); 
    printf("\t%-15s %6lu microsecs\n", "tupleKernel:", elapsed);
    
    timeval_subtract(&t_diff, &t_med3, &t_med2);
    elapsed = (t_diff.tv_sec*1e6+t_diff.tv_usec);
    printf("\t%-15s %6lu microsecs\n", "scanInc:", elapsed);
    
    timeval_subtract(&t_diff, &t_med4, &t_med3);
    elapsed = (t_diff.tv_sec*1e6+t_diff.tv_usec); 
    printf("\t%-15s %6lu microsecs\n", "indicesKernel:", elapsed);
    
    timeval_subtract(&t_diff, &t_med5, &t_med4);
    elapsed = (t_diff.tv_sec*1e6+t_diff.tv_usec); 
    printf("\t%-15s %6lu microsecs\n", "permuteKernel:", elapsed);
    
    timeval_subtract(&t_diff, &t_end, &t_med5);
    elapsed = (t_diff.tv_sec*1e6+t_diff.tv_usec); 
    printf("\t%-15s %6lu microsecs\n", "sizesKernel:", elapsed);
    
    // Free resources.
    cudaFree(classes);
    cudaFree(indices);
    cudaFree(columns);
    cudaFree(scan_results);
}

#endif //MDISCR_HOST
