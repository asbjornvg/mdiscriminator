#ifndef MDISCR_HOST
#define MDISCR_HOST

#include "MdiscrKernels.cu.h"
#include <thrust/scan.h>

#include <sys/time.h>
#include <time.h>

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

#define MAX_BLOCKS 65535

/**
 * num_elems     The size of both the input and output array.
 * d_in          The device input array. It is supposably allocated and holds
 *               valid values.
 * d_out         The device output array. If you want its data on CPU, you need
 *               to copy it back to host.
 * DISCR         Denotes the partitioning function (discriminator) and should
 *               have an implementation similar to 'class Mod4', i.e., exporting
 *               an 'apply' function and types `InType` and `TupleType'. The
 *               out-type of the discriminator is assumed to be int (equivalance
 *               classes).
 */
template<class DISCR>
typename DISCR::TupleType mdiscr( const unsigned int     num_elems,
                                 typename DISCR::InType* d_in,  // device
                                 typename DISCR::InType* d_out  // device
    ) {
    
    // Time measurement data structures.
    struct timeval t_start, t_med1, t_med2, t_med3, t_med4,
        t_end, t_diff;
    unsigned long int elapsed;
    
    // Sizes for the kernels.
    unsigned int block_size, num_blocks;
    
    // Intermediate result data structures.
    typename DISCR::TupleType sizes, offsets;
    int *classes, *indices;
    typename DISCR::TupleType *columns, *scan_results;
    
    // Compute the sizes for the kernels.
    block_size = nextMultOf( (num_elems + MAX_BLOCKS - 1) / MAX_BLOCKS, 32 );
    block_size = (block_size < 256) ? 256 : block_size;
    num_blocks = (num_elems + block_size - 1) / block_size;
    
    // Allocate memory for the intermediate results.
    cudaMalloc((void**)&classes, num_elems*sizeof(int));
    cudaMalloc((void**)&indices, num_elems*sizeof(int));
    cudaMalloc((void**)&columns, num_elems*sizeof(typename DISCR::TupleType));
    cudaMalloc((void**)&scan_results, num_elems*sizeof(typename DISCR::TupleType));
    
    // Map the condition on the array.
    gettimeofday(&t_start, NULL);
    mapCondKernel<DISCR><<<num_blocks, block_size>>>(d_in, classes, num_elems);
    cudaThreadSynchronize();
    gettimeofday(&t_med1, NULL);
    
    // Turn the elements into tuples of all zeros with a 1 in the k'th position,
    // where k is the class that the elements maps to.
    mapTupleKernel<DISCR><<<num_blocks, block_size>>>
        (classes, columns, num_elems);
    cudaThreadSynchronize();
    gettimeofday(&t_med2, NULL);
    
    // Scan the columns.
    //thrust::device_ptr<typename DISCR::TupleType> dev_ptr(columns);
    //thrust::inclusive_scan(columns, columns + num_elems, scan_results);
    scanInc<Add<typename DISCR::TupleType>,typename DISCR::TupleType>
        (block_size, num_elems, columns, scan_results);
    cudaThreadSynchronize();
    gettimeofday(&t_med3, NULL);
    
    // Now, the last tuple contains the reduction for each class, i.e., the
    // total number of elements belonging to each class.
    cudaMemcpy(&sizes, &scan_results[num_elems-1],
               sizeof(typename DISCR::TupleType), cudaMemcpyDeviceToHost);
    
    // "Exclusive scan" of the sizes tuple to produce the offsets.
    unsigned int tmp = 0;
    for(int k = 0; k < DISCR::TupleType::cardinal; k++) {
        offsets[k] = tmp;
        tmp += sizes[k];
    }
    
    // Extract the appropriate entries from the columns and add the
    // corresponding offset.
    zipWithKernel<DISCR><<<num_blocks, block_size>>>
        (classes, scan_results, offsets, indices, num_elems);
    cudaThreadSynchronize();
    gettimeofday(&t_med4, NULL);
    
    // Permute the elements based on the indices.
    permuteKernel<typename DISCR::InType><<<num_blocks, block_size>>>
        (d_in, indices, d_out, num_elems);
    cudaThreadSynchronize();
    gettimeofday(&t_end, NULL);
    
    timeval_subtract(&t_diff, &t_end, &t_start);
    elapsed = (t_diff.tv_sec*1e6+t_diff.tv_usec); 
    printf("mdiscr total runtime: %lu microsecs, from which:\n", elapsed);
    
    timeval_subtract(&t_diff, &t_med1, &t_start);
    elapsed = (t_diff.tv_sec*1e6+t_diff.tv_usec); 
    printf("\tmapCondKernel runs in: %lu microsecs\n", elapsed);
    
    timeval_subtract(&t_diff, &t_med2, &t_med1);
    elapsed = (t_diff.tv_sec*1e6+t_diff.tv_usec); 
    printf("\tmapTupleKernel runs in: %lu microsecs\n", elapsed);
    
    timeval_subtract(&t_diff, &t_med3, &t_med2);
    elapsed = (t_diff.tv_sec*1e6+t_diff.tv_usec);
    printf("\tscanInc runs in: %lu microsecs\n", elapsed);
    
    timeval_subtract(&t_diff, &t_med4, &t_med3);
    elapsed = (t_diff.tv_sec*1e6+t_diff.tv_usec); 
    printf("\tzipWithKernel runs in: %lu microsecs\n", elapsed);
    
    timeval_subtract(&t_diff, &t_end, &t_med4);
    elapsed = (t_diff.tv_sec*1e6+t_diff.tv_usec); 
    printf("\tpermuteKernel runs in: %lu microsecs\n", elapsed);
    
    // Free resources.
    cudaFree(classes);
    cudaFree(indices);
    cudaFree(columns);
    cudaFree(scan_results);
    
    return sizes;
}

#endif //MDISCR_HOST
