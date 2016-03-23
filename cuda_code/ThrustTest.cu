#include "MdiscrKernels.cu.h"
#include "HelpersHost.cu.h"
#include <thrust/scan.h>
#include <thrust/execution_policy.h>
#include <cuda_runtime.h>

__global__ void
copyKernel(int*          in_array,
           int*          out_array,
           unsigned int  d_size
    ) {
    const unsigned int gid = blockIdx.x*blockDim.x + threadIdx.x;
    if(gid < d_size) {
        out_array[gid] = in_array[gid];
    }
}

int main(int argc, char** argv) {
    if (argc != 2) {
        printf("The program takes <num_elems> as argument!\n");
        return EXIT_FAILURE;
    }
    const unsigned int num_elems = strtoul(argv[1], NULL, 10);
    
    typedef Mod4 DISCR;
    
    // Allocate memory.
    typename DISCR::InType* h_in =
        (typename DISCR::InType*) malloc(num_elems * sizeof(typename DISCR::InType));
    typename DISCR::InType* h_out =
        (typename DISCR::InType*) malloc(num_elems * sizeof(typename DISCR::InType));
    
    { // Initialize array.
        std::srand(time(NULL));
        for(unsigned int i = 0; i < num_elems; i++) {
            h_in[i] = std::rand() % 20;
        }
    }
    printIntArray(num_elems, "h_in", h_in);
    
    int *d_in, *d_out;
    { // Device allocation.
        cudaMalloc((void**)&d_in,  num_elems * sizeof(int));
        cudaMalloc((void**)&d_out, num_elems * sizeof(int));
        
        // Copy host memory to device.
        cudaMemcpy(d_in, h_in, num_elems * sizeof(int), cudaMemcpyHostToDevice);
        cudaThreadSynchronize();
    }
    
    // Kernels etc.
    unsigned int block_size = getBlockSize(num_elems);
    unsigned int num_blocks = getNumBlocks(num_elems, block_size);
    
    int *classes, *indices;
    typename DISCR::TupleType *columns, *scan_results;
    typename DISCR::TupleType reduction, offsets;
    
    // Allocate memory for the intermediate results.
    cudaMalloc((void**)&classes, num_elems*sizeof(int));
    cudaMalloc((void**)&indices, num_elems*sizeof(int));
    cudaMalloc((void**)&columns, num_elems*sizeof(typename DISCR::TupleType));
    cudaMalloc((void**)&scan_results, num_elems*sizeof(typename DISCR::TupleType));
    
    discrKernel<DISCR><<<num_blocks, block_size>>>(d_in, classes, num_elems);
    cudaThreadSynchronize();
    
    tupleKernel<typename DISCR::TupleType><<<num_blocks, block_size>>>
        (classes, columns, num_elems);
    cudaThreadSynchronize();
    
    thrust::inclusive_scan(thrust::device, columns, columns + num_elems, scan_results);
    cudaThreadSynchronize();
    //thrust::inclusive_scan(thrust::device, d_in, d_in + num_elems, d_out);
    
    cudaMemcpy(&reduction, &scan_results[num_elems-1],
               sizeof(typename DISCR::TupleType), cudaMemcpyDeviceToHost);
    cudaThreadSynchronize();
    
    // "Exclusive scan" of the reduction tuple to produce the offsets.
    unsigned int tmp = 0;
    for(int k = 0; k < DISCR::TupleType::cardinal; k++) {
        offsets[k] = tmp;
        tmp += reduction[k];
    }
    
     indicesKernel<typename DISCR::TupleType><<<num_blocks, block_size>>>
        (classes, scan_results, offsets, indices, num_elems);
    cudaThreadSynchronize();
    
    permuteKernel<typename DISCR::InType><<<num_blocks, block_size>>>
        (d_in, indices, d_out, num_elems);
    cudaThreadSynchronize();
    
    cudaMemcpy(h_out, d_out, num_elems * sizeof(int), cudaMemcpyDeviceToHost);
    cudaThreadSynchronize();
    
    cudaFree(classes);
    cudaFree(indices);
    cudaFree(columns);
    cudaFree(scan_results);
    
    cudaFree(d_in);
    cudaFree(d_out);
    
    printIntArray(num_elems, "h_out", h_out);
    
    // Validate?
    
    free(h_in);
    free(h_out);
    
    return EXIT_SUCCESS;
}
