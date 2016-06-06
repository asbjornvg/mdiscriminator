#include "MdiscrHost_baseline.cu.h"
#include "HelpersHost.cu.h"
#include "MainCommon.h"

#include <stdio.h>
//#include "cuda_profiler_api.h"

template<class ModN>
int test(const unsigned int num_elems) {
    
    // Allocate memory.
    typename ModN::InType* h_in =
        (typename ModN::InType*) malloc(num_elems * sizeof(typename ModN::InType));
    typename ModN::InType* h_out =
        (typename ModN::InType*) malloc(num_elems * sizeof(typename ModN::InType));
    unsigned int* h_out_sizes = (unsigned int*) malloc(num_elems * sizeof(unsigned int));
    
    { // Initialize array.
        
        // Seed the random number generator.
        std::srand(time(NULL));
        
        populateIntArray(num_elems, h_in);
    }
    /* printIntArray(num_elems, "h_in", h_in); */
    
    typename ModN::InType *d_in, *d_out;
    unsigned int *d_out_sizes;
    { // Device allocation.
        cudaMalloc((void**)&d_in ,   num_elems * sizeof(typename ModN::InType));
        cudaMalloc((void**)&d_out,   num_elems * sizeof(typename ModN::InType));
        cudaMalloc((void**)&d_out_sizes, num_elems * sizeof(unsigned int));
        
        // Copy host memory to device.
        cudaMemcpy(d_in, h_in, num_elems * sizeof(typename ModN::InType), cudaMemcpyHostToDevice);
        cudaThreadSynchronize();
    }
    
    /* cudaProfilerStart(); */
    
    // Call the discriminator function.
    mdiscr<ModN>(num_elems, d_in, d_out, d_out_sizes);
    //typename ModN::TupleType sizes = mdiscr<ModN>(num_elems, (1<<16), d_in, d_out);
    
    /* cudaProfilerStop(); */
    
    // Copy result back to host.
    cudaMemcpy(h_out, d_out, num_elems * sizeof(typename ModN::InType), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_out_sizes, d_out_sizes, num_elems * sizeof(unsigned int), cudaMemcpyDeviceToHost);
    
    // Free device memory.
    cudaFree(d_in);
    cudaFree(d_out);
    cudaFree(d_out_sizes);
    
    /* printIntArray(num_elems, "h_out", h_out); */
    /* printIntArray(num_elems, "h_out_sizes", h_out_sizes); */
    
    bool success = validateOneSegment<ModN>(h_in, h_out, h_out_sizes, num_elems);
    
    // Cleanup memory.
    free(h_in);
    free(h_out);
    free(h_out_sizes);
    
    if (success) {
        fprintf(stderr, "mdiscr on %d elems: VALID RESULT!\n", num_elems);
        return EXIT_SUCCESS;
    }
    else {
        fprintf(stderr, "mdiscr on %d elems: INVALID RESULT!\n", num_elems);
        return EXIT_FAILURE;
    }
}
