#include "MdiscrHost_arrays.cu.h"
#include "HelpersHost.cu.h"

#include <stdio.h>
//#include "cuda_profiler_api.h"

template<class ModN, unsigned int LEN>
int test(const unsigned int num_elems) {
    
    // Allocate memory.
    typename ModN::InType* h_in =
        (typename ModN::InType*) malloc(LEN * num_elems * sizeof(typename ModN::InType));
    typename ModN::InType* h_out =
        (typename ModN::InType*) malloc(LEN * num_elems * sizeof(typename ModN::InType));
    unsigned int* h_out_sizes = (unsigned int*)  malloc(num_elems * sizeof(unsigned int));
    
    { // Initialize array.
        
        // Seed the random number generator.
        std::srand(time(NULL));
        
        populateIntArray(LEN * num_elems, h_in);
    }
    // printIntArray(LEN * num_elems, "h_in", h_in);
    
    typename ModN::InType *d_in, *d_out;
    unsigned int *d_out_sizes;
    { // Device allocation.
        gpuErrchk( cudaMalloc((void**)&d_in ,       LEN * num_elems * sizeof(typename ModN::InType)) );
        gpuErrchk( cudaMalloc((void**)&d_out,       LEN * num_elems * sizeof(typename ModN::InType)) );
        gpuErrchk( cudaMalloc((void**)&d_out_sizes, num_elems * sizeof(unsigned int)) );
        
        // Copy host memory to device.
        gpuErrchk( cudaMemcpy(d_in, h_in, LEN * num_elems * sizeof(typename ModN::InType),
                              cudaMemcpyHostToDevice) );
        gpuErrchk( cudaThreadSynchronize() );
    }
    
    /* cudaProfilerStart(); */
    
    // Call the discriminator function.
    typename ModN::TupleType sizes =
        mdiscr<ModN, LEN> (num_elems, (1<<16), d_in, d_out);
    
    /* cudaProfilerStop(); */
    
    // Copy result back to host.
    gpuErrchk( cudaMemcpy(h_out, d_out, LEN * num_elems * sizeof(typename ModN::InType),
                          cudaMemcpyDeviceToHost) );
    gpuErrchk( cudaMemcpy(h_out_sizes, d_out_sizes, num_elems * sizeof(unsigned int),
                          cudaMemcpyDeviceToHost) );
    
    // Free device memory.
    gpuErrchk( cudaFree(d_in) );
    gpuErrchk( cudaFree(d_out) );
    gpuErrchk( cudaFree(d_out_sizes) );
    
    /* printIntArray(num_elems, "h_out", h_out); */
    /* printIntArray(num_elems, "h_out_sizes", h_out_sizes); */
    
    bool success = validateOneSegment<ModN, LEN>(h_in, h_out, sizes, num_elems);
    
    if (success) {
        fprintf(stderr, "mdiscr on %d elems: VALID RESULT!\n", num_elems);
    }
    else {
        fprintf(stderr, "mdiscr on %d elems: INVALID RESULT!\n", num_elems);
    }
    
    // Cleanup memory.
    free(h_in);
    free(h_out);
    free(h_out_sizes);
    
    if (success) {
        return EXIT_SUCCESS;
    }
    else {
        return EXIT_FAILURE;
    }
}

int main(int argc, char** argv) {
    if (argc != 2) {
        printf("The program takes <num_elems> as argument!\n");
        return EXIT_FAILURE;
    }
    
    const unsigned int num_elems = strtoul(argv[1], NULL, 10);
    
#ifndef NUM_CLASSES
#error NUM_CLASSES not defined!
#else
#ifdef SPECIALIZED
#if NUM_CLASSES == 4
    fprintf(stderr, "Discriminator is ModArray4\n");
    return test< ModArray4<INNER_LENGTH>, INNER_LENGTH >(num_elems);
#else
#error Unsupported number of equivalence classes!
#endif
#else
    fprintf(stderr, "Discriminator is ModArray<%d>\n", NUM_CLASSES);
    return test< ModArray<NUM_CLASSES,INNER_LENGTH>, INNER_LENGTH >(num_elems);
#endif
#endif
}
