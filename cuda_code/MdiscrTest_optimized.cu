#include "MdiscrHost_optimized.cu.h"
#include "HelpersHost.cu.h"
#include "MainCommon.h"

#include <stdio.h>
//#include "cuda_profiler_api.h"

/*
 * This version of the program uses the non-compact representation of the
 * tuple as default. To use the compact representation, the flag
 * COMPACT_REPRESENTATION must be set. Flags/constants can be set/defined,
 * e.g., on the command-line using the -D flag. To run the program, these
 * constants also need to be defined:
 *   - MAX_CHUNK
 *   - MAP_X
 *   - MAP_Y
 *   - WRITE_X
 *   - WRITE_Y
 *
 * The default is to use Mod4 as the discriminator. If NUM_CLASSES=X is defined,
 * then Mod<X> is used, but in that case we need to define PACKED_VY where Y is
 * either 1, 2, or 3.
 *
 * Also, these options can be given at the command-line:
 *   - optimization level, e.g., -O3 (optional)
 *   - NDEBUG (optional)
 *   - arch=sm_20 (optional, eliminate warnings)
 */

template<class ModN>
int test(const unsigned int num_elems) {
    fprintf(stderr, "sizeof(typename ModN::TupleType) = %d\n", sizeof(typename ModN::TupleType));
    fprintf(stderr, "sizeof(typename ModN::TupleType::SmallType) = %d\n", sizeof(typename ModN::TupleType::SmallType));
    fprintf(stderr, "sizeof(typename ModN::TupleType::MediumType) = %d\n", sizeof(typename ModN::TupleType::MediumType));
    
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
#ifdef PRINT
    printIntArray(num_elems, "h_in", h_in);
#endif
    
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
    typename ModN::TupleType sizes = mdiscr<ModN>(num_elems, (1<<16), d_in, d_out);
    
    /* cudaProfilerStop(); */
    
    // Copy result back to host.
    cudaMemcpy(h_out, d_out, num_elems * sizeof(typename ModN::InType), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_out_sizes, d_out_sizes, num_elems * sizeof(unsigned int), cudaMemcpyDeviceToHost);
    
    // Free device memory.
    cudaFree(d_in);
    cudaFree(d_out);
    cudaFree(d_out_sizes);
    
#ifdef PRINT
    printIntArray(num_elems, "h_out", h_out);
    /* printIntArray(num_elems, "h_out_sizes", h_out_sizes); */
#endif
    
    bool success = validateOneSegment<ModN>(h_in, h_out, sizes, num_elems);
    
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
        
        // This generates a core dump (run "ulimit -c unlimited" beforehand to
        // allow core dumps).
        // abort();
    }
}
