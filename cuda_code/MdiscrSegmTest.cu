#include "MdiscrSegmHost.cu.h"
#include "HelpersHost.cu.h"

#include <stdio.h>
#include <cassert>

int getNextSegmentSize() {
    return (std::rand() % 50) + 25;
    //return (std::rand() % 5) + 3;
}

template<class ModN>
bool validate(typename ModN::InType*  h_in,
              int*                    h_in_sizes,
              typename ModN::InType*  h_out,
              int*                    h_out_sizes,
              unsigned int            num_elems
    ) {
    
    bool success = true;
    unsigned int i = 0;
    unsigned int j = 0;
    int size;
    
    while (i < num_elems) {
        // Size of the current segment.
        size = h_in_sizes[i];
        
        // We are only accessing segment starts, all sizes should be non-zero.
        assert(size != 0);
        
        // Validate one segment at a time.
        //printf("Validating segment number %d with size %d...\n", j, size);
        if (!validateOneSegment<ModN>(&h_in[i], &h_out[i], &h_out_sizes[i], size)) {
            success = false;
        }
        
        // Jump to the next segment start.
        i+=size;
        
        j++;
    }
    
    return success;
}

template<class ModN>
int testMdiscrSegm(const unsigned int num_elems
    ) {
    
    // Allocate memory.
    typename ModN::InType* h_in  =
        (typename ModN::InType*) malloc(num_elems * sizeof(typename ModN::InType));
    typename ModN::InType* h_out =
        (typename ModN::InType*) malloc(num_elems * sizeof(typename ModN::InType));
    int* h_in_sizes  = (int*) malloc(num_elems * sizeof(int));
    int* h_out_sizes = (int*) malloc(num_elems * sizeof(int));
    
    { // Initialize array.
        
        // Seed the random number generator.
        //std::srand(33);
        std::srand(time(NULL));
        
        // Size of the first segment.
        int current_size = getNextSegmentSize();
        if (current_size > num_elems) {
            current_size = num_elems;
        }
        
        // How far into the current segment are we?
        unsigned int j = 0;
        
        for(unsigned int i = 0; i < num_elems; i++) {
            // New random element.
            h_in[i] = std::rand() % 20;
            
            if (j == 0) {
                // If we are at the segment start, write the size.
                h_in_sizes[i] = current_size;
            }
            else {
                // Otherwise, write a zero.
                h_in_sizes[i] = 0;
            }
            
            if (j == (current_size-1)) {
                // If we are at the last element of a segment, pick at new
                // size for the next segment.
                current_size = getNextSegmentSize();
                if (current_size > (num_elems-(i+1))) {
                    current_size = num_elems-(i+1);
                }
                j = 0;
            }
            else {
                j++;
            }
        }
    }
    /* printIntArray(num_elems, "h_in", h_in); */
    /* printIntArray(num_elems, "h_in_sizes", h_in_sizes); */
    
    typename ModN::InType *d_in, *d_out;
    int *d_in_sizes, *d_out_sizes;
    { // Device allocation.
        cudaMalloc((void**)&d_in ,       num_elems * sizeof(typename ModN::InType));
        cudaMalloc((void**)&d_out,       num_elems * sizeof(typename ModN::InType));
        cudaMalloc((void**)&d_in_sizes,  num_elems * sizeof(int));
        cudaMalloc((void**)&d_out_sizes, num_elems * sizeof(int));
        
        // Copy host memory to device.
        cudaMemcpy(d_in, h_in, num_elems * sizeof(typename ModN::InType), cudaMemcpyHostToDevice);
        cudaThreadSynchronize();
        cudaMemcpy(d_in_sizes, h_in_sizes, num_elems * sizeof(int), cudaMemcpyHostToDevice);
        cudaThreadSynchronize();
    }
    
    // Call the discriminator function.
    mdiscrSegm<ModN>(num_elems, d_in, d_in_sizes, d_out, d_out_sizes);
    
    // Copy result back to host.
    cudaMemcpy(h_out, d_out, num_elems * sizeof(typename ModN::InType), cudaMemcpyDeviceToHost);
    cudaThreadSynchronize();
    cudaMemcpy(h_out_sizes, d_out_sizes, num_elems * sizeof(int), cudaMemcpyDeviceToHost);
    cudaThreadSynchronize();
    
    // Free device memory.
    cudaFree(d_in);
    cudaFree(d_out);
    cudaFree(d_in_sizes);
    cudaFree(d_out_sizes);
    
    /* printIntArray(num_elems, "h_out", h_out); */
    /* printIntArray(num_elems, "h_out_sizes", h_out_sizes); */
    
    bool success = validate<ModN>(h_in, h_in_sizes, h_out, h_out_sizes, num_elems);
    
    if (success) {
        printf("mdiscrSegm on %d elems: VALID RESULT!\n", num_elems);
    }
    else {
        printf("mdiscrSegm on %d elems: INVALID RESULT!\n", num_elems);
    }
    
    // Cleanup memory.
    free(h_in);
    free(h_out);
    free(h_in_sizes);
    free(h_out_sizes);

    return success;
    
}

int main(int argc, char** argv) {
    if (argc < 2) {
        printf("The program takes <num_elems> as argument!\n");
        return EXIT_FAILURE;
    }
    
    const unsigned int num_elems = strtoul(argv[1], NULL, 10);
    
    if (argc == 2) {
        printf("Mod4:\n");
        testMdiscrSegm<Mod4>(num_elems);
    }
    else {
        int num = strtol(argv[2], NULL, 10);
        switch (num) {
        case 1 :
            printf("Mod<ONE>:\n");
            testMdiscrSegm< Mod<ONE> >(num_elems);
            break;
        case 2 :
            printf("Mod<TWO>:\n");
            testMdiscrSegm< Mod<TWO> >(num_elems);
            break;
        case 3 :
            printf("Mod<THREE>:\n");
            testMdiscrSegm< Mod<THREE> >(num_elems);
            break;
        case 4 :
            printf("Mod<FOUR>:\n");
            testMdiscrSegm< Mod<FOUR> >(num_elems);
            break;
        case 5 :
            printf("Mod<FIVE>:\n");
            testMdiscrSegm< Mod<FIVE> >(num_elems);
            break;
        case 6 :
            printf("Mod<SIX>:\n");
            testMdiscrSegm< Mod<SIX> >(num_elems);
            break;
        case 7 :
            printf("Mod<SEVEN>:\n");
            testMdiscrSegm< Mod<SEVEN> >(num_elems);
            break;
        case 8 :
            printf("Mod<EIGHT>:\n");
            testMdiscrSegm< Mod<EIGHT> >(num_elems);
            break;
        default :
            printf("Unsupported modulo operation.\n");
            return EXIT_FAILURE;
        }
    }
    
    return EXIT_SUCCESS;
}
