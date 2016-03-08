#include "MdiscrHost.cu.h"
#include "HelpersHost.cu.h"

#include <stdio.h>

template<class ModN>
int testMdiscr(  const unsigned int num_elems
    ) {
    
    // Allocate memory.
    typename ModN::InType* h_in =
        (typename ModN::InType*) malloc(num_elems * sizeof(typename ModN::InType));
    typename ModN::InType* h_out =
        (typename ModN::InType*) malloc(num_elems * sizeof(typename ModN::InType));
    int* h_out_sizes = (int*) malloc(num_elems * sizeof(int));
    
    { // Initialize array.
        
        // Seed the random number generator.
        //std::srand(33);
        std::srand(time(NULL));
        
        for(unsigned int i = 0; i < num_elems; i++) {
            h_in[i] = std::rand() % 20;
        }
    }
    /* printIntArray(num_elems, "h_in", h_in); */
    
    typename ModN::InType *d_in, *d_out;
    int *d_out_sizes;
    { // Device allocation.
        cudaMalloc((void**)&d_in ,   num_elems * sizeof(typename ModN::InType));
        cudaMalloc((void**)&d_out,   num_elems * sizeof(typename ModN::InType));
        cudaMalloc((void**)&d_out_sizes, num_elems * sizeof(int));
        
        // Copy host memory to device.
        cudaMemcpy(d_in, h_in, num_elems * sizeof(typename ModN::InType), cudaMemcpyHostToDevice);
        cudaThreadSynchronize();
    }
    
    // Call the discriminator function.
    mdiscr<ModN>(num_elems, d_in, d_out, d_out_sizes);
    
    // Copy result back to host.
    cudaMemcpy(h_out, d_out, num_elems * sizeof(typename ModN::InType), cudaMemcpyDeviceToHost);
    cudaThreadSynchronize();
    cudaMemcpy(h_out_sizes, d_out_sizes, num_elems * sizeof(int), cudaMemcpyDeviceToHost);
    cudaThreadSynchronize();
    
    // Free device memory.
    cudaFree(d_in);
    cudaFree(d_out);
    cudaFree(d_out_sizes);
    
    /* printIntArray(num_elems, "h_out", h_out); */
    /* printIntArray(num_elems, "h_out_sizes", h_out_sizes); */
    
    bool success = validateOneSegment<ModN>(h_in, h_out, h_out_sizes, num_elems);
    
    if (success) {
        printf("mdiscr on %d elems: VALID RESULT!\n", num_elems);
    }
    else {
        printf("mdiscr on %d elems: INVALID RESULT!\n", num_elems);
    }
    
    // Cleanup memory.
    free(h_in);
    free(h_out);
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
        testMdiscr<Mod4>(num_elems);
    }
    else {
        int num = strtol(argv[2], NULL, 10);
        switch (num) {
        case 1 :
            printf("Mod<ONE>:\n");
            testMdiscr< Mod<ONE> >(num_elems);
            break;
        case 2 :
            printf("Mod<TWO>:\n");
            testMdiscr< Mod<TWO> >(num_elems);
            break;
        case 3 :
            printf("Mod<THREE>:\n");
            testMdiscr< Mod<THREE> >(num_elems);
            break;
        case 4 :
            printf("Mod<FOUR>:\n");
            testMdiscr< Mod<FOUR> >(num_elems);
            break;
        case 5 :
            printf("Mod<FIVE>:\n");
            testMdiscr< Mod<FIVE> >(num_elems);
            break;
        case 6 :
            printf("Mod<SIX>:\n");
            testMdiscr< Mod<SIX> >(num_elems);
            break;
        case 7 :
            printf("Mod<SEVEN>:\n");
            testMdiscr< Mod<SEVEN> >(num_elems);
            break;
        case 8 :
            printf("Mod<EIGHT>:\n");
            testMdiscr< Mod<EIGHT> >(num_elems);
            break;
        default :
            printf("Unsupported modulo operation.\n");
            return EXIT_FAILURE;
        }
    }
    
    return EXIT_SUCCESS;
}
