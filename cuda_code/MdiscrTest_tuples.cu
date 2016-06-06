#include "MdiscrHost_tuples.cu.h"
#include "HelpersHost.cu.h"

#include <stdio.h>
//#include "cuda_profiler_api.h"

/*
 * See http://www.gotw.ca/publications/mill17.htm for why this is defined
 * inside a struct.
 */
template<typename T>
struct populateImpl{
    static void populateArray(const unsigned int, T*);
};

template<> void
populateImpl<unsigned int>::populateArray(const unsigned int  num_elems,
                                          unsigned int*       in_array) {
    populateIntArray(num_elems, in_array);
}

template<> void
populateImpl<unsigned char>::populateArray(const unsigned int  num_elems,
                                           unsigned char*      in_array) {
    for(unsigned int i = 0; i < num_elems; i++) {
        in_array[i] = std::rand() % 20;
    }
}

template<> void
populateImpl<bool>::populateArray(const unsigned int  num_elems,
                                  bool*               in_array) {
    for(unsigned int i = 0; i < num_elems; i++) {
        in_array[i] = (std::rand() % 2 == 0);
    }
}

template<typename T>
void populateArray(const unsigned int  num_elems,
                   T*                  in_array) {
    populateImpl<T>::populateArray(num_elems, in_array);
}

template<class ModN>
int test(const unsigned int num_elems) {
    
    typedef typename ModN::InType1 T1;
    typedef typename ModN::InType2 T2;
    typedef typename ModN::InType3 T3;
    
    // Allocate memory.
    T1* h_in1                 = (T1*)            malloc(num_elems * sizeof(T1));
    T1* h_out1                = (T1*)            malloc(num_elems * sizeof(T1));
    T2* h_in2                 = (T2*)            malloc(num_elems * sizeof(T2));
    T2* h_out2                = (T2*)            malloc(num_elems * sizeof(T2));
    T3* h_in3                 = (T3*)            malloc(num_elems * sizeof(T3));
    T3* h_out3                = (T3*)            malloc(num_elems * sizeof(T3));
    unsigned int* h_out_sizes = (unsigned int*)  malloc(num_elems * sizeof(unsigned int));
    
    { // Initialize array.
        
        // Seed the random number generator.
        std::srand(time(NULL));
        
        populateArray<T1>(num_elems, h_in1);
        populateArray<T2>(num_elems, h_in2);
        populateArray<T3>(num_elems, h_in3);
    }
    
    T1 *d_in1, *d_out1;
    T2 *d_in2, *d_out2;
    T3 *d_in3, *d_out3;
    unsigned int *d_out_sizes;
    { // Device allocation.
        cudaMalloc((void**)&d_in1 ,   num_elems * sizeof(T1));
        cudaMalloc((void**)&d_out1,   num_elems * sizeof(T1));
        cudaMalloc((void**)&d_in2 ,   num_elems * sizeof(T2));
        cudaMalloc((void**)&d_out2,   num_elems * sizeof(T2));
        cudaMalloc((void**)&d_in3 ,   num_elems * sizeof(T3));
        cudaMalloc((void**)&d_out3,   num_elems * sizeof(T3));
        cudaMalloc((void**)&d_out_sizes, num_elems * sizeof(unsigned int));
        
        // Copy host memory to device.
        cudaMemcpy(d_in1, h_in1, num_elems * sizeof(T1), cudaMemcpyHostToDevice);
        cudaMemcpy(d_in2, h_in2, num_elems * sizeof(T2), cudaMemcpyHostToDevice);
        cudaMemcpy(d_in3, h_in3, num_elems * sizeof(T3), cudaMemcpyHostToDevice);
        cudaThreadSynchronize();
    }
    
    /* cudaProfilerStart(); */
    
    // Call the discriminator function.
    typename ModN::TupleType sizes = mdiscr<ModN>(num_elems, (1<<16),
                                                  d_in1, d_in2, d_in3,
                                                  d_out1, d_out2, d_out3);
    
    /* cudaProfilerStop(); */
    
    // Copy result back to host.
    cudaMemcpy(h_out1, d_out1, num_elems * sizeof(T1), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_out2, d_out2, num_elems * sizeof(T2), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_out3, d_out3, num_elems * sizeof(T3), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_out_sizes, d_out_sizes, num_elems * sizeof(unsigned int), cudaMemcpyDeviceToHost);
    
    // Free device memory.
    cudaFree(d_in1);
    cudaFree(d_in2);
    cudaFree(d_in3);
    cudaFree(d_out1);
    cudaFree(d_out2);
    cudaFree(d_out3);
    cudaFree(d_out_sizes);
    
    bool success = validateOneSegment<ModN>(h_in1, h_in2, h_in3,
                                            h_out1, h_out2, h_out3,
                                            sizes, num_elems);
    
    if (success) {
        fprintf(stderr, "mdiscr on %d elems: VALID RESULT!\n", num_elems);
    }
    else {
        fprintf(stderr, "mdiscr on %d elems: INVALID RESULT!\n", num_elems);
    }
    
    // Cleanup memory.
    free(h_in1);
    free(h_in2);
    free(h_in3);
    free(h_out1);
    free(h_out2);
    free(h_out3);
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
    fprintf(stderr, "Discriminator is ModTuple4\n");
    return test<ModTuple4>(num_elems);
#else
#error Unsupported number of equivalence classes!
#endif
#else
    fprintf(stderr, "Discriminator is ModTuple<%d>\n", NUM_CLASSES);
    return test< ModTuple<NUM_CLASSES> >(num_elems);
#endif
#endif
}
