#include <stdio.h>

#include "MdiscrHost.cu.h"

template<class ModN>
int testMdiscr(  const unsigned int num_elems
    ) {
    
    unsigned int mem_size = num_elems * sizeof(typename ModN::InType);
    
    // Allocate memory.
    typename ModN::InType* h_in    = (typename ModN::InType*) malloc(mem_size);
    typename ModN::InType* h_out   = (typename ModN::InType*) malloc(mem_size);
    
    { // Initialize array.
        
        // int tmp[13] = {5,4,2,3,7,8,6,4,1,9,11,12,10};
        // memcpy(h_in, tmp, 13 * sizeof(int));
        
        std::srand(33);
        for(unsigned int i = 0; i < num_elems; i++) {
            h_in[i] = std::rand(); //% 50;
        }
    }
    
    /* for(unsigned int i = 0; i < num_elems; i++) { */
    /*     printf("h_in[%d] = %d\n", i, h_in[i]); */
    /* } */
    /* for(unsigned int i = 0; i < 11; i++) { */
    /*     printf("h_in[%d] = %d\n", i, h_in[i]); */
    /* } */
    /* printf("...\n"); */
    
    typename ModN::InType *d_in, *d_out;
    { // Device allocation.
        cudaMalloc((void**)&d_in ,   mem_size);
        cudaMalloc((void**)&d_out,   mem_size);
        
        // Copy host memory to device.
        cudaMemcpy(d_in, h_in, mem_size, cudaMemcpyHostToDevice);
        cudaThreadSynchronize();
    }
    
    // Call the discriminator function.
    typename ModN::TupleType sizes = mdiscr<Mod4>( num_elems, d_in, d_out );
    
    // Copy result back to host.
    cudaMemcpy(h_out, d_out, mem_size, cudaMemcpyDeviceToHost);
    cudaThreadSynchronize();
    
    // Free device memory.
    cudaFree(d_in );
    cudaFree(d_out);
        
    /* for(unsigned int i = 0; i < num_elems; i++) { */
    /*     printf("h_out[%d] = %d\n", i, h_out[i]); */
    /* } */
    /* for(unsigned int i = 0; i < 11; i++) { */
    /*     printf("h_out[%d] = %d\n", i, h_out[i]); */
    /* } */
    /* printf("...\n"); */
    
    /* for(int k = 0; k < ModN::TupleType::cardinal; k++) { */
    /*     printf("sizes[%d] = %d\n", k, sizes[k]); */
    /* } */
        
    /*************
     * Validation
     *************/
    bool success = true;
    
    // Compare the sizes of the equivalence classes.
    typename ModN::TupleType sizes_cmp;
    for(int k = 0; k < ModN::TupleType::cardinal; k++) {
        sizes_cmp[k] = 0;
    }
    for(int i = 0; i < num_elems; i++) {
        // Increment the size of the corresponding equivalence class.
        sizes_cmp[ModN::apply(h_in[i])]++;
    }
    for(int i = 0; i < ModN::TupleType::cardinal; i++) {
        if ( sizes[i] != sizes_cmp[i] ) {
            success = false;
            printf( "Invalid size #%d, computed: %d, should be: %d!!! EXITING!\n\n", i, sizes[i], sizes_cmp[i]);
        }
    }
    if (success) {
        // "Exclusive scan" of the sizes tuple to produce the starting point offsets
        // for each equivalence class.
        unsigned int tmp = 0;
        typename ModN::TupleType offsets;
        for(int k = 0; k < ModN::TupleType::cardinal; k++) {
            offsets[k] = tmp;
            tmp += sizes[k];
        }
    
        // The "current" offsets into each equivalence class.
        typename ModN::TupleType count;
        for(int i = 0; i < ModN::TupleType::cardinal; i++) {
            count[i] = 0;
        }

        for(int i = 0; i < num_elems; i++) {
            int in_el = h_in[i];
            int eq_class = ModN::apply(in_el);
            int out_el  = h_out[count[eq_class] + offsets[eq_class]];
            if ( out_el != in_el) {
                success = false;
                printf("mdiscr violation: %d should be %d, eq class: %d, i: %d\n", out_el, in_el, eq_class, i);
                if(i > 9) break;
            }
            count[eq_class]++;
        }
    }
    if (success) {
        printf("mdiscr on %d elems: VALID RESULT!\n", num_elems);
    }
    else {
        printf("mdiscr on %d elems: INVALID RESULT!\n", num_elems);
    }
    
    // Cleanup memory.
    free(h_in );
    free(h_out);

    return 0;
    
}

int main(int argc, char** argv) {
    
    const unsigned int num_elems = 40000000; //50332001;
    
    return testMdiscr<Mod4>(num_elems);
}
