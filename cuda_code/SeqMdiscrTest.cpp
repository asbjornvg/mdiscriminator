#include "MainCommon.h"
#include "SeqHelpers.h"
//#include "SeqMdiscr.h"

template<class ModN>
void test(  const unsigned int num_elems
    ) {
    
    // Allocate memory.
    typename ModN::InType* h_in =
        (typename ModN::InType*) malloc(num_elems * sizeof(typename ModN::InType));
    typename ModN::InType* h_out =
        (typename ModN::InType*) malloc(num_elems * sizeof(typename ModN::InType));
    int* h_out_sizes = (int*) malloc(num_elems * sizeof(int));
    
    { // Initialize array.
        
        // Seed the random number generator.
        time_t t;
        std::srand(time(&t));
        
        populateIntArray(num_elems, h_in);
    }
    //printIntArray(num_elems, "h_in", h_in);
    
    seqMdiscr<ModN, true>(num_elems, h_in, h_out, h_out_sizes);
    
    //printIntArray(num_elems, "h_out", h_out);
    //printIntArray(num_elems, "h_out_sizes", h_out_sizes);
    
    // No validation since this is the sequential version.
    
    free(h_in);
    free(h_out);
    free(h_out_sizes);
}
