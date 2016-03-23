#ifndef SEQ_MDISCR
#define SEQ_MDISCR

#include<cstdio>
#include <sys/time.h>
#include <time.h>

int timeval_subtract(struct timeval *result, struct timeval *t2, struct timeval *t1);

template<class DISCR, bool MEASURE>
void seqMdiscr(const unsigned int       num_elems,
               typename DISCR::InType*  in_array,   // host
               typename DISCR::InType*  out_array,  // host
               int*                     sizes_array // host
    ) {
    
    // Time measurement data structures.
    struct timeval t_start, t_end, t_diff;
    unsigned long int elapsed;
    
    int* classes = (int*) malloc(num_elems * sizeof(int));
    typename DISCR::TupleType *scan_results =
        (typename DISCR::TupleType*) malloc(num_elems * sizeof(typename DISCR::TupleType));
    
    unsigned int i, k;
    
    if (MEASURE) {
        gettimeofday(&t_start, NULL);
    }
    
    typename DISCR::TupleType accum_tuple(0);
    for (i = 0; i < num_elems; i++) {
        int eq_class = DISCR::apply(in_array[i]);
        classes[i] = eq_class;
        typename DISCR::TupleType tuple(0);
        tuple[eq_class] = 1;
        
        // Scan
        accum_tuple = accum_tuple + tuple;
        scan_results[i] = accum_tuple;
    }
    
    typename DISCR::TupleType reduction = scan_results[num_elems-1];
    
    typename DISCR::TupleType offsets;
    unsigned int tmp = 0;
    for(k = 0; k < DISCR::TupleType::cardinal; k++) {
        offsets[k] = tmp;
        tmp += reduction[k];
    }
    
    for (i = 0; i < num_elems; i++) {
        int k = classes[i];
        int scan_result_k = scan_results[i][k];
        int offset_k = offsets[k];
        
        int index = scan_result_k + offset_k - 1;
        
        out_array[index] = in_array[i];
        
        sizes_array[i] = 0;
        for (unsigned int k = 0; k < DISCR::TupleType::cardinal; k++) {
            int reduction_k = reduction[k];
            int offset_k = offsets[k];
            if (i == offset_k && reduction_k > 0) {
                sizes_array[i] = reduction_k;
            }
        }
    }
    
    if (MEASURE) {
        gettimeofday(&t_end, NULL);
        timeval_subtract(&t_diff, &t_end, &t_start);
        elapsed = (t_diff.tv_sec*1e6+t_diff.tv_usec); 
        printf("seqMdiscr total runtime:   %6lu microsecs\n", elapsed);
    }
    
    free(classes);
    free(scan_results);
}

#endif //SEQ_MDISCR
