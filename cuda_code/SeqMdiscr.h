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
               unsigned int*            sizes_array // host
    ) {
    
    // Time measurement data structures.
    struct timeval t_start, t_end, t_diff;
    unsigned long int elapsed;
    
    unsigned int i, eq_class;
    typename DISCR::TupleType reduction, offsets, count;
    typename DISCR::InType in_el;
    
    if (MEASURE) {
        gettimeofday(&t_start, NULL);
    }
    
    for (i = 0; i < num_elems; i++) {
        eq_class = DISCR::apply(in_array[i]);
        reduction[eq_class]++;
    }
    
    unsigned int tmp = 0;
    for(eq_class = 0; eq_class < DISCR::TupleType::cardinal; eq_class++) {
        offsets[eq_class] = tmp;
        tmp += reduction[eq_class];
    }
    
    for (i = 0; i < num_elems; i++) {
        in_el = in_array[i];
        eq_class = DISCR::apply(in_el);
        out_array[(count[eq_class]++) + offsets[eq_class]] = in_el;
        sizes_array[i] = 0;
    }
    
    for(eq_class = 0; eq_class < DISCR::TupleType::cardinal; eq_class++) {
        sizes_array[offsets[eq_class]] = reduction[eq_class];
    }
    
    if (MEASURE) {
        gettimeofday(&t_end, NULL);
        timeval_subtract(&t_diff, &t_end, &t_start);
        elapsed = (t_diff.tv_sec*1e6+t_diff.tv_usec);
        printf("seqMdiscr total runtime:   %6lu microsecs\n", elapsed);
    }
}

#endif //SEQ_MDISCR
