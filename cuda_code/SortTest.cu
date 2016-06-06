#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <algorithm>
#include <cstdlib>

#include <thrust/version.h>
#include <iostream>

#include<sys/time.h>
#include<time.h>

#include <thrust/for_each.h>
#include <thrust/iterator/zip_iterator.h>

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess) 
    {
        fprintf(stderr,"GPUassert (%d): %s (%s, line %d)\n", code, cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

int timeval_subtract(struct timeval *result, struct timeval *t2, struct timeval *t1)
{
    unsigned int resolution=1000000;
    long int diff = (t2->tv_usec + resolution * t2->tv_sec) - (t1->tv_usec + resolution * t1->tv_sec);
    result->tv_sec = diff / resolution;
    result->tv_usec = diff % resolution;
    return (diff<0);
}

struct printf_functor{
    template<typename Tuple>
    __host__ __device__
    void operator()(Tuple t){
        printf("%d: %d\n",
               thrust::get<0>(t),
               thrust::get<1>(t));
    }
};

template<int NUM>
struct rand_mod_functor {
    __host__ __device__
    int operator()() {
        return rand() % NUM;
    }
};

template<>
struct rand_mod_functor<4> {
    __host__ __device__
    int operator()() {
        return rand() & 3;
    }
};

template<>
struct rand_mod_functor<8> {
    __host__ __device__
    int operator()() {
        return rand() & 7;
    }
};

// template<int NUM>
// struct mod_functor {
//     __host__ __device__
//     int operator()(int x) {
//         return x % NUM;
//     }
// };

// template<>
// struct mod_functor<4> {
//     __host__ __device__
//     int operator()(int x) {
//         return x & 3;
//     }
// };

// template<>
// struct mod_functor<8> {
//     __host__ __device__
//     int operator()(int x) {
//         return x & 7;
//     }
// };

int main(void)
{
    int major = THRUST_MAJOR_VERSION;
    int minor = THRUST_MINOR_VERSION;
    
    std::cout << "Thrust v" << major << "." << minor << std::endl;
    
    struct timeval t_start, t_end, t_diff;
    unsigned long int elapsed;
    
#ifdef PRINT
    thrust::host_vector<int> h_keys(20);
    thrust::host_vector<int> h_values(20);
#else
    thrust::host_vector<int> h_keys(50000000);
    thrust::host_vector<int> h_values(50000000);
#endif
    
    // generate
#ifndef NUM_CLASSES
    std::generate(h_keys.begin(), h_keys.end(), rand);
#else
    std::generate(h_keys.begin(), h_keys.end(), rand_mod_functor<NUM_CLASSES>());
#endif
    std::generate(h_values.begin(), h_values.end(), rand);
    
#ifdef PRINT
    // print
    thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(h_keys.begin(),
                                                                  h_values.begin())),
                     thrust::make_zip_iterator(thrust::make_tuple(h_keys.end(),
                                                                  h_values.end())),
                     printf_functor());
#endif
    
    // transfer data to the device
    thrust::device_vector<int> d_keys = h_keys;
    thrust::device_vector<int> d_values = h_values;
    
    // start the clock
    gettimeofday(&t_start, NULL);
    
    // sort data on the device
#ifdef ONLY_KEYS
    thrust::stable_sort(d_keys.begin(), d_keys.end());
    // thrust::stable_sort(d_values.begin(), d_values.end());
#else
    thrust::stable_sort_by_key(d_keys.begin(), d_keys.end(), d_values.begin());
#endif
    
    // wait for completion
    gpuErrchk( cudaDeviceSynchronize() );
    
    // stop the clock
    gettimeofday(&t_end, NULL);
    
    // calculate elapsed time
    timeval_subtract(&t_diff, &t_end, &t_start);
    elapsed = (t_diff.tv_sec*1e6+t_diff.tv_usec);
    printf("sort total runtime:   %6lu microsecs\n", elapsed);
    
    // transfer data back to host
    thrust::copy(d_keys.begin(), d_keys.end(), h_keys.begin());
    thrust::copy(d_values.begin(), d_values.end(), h_values.begin());
    
#ifdef PRINT
    // print
    thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(h_keys.begin(),
                                                                  h_values.begin())),
                     thrust::make_zip_iterator(thrust::make_tuple(h_keys.end(),
                                                                  h_values.end())),
                     printf_functor());
#endif
    
    return 0;
}
