#ifndef MDISCR_HOST_TUPLES
#define MDISCR_HOST_TUPLES

#include "MdiscrKernels_tuples.cu.h"
#include "HelpersHost.cu.h"

#include<iostream>

/*
 * This class takes an unsigned int, an unsigned char, and a bool, and produces
 * a number between 0 and 3, both included.
 */
class ModTuple4 {
public:
    typedef unsigned int   InType1;
    typedef unsigned char  InType2;
    typedef bool           InType3;
    // The out-type is always unsigned int for these discriminators.
    typedef MyInt4         TupleType;
    static const InType1 padelm1 = 3;
    static const InType2 padelm2 = 3;
    static const InType3 padelm3 = true;
    __device__ __host__ static inline unsigned int apply(volatile InType1 x,
                                                         volatile InType2 y,
                                                         volatile InType3 z) {
        return x & 3;
        //return z ? x & 3 : y & 3;
    }
};

template<unsigned int N>
class ModTuple {
public:
    typedef unsigned int   InType1;
    typedef unsigned char  InType2;
    typedef bool           InType3;
    // The out-type is always unsigned int for these discriminators.
    typedef MyInt<N>       TupleType;
    static const InType1 padelm1 = N-1;
    static const InType2 padelm2 = N-1;
    static const InType3 padelm3 = true;
    CUDA_DEVICE_HOST static inline unsigned int apply(volatile InType1 x,
                                                      volatile InType2 y,
                                                      volatile InType3 z) {
        return x % N;
        //return z ? x % N : y % N;
    }
};

#include<cstdio>
#include <sys/time.h>
#include <time.h>

int timeval_subtract(struct timeval *result, struct timeval *t2, struct timeval *t1);

template<class DISCR, bool MEASURE>
void seqMdiscr(const unsigned int        num_elems,
               typename DISCR::InType1*  in_array1,   // host
               typename DISCR::InType2*  in_array2,   // host
               typename DISCR::InType3*  in_array3,   // host
               typename DISCR::InType1*  out_array1,  // host
               typename DISCR::InType2*  out_array2,  // host
               typename DISCR::InType3*  out_array3,  // host
               unsigned int*             sizes_array  // host
    ) {
    
    // Time measurement data structures.
    struct timeval t_start, t_end, t_diff;
    unsigned long int elapsed;
    
    unsigned int i, eq_class, index;
    typename DISCR::TupleType reduction, offsets, count;
    typename DISCR::InType1 in_el1;
    typename DISCR::InType2 in_el2;
    typename DISCR::InType3 in_el3;
    
    if (MEASURE) {
        gettimeofday(&t_start, NULL);
    }
    
    for (i = 0; i < num_elems; i++) {
        eq_class = DISCR::apply(in_array1[i],
                                in_array2[i],
                                in_array3[i]);
        reduction[eq_class]++;
    }
    
    unsigned int tmp = 0;
    for(eq_class = 0; eq_class < DISCR::TupleType::cardinal; eq_class++) {
        offsets[eq_class] = tmp;
        tmp += reduction[eq_class];
    }
    
    for (i = 0; i < num_elems; i++) {
        in_el1 = in_array1[i];
        in_el2 = in_array2[i];
        in_el3 = in_array3[i];
        
        eq_class = DISCR::apply(in_el1, in_el2, in_el3);
        
        index = count[eq_class] + offsets[eq_class];
        out_array1[index] = in_el1;
        out_array2[index] = in_el2;
        out_array3[index] = in_el3;
        count[eq_class]++;
        
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

template<class ModN>
bool validateOneSegment(typename ModN::InType1*   in_array1,
                        typename ModN::InType2*   in_array2,
                        typename ModN::InType3*   in_array3,
                        typename ModN::InType1*   out_array1,
                        typename ModN::InType2*   out_array2,
                        typename ModN::InType3*   out_array3,
                        typename ModN::TupleType  sizes,
                        unsigned int              num_elems
    ) {
    
    bool success = true;
    
    // Allocate memory for the comparison (sequential) run.
    typename ModN::InType1* out_array1_cmp =
        (typename ModN::InType1*) malloc(num_elems * sizeof(typename ModN::InType1));
    typename ModN::InType2* out_array2_cmp =
        (typename ModN::InType2*) malloc(num_elems * sizeof(typename ModN::InType2));
    typename ModN::InType3* out_array3_cmp =
        (typename ModN::InType3*) malloc(num_elems * sizeof(typename ModN::InType3));
    unsigned int* sizes_array_cmp = (unsigned int*) malloc(num_elems * sizeof(unsigned int));
    
    // Allocate storage for an actual size-array.
    unsigned int* sizes_array = (unsigned int*) malloc(num_elems * sizeof(unsigned int));
    
    // Convert the tuple into a size-array.
    tupleToSizeArray<typename ModN::TupleType>(sizes, sizes_array, num_elems);
    
    // Run the sequential version.
    seqMdiscr<ModN, false>(num_elems, in_array1, in_array2, in_array3,
                           out_array1_cmp, out_array2_cmp, out_array3_cmp,
                           sizes_array_cmp);
    
    if (!compareSegmentToOther<unsigned int>(sizes_array, sizes_array_cmp, num_elems, "size")) {
        success = false;
    }
    if (!compareSegmentToOther<typename ModN::InType1>(out_array1, out_array1_cmp, num_elems, "element-0")) {
        success = false;
    }
    if (!compareSegmentToOther<typename ModN::InType2>(out_array2, out_array2_cmp, num_elems, "element-1")) {
        success = false;
    }
    if (!compareSegmentToOther<typename ModN::InType3>(out_array3, out_array3_cmp, num_elems, "element-2")) {
        success = false;
    }
    
    // unsigned int i, j;
    // unsigned int size = 0;
    // unsigned int size_cmp = 0;
    // i = 0;
    // j = 0;
    // for ( ; i < num_elems; i += size, j++) {
    //     size = sizes[j];
    //     size_cmp = sizes_array_cmp[i];
    //     if (size != size_cmp) {
    //         success = false;
    //         fprintf(stderr, "Invalid size: %d should be %d (i = %d, j = %d)\n", size, size_cmp, i, j);
    //     }
    // }
    
    // typename ModN::InType1 out_element1, out_element1_cmp;
    // typename ModN::InType2 out_element2, out_element2_cmp;
    // typename ModN::InType3 out_element3, out_element3_cmp;
    // for(i = 0; i < num_elems; i++) {
        
    //     // Compare elements.
    //     out_element1 = out_array1[i];
    //     out_element1_cmp = out_array1_cmp[i];
    //     if (out_element1 != out_element1_cmp) {
    //         success = false;
    //         fprintf(stderr, "Violation: %d should be %d (i = %d)\n", out_element1, out_element1_cmp, i);
    //     }
    //     out_element2 = out_array2[i];
    //     out_element2_cmp = out_array2_cmp[i];
    //     if (out_element2 != out_element2_cmp) {
    //         success = false;
    //         fprintf(stderr, "Violation: %d should be %d (i = %d)\n", out_element2, out_element2_cmp, i);
    //     }
    //     out_element3 = out_array3[i];
    //     out_element3_cmp = out_array3_cmp[i];
    //     if (out_element3 != out_element3_cmp) {
    //         success = false;
    //         fprintf(stderr, "Violation: %d should be %d (i = %d)\n", out_element3, out_element3_cmp, i);
    //     }
        
    //     // Only print the first few violations.
    //     if (!success && i > 9) {
    //         break;
    //     }
    // }
    
    // Free memory.
    free(out_array1_cmp);
    free(out_array2_cmp);
    free(out_array3_cmp);
    free(sizes_array_cmp);
    free(sizes_array);
    
    return success;
}

template<class T1, class T2, class T3, int TILE>
void transposePad( T1*                  inp1_d,
                   T2*                  inp2_d,
                   T3*                  inp3_d,
                   T1*                  out1_d,
                   T2*                  out2_d,
                   T3*                  out3_d,
                   const unsigned int   height,
                   const unsigned int   width,
                   const unsigned int   oinp_size,
                   T1                   pad_elem1,
                   T2                   pad_elem2,
                   T3                   pad_elem3
    ) {
    
    // Number of iterations on the y-axis.
    int elements_pr_thread = 4;
    
    // The tile dimension must be divisible by elements_pr_thread.
    assert(TILE % elements_pr_thread == 0);
    
    // 1. setup block and grid parameters
    
    // Instead of one element per thread, each thread works on
    // elements_pr_thread elements.
    dim3 block(TILE, TILE/elements_pr_thread, 1);
    //dim3 block(TILE, TILE, 1);
    
    int  dimy = ceil( ((float)height)/TILE ); 
    int  dimx = ceil( ((float) width)/TILE );
    dim3 grid (dimx, dimy, 1);
    
    // 2. execute the kernel
    matTransposeTiledPadKer<T1,T2,T3,TILE> <<< grid, block >>>
        (inp1_d, inp2_d, inp3_d, out1_d, out2_d, out3_d,
         height, width, oinp_size,
         pad_elem1, pad_elem2, pad_elem3);
}

template<class DISCR>
typename DISCR::TupleType
mdiscr(const unsigned int        num_elems,
       const unsigned int        num_hwd_thds,
       typename DISCR::InType1*  d_in1,  // device
       typename DISCR::InType2*  d_in2,  // device
       typename DISCR::InType3*  d_in3,  // device
       typename DISCR::InType1*  d_out1, // device
       typename DISCR::InType2*  d_out2, // device
       typename DISCR::InType3*  d_out3  // device
) {
    //const unsigned int MAX_CHUNK = 384; //224; //128; //96; //256; //384;
    /* You need to define through preprocessor options on command-line, e.g.,
     * -DMAX_CHUNK=384.
     */
    
    const unsigned int WIDTH_MULT = lcm(MAP_Y, WRITE_Y);
    const unsigned int HEIGHT_MULT = lcm(MAP_X, WRITE_X);
    
    const unsigned int D_WIDTH = nextMultOf(min(max(num_elems/num_hwd_thds,1),
                                                MAX_CHUNK),
                                            WIDTH_MULT);
    
    const unsigned int D_HEIGHT = nextMultOf((num_elems + D_WIDTH - 1) / D_WIDTH,
                                             HEIGHT_MULT);
    
    const unsigned int PADD = nextMultOf(D_HEIGHT*D_WIDTH, HEIGHT_MULT*D_WIDTH) - num_elems;
    
    struct timeval t_start, t_med0, t_med1, t_med2, t_end, t_diff;
    unsigned long int elapsed;
    
    typename DISCR::InType1 *d_tr_in1;
    typename DISCR::InType2 *d_tr_in2;
    typename DISCR::InType3 *d_tr_in3;
    unsigned int *cond_res;
    typename DISCR::TupleType *inds_res;
    typename DISCR::TupleType  filt_size;
    
    gpuErrchk( cudaMalloc((void**)&d_tr_in1,
                          D_HEIGHT*D_WIDTH*sizeof(typename DISCR::InType1)) );
    gpuErrchk( cudaMalloc((void**)&d_tr_in2,
                          D_HEIGHT*D_WIDTH*sizeof(typename DISCR::InType2)) );
    gpuErrchk( cudaMalloc((void**)&d_tr_in3,
                          D_HEIGHT*D_WIDTH*sizeof(typename DISCR::InType3)) );
    gpuErrchk( cudaMalloc((void**)&cond_res,
                          D_HEIGHT*D_WIDTH*sizeof(unsigned int)) );
    gpuErrchk( cudaMalloc((void**)&inds_res,
                          2*D_HEIGHT*sizeof(typename DISCR::TupleType)) );
    
    gettimeofday(&t_start, NULL);
    { // 1. Transpose with padding!
        transposePad<typename DISCR::InType1,typename DISCR::InType2,
                     typename DISCR::InType3,32>
            (d_in1, d_in2, d_in3, d_tr_in1, d_tr_in2, d_tr_in3,
             D_HEIGHT, D_WIDTH, num_elems,
             DISCR::padelm1, DISCR::padelm2, DISCR::padelm3);
    }
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
    gettimeofday(&t_med0, NULL);
    
    { // 2. The Map Condition Kernel Call
        //const unsigned int MAP_X       = 64;
        //const unsigned int MAP_Y       = 1;
        /* You need to define through preprocessor options on command-line,
         * e.g., -DMAP_X=64 -DMAP_Y=1.
         */
        
        //const unsigned int block_size = 32; //64; //256;
        const unsigned int num_blocks  = (D_HEIGHT + MAP_X - 1) / MAP_X;
        const unsigned int SEQ_CHUNK   = D_WIDTH / MAP_Y;
        const unsigned int SH_MEM_MAP  = MAP_X * MAP_Y * sizeof(typename DISCR::TupleType);
        fprintf(stderr, "SH_MEM_MAP = %d\n", SH_MEM_MAP);
        
        dim3 block(MAP_X, MAP_Y, 1);
        dim3 grid(num_blocks, 1, 1);
        
        // map the condition
        mapVctKernel<DISCR,MAP_Y><<<grid, block, SH_MEM_MAP>>>
            (d_tr_in1, d_tr_in2, d_tr_in3,
             cond_res, inds_res, D_HEIGHT, SEQ_CHUNK);
    }
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
    gettimeofday(&t_med1, NULL);
    
    { // 3. the inclusive scan of the condition results
        const unsigned int block_size = 128;
        scanInc<Add<typename DISCR::TupleType>,typename DISCR::TupleType>
                (block_size, D_HEIGHT, inds_res, inds_res+D_HEIGHT);
        gpuErrchk( cudaPeekAtLastError() );
        
        gpuErrchk( cudaMemcpy( &filt_size, &inds_res[2*D_HEIGHT - 1],
                               sizeof(typename DISCR::TupleType),
                               cudaMemcpyDeviceToHost ) );
        
        // The padding shouldn't count in the totals.
        filt_size[DISCR::apply(DISCR::padelm1,
                               DISCR::padelm2,
                               DISCR::padelm3)] -= PADD;
    }
    //gpuErrchk( cudaDeviceSynchronize() );
    gettimeofday(&t_med2, NULL);
    
    { // 4. the write to global memory part
        
        //const unsigned int WRITE_X     = 32;
        //const unsigned int WRITE_Y     = 32;
        /* You need to define through preprocessor options on command-line,
         * e.g., -DWRITE_X=32 -DWRITE_Y=32.
         */
        
        const unsigned int num_blocks  = (D_HEIGHT + WRITE_X - 1) / WRITE_X;
        const unsigned int SEQ_CHUNK   = D_WIDTH / WRITE_Y;
        
        const unsigned int SH_MEM_SIZE =
            std::max(WRITE_Y * WRITE_X * SEQ_CHUNK *
                     (sizeof(typename DISCR::InType1) +
                      sizeof(typename DISCR::InType2) +
                      sizeof(typename DISCR::InType3)),
                     (WRITE_Y + 1) * WRITE_X * sizeof(typename DISCR::TupleType));
        
        fprintf(stderr, "SH_MEM_SIZE = %d\n", SH_MEM_SIZE);
        
        dim3 block(WRITE_X, WRITE_Y, 1);
        dim3 grid(num_blocks, 1, 1);
        
        writeMultiKernel<DISCR,WRITE_Y><<<grid, block, SH_MEM_SIZE>>>
            (d_tr_in1, d_tr_in2, d_tr_in3, cond_res, inds_res+D_HEIGHT,
             d_out1, d_out2, d_out3, D_HEIGHT, num_elems, SEQ_CHUNK);
    }
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
    gettimeofday(&t_end, NULL);
    
    timeval_subtract(&t_diff, &t_end, &t_start);
    elapsed = (t_diff.tv_sec*1e6+t_diff.tv_usec);
    printf("%d\n", elapsed);
    fprintf(stderr, "mdiscr total runtime:             %6lu microsecs, from which:\n", elapsed);
    
    timeval_subtract(&t_diff, &t_med0, &t_start);
    elapsed = (t_diff.tv_sec*1e6+t_diff.tv_usec);
    fprintf(stderr, "\t%-25s %6lu microsecs\n", "Transposition:", elapsed);
    
    timeval_subtract(&t_diff, &t_med1, &t_med0);
    elapsed = (t_diff.tv_sec*1e6+t_diff.tv_usec);
    printf("%d\n", elapsed);
    fprintf(stderr, "\t%-25s %6lu microsecs\n", "Map Cond Kernel:", elapsed);
    
    timeval_subtract(&t_diff, &t_med2, &t_med1);
    elapsed = (t_diff.tv_sec*1e6+t_diff.tv_usec);
    fprintf(stderr, "\t%-25s %6lu microsecs\n", "Scan Addition Kernel:", elapsed);
    
    timeval_subtract(&t_diff, &t_end, &t_med2);
    elapsed = (t_diff.tv_sec*1e6+t_diff.tv_usec);
    printf("%d\n", elapsed);
    fprintf(stderr, "\t%-25s %6lu microsecs\n", "Global Write Kernel:", elapsed);
    
    // free resources
    cudaFree(inds_res);
    cudaFree(cond_res);
    cudaFree(d_tr_in3);
    cudaFree(d_tr_in2);
    cudaFree(d_tr_in1);
    
    return filt_size;
    
}

#endif //MDISCR_HOST_TUPLES
