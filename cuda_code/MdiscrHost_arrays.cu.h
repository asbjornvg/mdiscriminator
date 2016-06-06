#ifndef MDISCR_HOST_ARRAYS
#define MDISCR_HOST_ARRAYS

#include "MdiscrKernels_arrays.cu.h"
#include "HelpersHost.cu.h"

#include<iostream>
#include<sstream>

/*
 * This class takes an array of unsigned int (of length LEN), and produces a
 * number between 0 and 3, both included.
 */
template<unsigned int LEN>
class ModArray4 {
public:
    typedef unsigned int   InType;
    // The out-type is always unsigned int for these discriminators.
    typedef MyInt4         TupleType;
    static const InType padelm = 3;
    __device__ __host__ static inline unsigned int apply(volatile InType* x) {
        return x[0] & 3;
    }
};

template<unsigned int N, unsigned int LEN>
class ModArray {
public:
    typedef unsigned int   InType;
    // The out-type is always unsigned int for these discriminators.
    typedef MyInt<N>       TupleType;
    static const InType padelm = N-1;
    CUDA_DEVICE_HOST static inline unsigned int apply(volatile InType* x) {
        return x[0] % N;
    }
};

#include<cstdio>
#include <sys/time.h>
#include <time.h>

int timeval_subtract(struct timeval *result, struct timeval *t2, struct timeval *t1);

template<class DISCR, unsigned int LEN, bool MEASURE>
void seqMdiscr(const unsigned int       num_elems,
               typename DISCR::InType*  in_array,   // host
               typename DISCR::InType*  out_array,  // host
               unsigned int*            sizes_array // host
    ) {
    
    // Time measurement data structures.
    struct timeval t_start, t_end, t_diff;
    unsigned long int elapsed;
    
    unsigned int i, eq_class, index, d;
    typename DISCR::TupleType reduction, offsets, count;
    
    if (MEASURE) {
        gettimeofday(&t_start, NULL);
    }
    
    for (i = 0; i < LEN * num_elems; i += LEN) {
        eq_class = DISCR::apply(in_array + i);
        reduction[eq_class]++;
    }
    
    unsigned int tmp = 0;
    for(eq_class = 0; eq_class < DISCR::TupleType::cardinal; eq_class++) {
        offsets[eq_class] = tmp;
        tmp += reduction[eq_class];
    }
    
    for (i = 0; i < num_elems; i++) {
        eq_class = DISCR::apply(in_array + i*LEN);
        
        index = count[eq_class] + offsets[eq_class];
        
        for (d = 0; d < LEN; d++) {
            out_array[index*LEN + d] = in_array[i*LEN + d];
        }
        
        sizes_array[i] = 0;
        count[eq_class]++;
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

template<class ModN, unsigned int LEN>
bool validateOneSegment(typename ModN::InType*    in_array,
                        typename ModN::InType*    out_array,
                        typename ModN::TupleType  sizes,
                        unsigned int              num_elems
    ) {
    
    unsigned int d;
    bool success = true;
    
    // Allocate memory for the comparison (sequential) run.
    typename ModN::InType* out_array_cmp =
        (typename ModN::InType*) malloc(LEN * num_elems * sizeof(typename ModN::InType));
    unsigned int* sizes_array_cmp = (unsigned int*) malloc(num_elems * sizeof(unsigned int));
    
    // Allocate storage for an actual size-array.
    unsigned int* sizes_array = (unsigned int*) malloc(num_elems * sizeof(unsigned int));
    
    // Convert the tuple into a size-array.
    tupleToSizeArray<typename ModN::TupleType>(sizes, sizes_array, num_elems);
    
    // Run the sequential version.
    seqMdiscr<ModN, LEN, false>(num_elems, in_array, out_array_cmp, sizes_array_cmp);
    
    if (!compareSegmentToOther<unsigned int>(sizes_array, sizes_array_cmp, num_elems, "size")) {
        success = false;
    }
    
    for (d = 0; d < LEN; d++) {
        std::stringstream sstm;
        sstm << "element-" << d;
        std::string s = sstm.str();
        if (!compareSegmentToOtherGeneral<typename ModN::InType, LEN>
            (out_array, out_array_cmp, num_elems, s, d)) {
            success = false;
        }
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
    
    // typename ModN::InType out_element, out_element_cmp;
    // for(i = 0; i < LEN * num_elems; i++) {
        
    //     // Compare elements.
    //     out_element = out_array[i];
    //     out_element_cmp = out_array_cmp[i];
    //     if (out_element != out_element_cmp) {
    //         success = false;
    //         fprintf(stderr, "Violation: %d should be %d (i = %d)\n", out_element, out_element_cmp, i);
    //     }
        
    //     // Only print the first few violations.
    //     if (!success && i > 9) {
    //         break;
    //     }
    // }
    
    // Free memory.
    free(out_array_cmp);
    free(sizes_array_cmp);
    free(sizes_array);
    
    return success;
}

template<class T, int TILE, unsigned int LEN>
void transposePadArray( T*                   inp_d,
                        T*                   out_d,
                        const unsigned int   height,
                        const unsigned int   width,
                        const unsigned int   oinp_size,
                        T                    pad_elem
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
    matTransposeTiledPadKerArray<T,TILE,LEN> <<< grid, block >>>
        (inp_d, out_d, height, width, oinp_size, pad_elem);
}

template<class T, unsigned int LEN_X, bool REVERSE>
void inline transposeSpecialCommon( T*                  inp_d,
                                    T*                  out_d,
                                    const unsigned int  height
    ) {
    
    // Number of iterations on the y-axis.
    const unsigned int elements_pr_thread = 4;
    const unsigned int threads_x = LEN_X;
    const unsigned int threads_y = 32;
    const unsigned int num_threads = threads_x * threads_y;
    const unsigned int len_y = threads_y * elements_pr_thread;
    const unsigned int num_blocks = ceil( ((float)height)/len_y );
    
    fprintf(stderr, "num_threads = %d, num_blocks = %d\n", num_threads, num_blocks);
    
    if (REVERSE) {
        matTransposeSpecialReverse<T,LEN_X,len_y> <<< num_blocks, num_threads >>>
            (inp_d, out_d, height);
    }
    else {
        matTransposeSpecial<T,LEN_X,len_y> <<< num_blocks, num_threads >>>
            (inp_d, out_d, height);
    }
    
    
}

template<class T, unsigned int LEN_X>
void transposeSpecial( T*                  inp_d,
                       T*                  out_d,
                       const unsigned int  height
    ) {
    transposeSpecialCommon<T, LEN_X, false>(inp_d, out_d, height);
}

template<class T, unsigned int LEN_X>
void transposeSpecialReverse( T*                  inp_d,
                              T*                  out_d,
                              const unsigned int  height
    ) {
    transposeSpecialCommon<T, LEN_X, true>(inp_d, out_d, height);
}

template<class DISCR, unsigned int LEN>
typename DISCR::TupleType
mdiscr(const unsigned int        num_elems,
       const unsigned int        num_hwd_thds,
       typename DISCR::InType*  d_in,  // device
       typename DISCR::InType*  d_out  // device
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
    
    struct timeval t_start, t_med0a, t_med0, t_med1, t_med2, t_endb, t_end, t_diff;
    unsigned long int elapsed;
    
    typename DISCR::InType *d_tr_in1;
    typename DISCR::InType *d_tr_in2;
    typename DISCR::InType *d_tr_out;
    unsigned int *cond_res;
    typename DISCR::TupleType *inds_res;
    typename DISCR::TupleType  filt_size;
    
    fprintf(stderr, "LEN*num_elems*sizeof(typename DISCR::InType) = %d\n", LEN*num_elems*sizeof(typename DISCR::InType));
    fprintf(stderr, "D_HEIGHT*D_WIDTH*LEN*sizeof(typename DISCR::InType) = %d\n", D_HEIGHT*D_WIDTH*LEN*sizeof(typename DISCR::InType));
    
    gpuErrchk( cudaMalloc((void**)&d_tr_in1,
                          LEN*num_elems*sizeof(typename DISCR::InType)) );
    gpuErrchk( cudaMalloc((void**)&d_tr_in2,
                          D_HEIGHT*D_WIDTH*LEN*sizeof(typename DISCR::InType)) );
    // gpuErrchk( cudaMalloc((void**)&d_tr_out,
    //                       LEN*num_elems*sizeof(typename DISCR::InType)) );
    d_tr_out = d_tr_in1;
    gpuErrchk( cudaMalloc((void**)&cond_res,
                          D_HEIGHT*D_WIDTH*sizeof(unsigned int)) );
    gpuErrchk( cudaMalloc((void**)&inds_res,
                          2*D_HEIGHT*sizeof(typename DISCR::TupleType)) );
    
    gettimeofday(&t_start, NULL);
    {
        // 1a. Transpose from array (of length num_elems) of elements that are
        // themselves arrays (of length LEN) to array (of length LEN) of
        // arrays (of length num_elems).
        // transposePad<typename DISCR::InType,16>
        //     (d_in, d_tr_in1, num_elems, LEN, LEN * num_elems, DISCR::padelm);
        transposeSpecial<typename DISCR::InType,LEN>
            (d_in, d_tr_in1, num_elems);
    }
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
    gettimeofday(&t_med0a, NULL);
    
    // const unsigned int h2 = 5;
    // const unsigned int w2 = 2;
    // unsigned int p2 = 42;
    
    { // 1. Transpose with padding!
        transposePadArray<typename DISCR::InType,32,LEN>
            (d_tr_in1, d_tr_in2, D_HEIGHT, D_WIDTH, num_elems, DISCR::padelm);
        // transposePadArray<typename DISCR::InType,32,LEN>
        //     (d_tr_in1, d_tr_in2, h2, w2, num_elems, p2);
    }
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
    gettimeofday(&t_med0, NULL);
    
    // typename DISCR::InType *tmp1, *tmp2;
    // tmp1 = (typename DISCR::InType*) malloc(LEN * num_elems * sizeof(typename DISCR::InType));
    // tmp2 = (typename DISCR::InType*) malloc(D_HEIGHT * D_WIDTH * LEN * sizeof(typename DISCR::InType));
    // gpuErrchk( cudaMemcpy(tmp1, d_tr_in1, LEN * num_elems * sizeof(typename DISCR::InType),
    //                       cudaMemcpyDeviceToHost) );
    // gpuErrchk( cudaMemcpy(tmp2, d_tr_in2, D_HEIGHT * D_WIDTH * LEN * sizeof(typename DISCR::InType),
    //                       cudaMemcpyDeviceToHost) );
    // printIntArray(LEN * num_elems, "tmp1", tmp1);
    // printIntArray(D_HEIGHT * D_WIDTH * LEN, "tmp2", tmp2);
    // free(tmp1);
    // free(tmp2);
    
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
        mapVctKernel<DISCR,MAP_Y,LEN><<<grid, block, SH_MEM_MAP>>>
            (d_tr_in2, cond_res, inds_res, D_HEIGHT, D_WIDTH, SEQ_CHUNK);
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
        typename DISCR::InType arr[LEN];
        for (unsigned int d = 0; d < LEN; d++) {
            arr[d] = DISCR::padelm;
        }
        
        filt_size[DISCR::apply(arr)] -= PADD;
    }
    //gpuErrchk( cudaDeviceSynchronize() );
    gettimeofday(&t_med2, NULL);
    
    { // 4. the write to global memory part
        // By construction: D_WIDTH  is guaranteed to be a multiple of 32 AND
        //                  D_HEIGHT is guaranteed to be a multiple of 64 !!!
        
        //const unsigned int WRITE_X     = 32;
        //const unsigned int WRITE_Y     = 32;
        /* You need to define through preprocessor options on command-line,
         * e.g., -DWRITE_X=32 -DWRITE_Y=32.
         */
        
        const unsigned int num_blocks  = (D_HEIGHT + WRITE_X - 1) / WRITE_X;
        const unsigned int SEQ_CHUNK   = D_WIDTH / WRITE_Y;
        
        const unsigned int SH_MEM_SIZE =
            std::max(WRITE_Y * WRITE_X * SEQ_CHUNK *
                     LEN * sizeof(typename DISCR::InType),
                     (WRITE_Y + 1) * WRITE_X * sizeof(typename DISCR::TupleType));
        
        fprintf(stderr, "SH_MEM_SIZE = %d\n", SH_MEM_SIZE);
        
        dim3 block(WRITE_X, WRITE_Y, 1);
        dim3 grid(num_blocks, 1, 1);
        
        // writeMultiKernel<DISCR,WRITE_Y,LEN><<<grid, block, SH_MEM_SIZE>>>
        //     (d_tr_in2, cond_res, inds_res+D_HEIGHT,
        //      d_out, D_HEIGHT, D_WIDTH, num_elems, SEQ_CHUNK);
        writeMultiKernel<DISCR,WRITE_Y,LEN><<<grid, block, SH_MEM_SIZE>>>
            (d_tr_in2, cond_res, inds_res+D_HEIGHT,
             d_tr_out, D_HEIGHT, D_WIDTH, num_elems, SEQ_CHUNK);
    }
    
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
    gettimeofday(&t_endb, NULL);
    
    {
        // 4b. Transpose back.
        // transposePad<typename DISCR::InType,16>
        //     (d_tr_out, d_out, LEN, num_elems, LEN * num_elems, DISCR::padelm);
        transposeSpecialReverse<typename DISCR::InType,LEN>
            (d_tr_out, d_out, num_elems);
    }
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
    gettimeofday(&t_end, NULL);
    
    timeval_subtract(&t_diff, &t_end, &t_start);
    elapsed = (t_diff.tv_sec*1e6+t_diff.tv_usec);
    printf("%d\n", elapsed);
    fprintf(stderr, "mdiscr total runtime:             %6lu microsecs, from which:\n", elapsed);
    
    timeval_subtract(&t_diff, &t_med0a, &t_start);
    elapsed = (t_diff.tv_sec*1e6+t_diff.tv_usec);
    fprintf(stderr, "\t%-25s %6lu microsecs\n", "Transposition (a):", elapsed);
    
    timeval_subtract(&t_diff, &t_med0, &t_med0a);
    elapsed = (t_diff.tv_sec*1e6+t_diff.tv_usec);
    fprintf(stderr, "\t%-25s %6lu microsecs\n", "Transposition:", elapsed);
    
    timeval_subtract(&t_diff, &t_med1, &t_med0);
    elapsed = (t_diff.tv_sec*1e6+t_diff.tv_usec);
    printf("%d\n", elapsed);
    fprintf(stderr, "\t%-25s %6lu microsecs\n", "Map Cond Kernel:", elapsed);
    
    timeval_subtract(&t_diff, &t_med2, &t_med1);
    elapsed = (t_diff.tv_sec*1e6+t_diff.tv_usec);
    fprintf(stderr, "\t%-25s %6lu microsecs\n", "Scan Addition Kernel:", elapsed);
    
    timeval_subtract(&t_diff, &t_endb, &t_med2);
    elapsed = (t_diff.tv_sec*1e6+t_diff.tv_usec);
    printf("%d\n", elapsed);
    fprintf(stderr, "\t%-25s %6lu microsecs\n", "Global Write Kernel:", elapsed);
    
    timeval_subtract(&t_diff, &t_end, &t_endb);
    elapsed = (t_diff.tv_sec*1e6+t_diff.tv_usec);
    printf("%d\n", elapsed);
    fprintf(stderr, "\t%-25s %6lu microsecs\n", "Transposition (b):", elapsed);
    
    // free resources
    gpuErrchk( cudaFree(inds_res) );
    gpuErrchk( cudaFree(cond_res) );
    gpuErrchk( cudaFree(d_tr_in2) );
    gpuErrchk( cudaFree(d_tr_in1) );
    
    return filt_size;
    
}

#endif //MDISCR_HOST_ARRAYS
