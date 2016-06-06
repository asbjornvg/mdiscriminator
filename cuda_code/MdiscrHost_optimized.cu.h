#ifndef MDISCR_HOST_OPTIMIZED
#define MDISCR_HOST_OPTIMIZED

#include "MdiscrKernels_optimized.cu.h"
#include "HelpersHost.cu.h"

#include<iostream>

template<class DISCR>
typename DISCR::TupleType
mdiscr(const unsigned int       num_elems,
       const unsigned int       num_hwd_thds,
       typename DISCR::InType*  d_in,  // device
       typename DISCR::InType*  d_out  // device
) {
    //const unsigned int MAX_CHUNK = 384; //224; //128; //96; //256; //384;
    /* You need to define through preprocessor options on command-line, e.g.,
     * -DMAX_CHUNK=384.
     */
    
    const unsigned int WIDTH_MULT = lcm(MAP_Y, WRITE_Y);
    const unsigned int HEIGHT_MULT = lcm(MAP_X, WRITE_X);
    fprintf(stderr, "WIDTH_MULT = %d\n", WIDTH_MULT);
    fprintf(stderr, "HEIGHT_MULT = %d\n", HEIGHT_MULT);
    
    // const unsigned int D_WIDTH = min( nextMultOf(max(num_elems/num_hwd_thds,1), 32), MAX_CHUNK);
    // const unsigned int D_WIDTH = min(nextMultOf(max(num_elems/num_hwd_thds,1),
    //                                             WIDTH_MULT),
    //                                  MAX_CHUNK);
    const unsigned int D_WIDTH = nextMultOf(min(max(num_elems/num_hwd_thds,1),
                                                MAX_CHUNK),
                                            WIDTH_MULT);
    
    // const unsigned int D_HEIGHT = nextMultOf( (num_elems + D_WIDTH - 1) / D_WIDTH, 64 );
    // const unsigned int D_HEIGHT = nextMultOf( (num_elems + D_WIDTH - 1) / D_WIDTH, MAP_X );
    const unsigned int D_HEIGHT = nextMultOf((num_elems + D_WIDTH - 1) / D_WIDTH,
                                             HEIGHT_MULT);
    
    // const unsigned int PADD = nextMultOf(D_HEIGHT*D_WIDTH, 64*D_WIDTH) - num_elems;
    const unsigned int PADD = nextMultOf(D_HEIGHT*D_WIDTH, HEIGHT_MULT*D_WIDTH) - num_elems;
    
    fprintf(stderr, "D_HEIGHT = %d, D_WIDTH = %d, PADD = %d\n", D_HEIGHT, D_WIDTH, PADD);
    
    struct timeval t_start, t_med0, t_med1, t_med2, t_end, t_diff;
    unsigned long int elapsed;
    
    typename DISCR::InType *d_tr_in;
    unsigned int *cond_res;
    typename DISCR::TupleType *inds_res;
    typename DISCR::TupleType  filt_size;
    
    gpuErrchk( cudaMalloc((void**)&d_tr_in, D_HEIGHT*D_WIDTH*sizeof(typename DISCR::InType)) );
    gpuErrchk( cudaMalloc((void**)&cond_res, D_HEIGHT*D_WIDTH*sizeof(unsigned int)) );
    gpuErrchk( cudaMalloc((void**)&inds_res, 2*D_HEIGHT*sizeof(typename DISCR::TupleType)) );
    
    gettimeofday(&t_start, NULL);
    { // 1. Transpose with padding!
        transposePad<typename DISCR::InType,32>
            (d_in, d_tr_in, D_HEIGHT, D_WIDTH, num_elems, DISCR::padelm);
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
        fprintf(stderr, "num_blocks = %d\n", num_blocks);
        
        dim3 block(MAP_X, MAP_Y, 1);
        dim3 grid(num_blocks, 1, 1);
        
        // map the condition
        /* mapVctKernel<DISCR><<<num_blocks, block_size, SH_MEM_MAP>>> */
        /*     (d_tr_in, cond_res, inds_res, D_HEIGHT, D_WIDTH); */
        mapVctKernel2<DISCR,MAP_Y><<<grid, block, SH_MEM_MAP>>>
            (d_tr_in, cond_res, inds_res, D_HEIGHT, SEQ_CHUNK);
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
        filt_size[DISCR::apply(DISCR::padelm)] -= PADD;
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
            std::max(WRITE_Y * WRITE_X * SEQ_CHUNK * sizeof(typename DISCR::InType),
                     (WRITE_Y + 1) * WRITE_X * sizeof(typename DISCR::TupleType));
        /* const unsigned int SH_MEM_SIZE = */
        /*     WRITE_Y * WRITE_X * SEQ_CHUNK * sizeof(typename DISCR::InType) + */
        /*     (WRITE_Y + 1) * WRITE_X * sizeof(typename DISCR::TupleType); */
        
        //printf("WRITE_Y = %d, WRITE_X = %d, SEQ_CHUNK = %d, sizeof(unsigned int) = %d, sizeof(typename DISCR::TupleType) = %d\n", WRITE_Y, WRITE_X, SEQ_CHUNK, sizeof(unsigned int), sizeof(typename DISCR::TupleType));
        fprintf(stderr, "SH_MEM_SIZE = %d\n", SH_MEM_SIZE);
        
        dim3 block(WRITE_X, WRITE_Y, 1);
        dim3 grid(num_blocks, 1, 1);
        /* if (WRITE_Y > 1) { */
        writeMultiKernel<DISCR,WRITE_Y><<<grid, block, SH_MEM_SIZE>>>
            (d_tr_in, cond_res, inds_res+D_HEIGHT, d_out, D_HEIGHT, num_elems, SEQ_CHUNK);
        /* } */
        /* else { */
        /* // Columns completely sequential. */
        /* writeMultiKernel2<DISCR><<<grid, block, SH_MEM_SIZE>>> */
        /* (d_tr_in, cond_res, inds_res+D_HEIGHT, d_out, D_HEIGHT, num_elems, SEQ_CHUNK); */
        /* } */
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
    cudaFree(d_tr_in );
    
    return filt_size;
    
}

#endif //MDISCR_HOST_OPTIMIZED
