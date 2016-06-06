#ifndef HELPERS_HOST
#define HELPERS_HOST

#include "HelpersCommon.h"
#include "HelpersKernels.cu.h"

#include<stdio.h>
#include<string>
#include<cassert>

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
       fprintf(stderr,"GPUassert (%d): %s (%s, line %d)\n", code, cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

/**
 * d_in       is the device matrix; it is supposably
 *                allocated and holds valid values (input).
 *                semantically of size [height x width]
 * d_out      is the output GPU array -- if you want
 *            its data on CPU needs to copy it back to host.
 *                semantically of size [width x height]
 * height     is the height of the input matrix
 * width      is the width  of the input matrix
 */
template<class T, int TILE>
void transposePad( T*                  inp_d,
                   T*                  out_d,
                   const unsigned int  height,
                   const unsigned int  width,
                   const unsigned int  oinp_size,
                   T                   pad_elem
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
    
    int dimx = ceil( ((float) width)/TILE );
    int dimy = ceil( ((float)height)/TILE );
    
    dim3 grid (dimx, dimy, 1);
    
    fprintf(stderr, "TILE = %d, TILE/elements_pr_thread = %d, dimy = %d, dimx = %d\n", TILE, TILE/elements_pr_thread, dimy, dimx);
    
    // 2. execute the kernel
    matTransposeTiledPadKer<T,TILE> <<< grid, block >>>
        (inp_d, out_d, height, width, oinp_size, pad_elem);
}

/**
 * block_size is the size of the cuda block (must be a multiple 
 *                of 32 less than 1025)
 * d_size     is the size of both the input and output arrays.
 * d_in       is the device array; it is supposably
 *                allocated and holds valid values (input).
 * d_out      is the output GPU array -- if you want 
 *            its data on CPU needs to copy it back to host.
 *
 * OP         class denotes the associative binary operator 
 *                and should have an implementation similar to 
 *                `class Add' in ScanUtil.cu, i.e., exporting
 *                `identity' and `apply' functions.
 * T          denotes the type on which OP operates, 
 *                e.g., float or int. 
 */
template<class OP, class T>
void scanInc(    unsigned int  block_size,
                 unsigned long d_size, 
                 T*            d_in,  // device
                 T*            d_out  // device
) {
    unsigned int num_blocks;
    unsigned int sh_mem_size = block_size * sizeof(T);

    num_blocks = ( (d_size % block_size) == 0) ?
                    d_size / block_size     :
                    d_size / block_size + 1 ;

    scanIncKernel<OP,T><<< num_blocks, block_size, sh_mem_size >>>(d_in, d_out, d_size);
    /* cudaThreadSynchronize(); */
    
    if (block_size >= d_size) { return; }

    /**********************/
    /*** Recursive Case ***/
    /**********************/

    //   1. allocate new device input & output array of size num_blocks
    T *d_rec_in, *d_rec_out;
    cudaMalloc((void**)&d_rec_in , num_blocks*sizeof(T));
    cudaMalloc((void**)&d_rec_out, num_blocks*sizeof(T));

    unsigned int num_blocks_rec = ( (num_blocks % block_size) == 0 ) ?
                                  num_blocks / block_size     :
                                  num_blocks / block_size + 1 ; 

    //   2. copy in the end-of-block results of the previous scan 
    copyEndOfBlockKernel<T><<< num_blocks_rec, block_size >>>(d_out, d_rec_in, num_blocks, d_size);
    /* cudaThreadSynchronize(); */

    //   3. scan recursively the last elements of each CUDA block
    scanInc<OP,T>( block_size, num_blocks, d_rec_in, d_rec_out );

    //   4. distribute the the corresponding element of the 
    //      recursively scanned data to all elements of the
    //      corresponding original block
    distributeEndBlock<OP,T><<< num_blocks, block_size >>>(d_rec_out, d_out, d_size);
    /* cudaThreadSynchronize(); */

    //   5. clean up
    cudaFree(d_rec_in );
    cudaFree(d_rec_out);
}

/**
 * block_size is the size of the cuda block (must be a multiple 
 *                of 32 less than 1025)
 * d_size     is the size of both the input and output arrays.
 * d_in       is the device array; it is supposably
 *                allocated and holds valid values (input).
 * flags      is the flag array, in which !=0 indicates 
 *                start of a segment.
 * d_out      is the output GPU array -- if you want 
 *            its data on CPU you need to copy it back to host.
 *
 * OP         class denotes the associative binary operator 
 *                and should have an implementation similar to 
 *                `class Add' in ScanUtil.cu, i.e., exporting
 *                `identity' and `apply' functions.
 * T          denotes the type on which OP operates, 
 *                e.g., float or int. 
 */
template<class OP, class T>
void sgmScanInc( const unsigned int  block_size,
                 const unsigned long d_size,
                 T*            d_in,  //device
                 int*          flags, //device
                 T*            d_out  //device
) {
    unsigned int num_blocks;
    unsigned int val_sh_size = block_size * sizeof(T  );
    unsigned int flg_sh_size = block_size * sizeof(int);
    
    num_blocks = ( (d_size % block_size) == 0) ?
                    d_size / block_size     :
                    d_size / block_size + 1 ;
    
    T     *d_rec_in;
    int   *f_rec_in;
    cudaMalloc((void**)&d_rec_in, num_blocks*sizeof(T  ));
    cudaMalloc((void**)&f_rec_in, num_blocks*sizeof(int));

    sgmScanIncKernel<OP,T> <<< num_blocks, block_size, val_sh_size+flg_sh_size >>>
                    (d_in, flags, d_out, f_rec_in, d_rec_in, d_size);
    cudaThreadSynchronize();
    //cudaError_t err = cudaThreadSynchronize();
    //if( err != cudaSuccess)
    //    printf("cudaThreadSynchronize error: %s\n", cudaGetErrorString(err));

    if (block_size >= d_size) { cudaFree(d_rec_in); cudaFree(f_rec_in); return; }

    //   1. allocate new device input & output array of size num_blocks
    T   *d_rec_out;
    int *f_inds;
    cudaMalloc((void**)&d_rec_out, num_blocks*sizeof(T   ));
    cudaMalloc((void**)&f_inds,    d_size    *sizeof(int ));

    //   2. recursive segmented scan on the last elements of each CUDA block
    sgmScanInc<OP,T>
                ( block_size, num_blocks, d_rec_in, f_rec_in, d_rec_out );

    //   3. create an index array that is non-zero for all elements
    //      that correspond to an open segment that crosses two blocks,
    //      and different than zero otherwise. This is implemented
    //      as a CUDA-block level inclusive scan on the flag array,
    //      i.e., the segment that start the block has zero-flags,
    //      which will be preserved by the inclusive scan. 
    scanIncKernel<Add<int>,int> <<< num_blocks, block_size, flg_sh_size >>>
                ( flags, f_inds, d_size );

    //   4. finally, accumulate the recursive result of segmented scan
    //      to the elements from the first segment of each block (if 
    //      segment is open).
    sgmDistributeEndBlock <OP,T> <<< num_blocks, block_size >>>
                ( d_rec_out, d_out, f_inds, d_size );
    cudaThreadSynchronize();

    //   5. clean up
    cudaFree(d_rec_in );
    cudaFree(d_rec_out);
    cudaFree(f_rec_in );
    cudaFree(f_inds   );
}

template<class OP, class T>
void sgmScanExc( const unsigned int  block_size,
                 const unsigned long d_size,
                 T*            d_in,  //device
                 int*          flags, //device
                 T*            d_out  //device
    ) {
    
    unsigned int num_blocks = ( (d_size % block_size) == 0) ?
        d_size / block_size     :
        d_size / block_size + 1 ;
    
    T* d_intermediate;
    cudaMalloc((void**)&d_intermediate, d_size*sizeof(T));
    
    sgmScanInc<OP,T>(block_size, d_size, d_in, flags, d_intermediate);
    
    sgmShiftRightByOne<T><<<num_blocks, block_size>>>
        (d_intermediate, flags, d_out, OP::identity(), d_size);
    cudaThreadSynchronize();
    
    cudaFree(d_intermediate);
}

#endif //HELPERS_HOST
