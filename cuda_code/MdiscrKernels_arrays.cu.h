#ifndef MDISCR_KERNELS_ARRAYS
#define MDISCR_KERNELS_ARRAYS

template <class T, int TILE, unsigned int LEN>
__global__ void matTransposeTiledPadKerArray(T* A, T* B, int heightA, int widthA, int orig_size, T padel) {
    
    __shared__ T tile[LEN][TILE][TILE+1];
    
    unsigned int block_rows = blockDim.y;
    unsigned int j, ind, yj, d;
    
    unsigned int x = blockIdx.x * TILE + threadIdx.x;
    unsigned int y = blockIdx.y * TILE + threadIdx.y;
    
    for (j = 0; j < TILE; j += block_rows) {
        yj = y + j;
        ind = yj * widthA + x;
        if( x < widthA && yj < heightA ) {
            for (d = 0; d < LEN; d++) {
                tile[d][threadIdx.y + j][threadIdx.x] = (ind < orig_size) ? A[ind + d*orig_size] : padel;
            }
        }
    }
    
    __syncthreads();
    
    x = blockIdx.y * TILE + threadIdx.x;
    y = blockIdx.x * TILE + threadIdx.y;
    
    for (j = 0; j < TILE; j += block_rows) {
        yj = y + j;
        ind = yj * heightA + x;
        if( x < heightA && yj < widthA ) {
            for (d = 0; d < LEN; d++) {
                B[ind + d*heightA*widthA] = tile[d][threadIdx.x][threadIdx.y + j];
            }
        }
    }
}

template <class T, unsigned int LEN_X, unsigned LEN_Y, bool WRITE_TO_SH_MEM>
__device__ inline void transposeSpecialHelper1(T* gl_mem, T* sh_mem, int height) {
    
    unsigned int block_rows = blockDim.x / LEN_X; // Number of threads on the y-axis
    unsigned int block, x, y, j, ind;
    
    block = blockIdx.x * LEN_X * LEN_Y; // Index to the start of the block
    x = threadIdx.x % LEN_X;
    y = threadIdx.x / LEN_X;
    
    for (j = 0; j < LEN_Y; j += block_rows) {
        ind = block + (y+j) * LEN_X + x;
        if (ind < height * LEN_X) {
            if (WRITE_TO_SH_MEM) {
                sh_mem[x * LEN_Y + y + j] = gl_mem[ind];
            }
            else {
                gl_mem[ind] = sh_mem[x * LEN_Y + y + j];
            }
        }
    }
}

template <class T, unsigned int LEN_X, unsigned LEN_Y, bool WRITE_TO_SH_MEM>
__device__ inline void transposeSpecialHelper2(T* gl_mem, T* sh_mem, int height) {
    
    unsigned int block_rows = blockDim.x / LEN_X; // Number of threads on the y-axis
    unsigned int block, x, y, j, ind;
    
    block = blockIdx.x * LEN_Y;
    x = threadIdx.x % block_rows;
    y = threadIdx.x / block_rows;
    
    for (j = 0; j < LEN_Y; j += block_rows) {
        ind = block + y * height + x + j;
        if(ind < height * LEN_X) {
            if (WRITE_TO_SH_MEM) {
                sh_mem[y * LEN_Y + x + j] = gl_mem[ind];
            }
            else {
                gl_mem[ind] = sh_mem[y * LEN_Y + x + j];
            }
        }
    }
}

template <class T, unsigned int LEN_X, unsigned LEN_Y>
__global__ void matTransposeSpecial(T* A, T* B, int height) {
    
    __shared__ T sh_mem[LEN_X * LEN_Y];
    
    transposeSpecialHelper1<T, LEN_X, LEN_Y, true>(A, sh_mem, height);
    __syncthreads();
    transposeSpecialHelper2<T, LEN_X, LEN_Y, false>(B, sh_mem, height);
}

template <class T, unsigned int LEN_X, unsigned LEN_Y>
__global__ void matTransposeSpecialReverse(T* A, T* B, int height) {
    
    __shared__ T sh_mem[LEN_X * LEN_Y];
    
    transposeSpecialHelper2<T, LEN_X, LEN_Y, true>(A, sh_mem, height);
    __syncthreads();
    transposeSpecialHelper1<T, LEN_X, LEN_Y, false>(B, sh_mem, height);
}

template<class> class Add;
template<class, class T, unsigned int> __device__
T scanIncGenPad(volatile T*, const unsigned int, const unsigned int);
template<class, class T, unsigned int> __device__
T scanIncGen(volatile T*, const unsigned int);

template<class MapLambda, unsigned int BLOCK_Y, unsigned int LEN>
__global__ void
mapVctKernel(   typename MapLambda::InType*     d_in,
                unsigned int*                   d_out,
                typename MapLambda::TupleType*  d_out_chunk,
                const unsigned int              d_height,
                const unsigned int              d_width,
                const unsigned int              CHUNK
    ) {
    unsigned int gidx = blockIdx.x*blockDim.x + threadIdx.x;
    
    if (gidx < d_height) {
        
#if MAX_CHUNK < 256
        typedef typename MapLambda::TupleType::SmallType M;
#else
        typedef typename MapLambda::TupleType::MediumType M;
#endif
        
        // Shared memory bytes.
        extern __shared__ char map_sh_mem[];
        
        // We are traversing the CHUNK vertically, so transpose the acc's in
        // shared memory.
        
        // Definition of acc.
        volatile M* ind_sh_mem = (volatile M*) map_sh_mem;
        volatile M* acc = ind_sh_mem + threadIdx.x*blockDim.y + threadIdx.y;
        
        // Local variables.
        unsigned int k, d;
        unsigned int tmp_id = gidx + threadIdx.y*CHUNK*d_height;
        
        // Initialize acc.
        (*acc) = M();
        
        typename MapLambda::InType arr[LEN];
        
        // Traverse the CHUNK vertically.
        for (k = 0; k < CHUNK; k++, tmp_id += d_height) {
            
            for (d = 0; d < LEN; d++) {
                arr[d] = d_in[tmp_id + d*d_height*d_width];
            }
            
            unsigned int res = MapLambda::apply(arr);
            
            d_out[tmp_id] = res;
            
            acc->increment(res);
        }
        
        // If there were several chunks in each row, scan them.
        if (BLOCK_Y > 1) {
            __syncthreads();
            
            // Scan the current row horizontally (because we have transposed the acc's).
            scanIncGen<Add<M>,M,BLOCK_Y>(ind_sh_mem,
                                         threadIdx.y*blockDim.x + threadIdx.x);
            __syncthreads();
        }
        
        // The last thread on the y-axis stores the reduction in d_out_chunk.
        if (threadIdx.y == (BLOCK_Y-1)) {
            d_out_chunk[gidx] = (typename MapLambda::TupleType)(*acc);
        }
    }
}

__device__ inline
unsigned int myHash(unsigned int ind) {
    return ind;
}

/**
 * The use of this kernel should guarantee that the blocks are full;
 * this is achieved by padding in the host. However the result array
 * is not considered padded (parameter), hence orig_size is tested
 * in the last stage (the update to global from shared memory) so
 * that we do not write out of bounds.
 * If WITH_ARRAY is defined, then accumulator's representation is
 *    an int[2/4/8] local array, hence in L1 => requires replay instrs
 *    OTHERWISE: an MyInt2/4/8 => hold in registers, but leads to divergence.
 * MyInt4 representation seems to be a bit better than int[4]. 
 */
template<class OP, unsigned int BLOCK_Y, unsigned int LEN>
__global__ void
writeMultiKernel(   typename OP::InType*     d_in,
                    unsigned int*            cond_res,
                    typename OP::TupleType*  perm_chunk,
                    typename OP::InType*     d_out,
                    const unsigned int       d_height,
                    const unsigned int       d_width,
                    const unsigned int       orig_size,
                    const unsigned int       CHUNK
) {
    unsigned int gidx = blockIdx.x*blockDim.x + threadIdx.x;
    
    if (gidx < d_height) {
        
        typedef typename OP::InType                T;
        typedef typename OP::TupleType             M;
        typedef typename OP::TupleType::MediumType CompactM;
        
        extern __shared__ char gen_sh_mem[];
        
        volatile CompactM* ind_sh_mem = (volatile CompactM*) gen_sh_mem;
        volatile T* elm_sh_mem = (volatile T*) gen_sh_mem;
        unsigned int k, idx, padded_idx, d;
        
        CompactM acc0;
        
        idx = threadIdx.x*blockDim.y+threadIdx.y;
        padded_idx = idx + idx/BLOCK_Y;
        
        // 1. vertically scan sequentially CHUNK elements, result in acc0
        unsigned int tmp_id = blockIdx.x*blockDim.x + threadIdx.y*CHUNK*d_height + threadIdx.x;
        
        for(k = 0; k < CHUNK; k++, tmp_id+=d_height) {
            acc0.increment(cond_res[tmp_id]);
        }
        
        ind_sh_mem[padded_idx] = acc0;
        __syncthreads();
        
        // 2. vertical warp-scan of the results from the seq scan step, put result back in acc0
        idx = threadIdx.y*blockDim.x+threadIdx.x;
        padded_idx = idx + idx/BLOCK_Y;
        scanIncGenPad<Add<CompactM>,CompactM,BLOCK_Y>(ind_sh_mem, idx, padded_idx);
        __syncthreads();
        
        if (threadIdx.y > 0) {
            idx = threadIdx.x*blockDim.y+threadIdx.y-1;
            padded_idx = idx + idx/BLOCK_Y;
            acc0 = ind_sh_mem[padded_idx];
        } else {
            acc0 = CompactM();
        }
        
        unsigned int arrtmp[M::cardinal];
        // 3. adjust acc0 to reflect block-level indices of the multi-index.
        tmp_id = blockIdx.x*blockDim.x;
        acc0.adjustRowToBlock(arrtmp, perm_chunk + tmp_id - 1,
                              perm_chunk + tmp_id + threadIdx.x - 1,
                              perm_chunk + min(tmp_id+blockDim.x,d_height) - 1 );
        
        // 4. performs an input-array traversal in which the elements are
        //    recorded in shared mem (using the block-adjusted indices of acc0)
        tmp_id = blockIdx.x*blockDim.x + threadIdx.y*CHUNK*d_height + threadIdx.x;
        for(unsigned int k = 0; k < CHUNK; k++, tmp_id+=d_height) {
            unsigned int iind = cond_res[tmp_id];
            unsigned int shind = acc0.increment(iind);
            for (d = 0; d < LEN; d++) {
                elm_sh_mem[myHash(shind) + d*blockDim.x*blockDim.y*CHUNK] =
                    d_in[tmp_id + d*d_height*d_width];
            }
        }
        __syncthreads();
        
        // 6. Finally, the shared memory is traverse in order and
        //    and the filtered array is written to global memory;
        //    Since all the elements of an equiv-class are contiguous
        //    in shared memory (and global memory), the uncoalesced 
        //    writes are minimized. 
        unsigned int* blk_vlst= ((unsigned int*)(perm_chunk + d_height - 1)); // very last (row) scan result
        k   = threadIdx.y*blockDim.x + threadIdx.x;
        unsigned int total_len = blockDim.x*blockDim.y*CHUNK;
        for( ; k < total_len; k+=blockDim.x*blockDim.y) {
            unsigned int glb_ind = 0, loc_ind = k;
            tmp_id = 0;
            while( loc_ind >= arrtmp[tmp_id] && tmp_id < M::cardinal) {
                glb_ind += blk_vlst[tmp_id];
                loc_ind -= arrtmp[tmp_id];
                tmp_id++;
            }
            
            tmp_id = glb_ind + loc_ind + (blockIdx.x > 0) * 
                ((unsigned int*) (perm_chunk + blockIdx.x*blockDim.x - 1))[tmp_id]; // blk_beg;
            if(tmp_id < orig_size) {
                for (d = 0; d < LEN; d++) {
                    d_out[tmp_id + d*orig_size] =
                        elm_sh_mem[myHash(k) + d*blockDim.x*blockDim.y*CHUNK];
                }
            }
        }
    }
}

#endif //MDISCR_KERNELS_ARRAYS
