#ifndef MDISCR_KERNELS_TUPLES
#define MDISCR_KERNELS_TUPLES

template <class T1, class T2, class T3, int TILE>
__global__ void matTransposeTiledPadKer(T1* A1, T2* A2, T3* A3, T1* B1, T2* B2, T3* B3,
                                        int heightA, int widthA, int orig_size,
                                        T1 padel1, T2 padel2, T3 padel3) {
    
    __shared__ T1 tile1[TILE][TILE+1];
    __shared__ T1 tile2[TILE][TILE+1];
    __shared__ T1 tile3[TILE][TILE+1];
    
    int block_rows = blockDim.y;
    int j, ind, yj;
    
    int x = blockIdx.x * TILE + threadIdx.x;
    int y = blockIdx.y * TILE + threadIdx.y;
    
    for (j = 0; j < TILE; j += block_rows) {
        yj = y + j;
        ind = yj * widthA + x;
        if( x < widthA && yj < heightA ) {
            tile1[threadIdx.y + j][threadIdx.x] = (ind < orig_size) ? A1[ind] : padel1;
            tile2[threadIdx.y + j][threadIdx.x] = (ind < orig_size) ? A2[ind] : padel2;
            tile3[threadIdx.y + j][threadIdx.x] = (ind < orig_size) ? A3[ind] : padel3;
        }
    }
    
    __syncthreads();
    
    x = blockIdx.y * TILE + threadIdx.x;
    y = blockIdx.x * TILE + threadIdx.y;
    
    for (j = 0; j < TILE; j += block_rows) {
        yj = y + j;
        ind = yj * heightA + x;
        if( x < heightA && yj < widthA ) {
            B1[ind] = tile1[threadIdx.x][threadIdx.y + j];
            B2[ind] = tile2[threadIdx.x][threadIdx.y + j];
            B3[ind] = tile3[threadIdx.x][threadIdx.y + j];
        }
    }
}

template<class> class Add;
template<class, class T, unsigned int> __device__
T scanIncGenPad(volatile T*, const unsigned int, const unsigned int);
template<class, class T, unsigned int> __device__
T scanIncGen(volatile T*, const unsigned int);

template<class MapLambda, unsigned int BLOCK_Y>
__global__ void
mapVctKernel(   unsigned int*                   d_in1,
                unsigned char*                  d_in2,
                bool*                           d_in3,
                unsigned int*                   d_out,
                typename MapLambda::TupleType*  d_out_chunk,
                const unsigned int              d_height,
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
        unsigned int k;
        unsigned int tmp_id = gidx + threadIdx.y*CHUNK*d_height;
        
        // Initialize acc.
        (*acc) = M();
        
        // Traverse the CHUNK vertically.
        for (k = 0; k < CHUNK; k++, tmp_id += d_height) {
            unsigned int res = MapLambda::apply(d_in1[tmp_id],
                                                d_in2[tmp_id],
                                                d_in3[tmp_id]);
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
template<class OP, unsigned int BLOCK_Y>
__global__ void
writeMultiKernel(   unsigned int*            d_in1,
                    unsigned char*           d_in2,
                    bool*                    d_in3,
                    unsigned int*            cond_res,
                    typename OP::TupleType*  perm_chunk,
                    unsigned int*            d_out1,
                    unsigned char*           d_out2,
                    bool*                    d_out3,
                    const unsigned int       d_height,
                    const unsigned int       orig_size,
                    const unsigned int       CHUNK
) {
    unsigned int gidx = blockIdx.x*blockDim.x + threadIdx.x;
    
    if (gidx < d_height) {
        
        typedef unsigned int  T1;
        typedef unsigned char T2;
        typedef bool          T3;
        typedef typename OP::TupleType M;
        
        extern __shared__ char gen_sh_mem[];
        typedef typename OP::TupleType::MediumType CompactM;
        volatile CompactM* ind_sh_mem = (volatile CompactM*) gen_sh_mem;
        volatile T1* elm_sh_mem1 = (volatile T1*) gen_sh_mem;
        volatile T2* elm_sh_mem2 = (volatile T2*) &elm_sh_mem1[blockDim.x*blockDim.y*CHUNK];
        volatile T3* elm_sh_mem3 = (volatile T3*) &elm_sh_mem2[blockDim.x*blockDim.y*CHUNK];
        unsigned int k, idx, padded_idx;
        
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
            elm_sh_mem1[myHash(shind)] = d_in1[tmp_id];
            elm_sh_mem2[myHash(shind)] = d_in2[tmp_id];
            elm_sh_mem3[myHash(shind)] = d_in3[tmp_id];
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
                d_out1[tmp_id] = elm_sh_mem1[myHash(k)];
                d_out2[tmp_id] = elm_sh_mem2[myHash(k)];
                d_out3[tmp_id] = elm_sh_mem3[myHash(k)];
            }
        }
    }
}

#endif //MDISCR_KERNELS_TUPLES
