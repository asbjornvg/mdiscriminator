#ifndef MDISCR_KERNELS_OPTIMIZED
#define MDISCR_KERNELS_OPTIMIZED

template<class MapLambda>
__global__ void
mapVctKernel(   typename MapLambda::InType*     d_in,
                unsigned int*                   d_out,
                typename MapLambda::TupleType*  d_out_chunk,
                const unsigned int              d_height,
                const unsigned int              d_width
) {
    typedef typename MapLambda::InType T;
    typedef typename MapLambda::TupleType M;

    extern __shared__ char map_sh_mem[];
    volatile M* acc = ((volatile M*)map_sh_mem) + threadIdx.x;
    unsigned int gid = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int i;

    if(gid < d_height) {
        
        (*acc) = M(0);

        for (i = 0; i < d_width; i++, gid += d_height) {
            unsigned int res = MapLambda::apply(d_in[gid]);
            d_out[gid] = res;
            (*acc)[res]++;
        }
        gid = blockIdx.x*blockDim.x + threadIdx.x; // -= d_height * d_width;
        d_out_chunk[gid] = *acc;
    }
}

template<class> class Add;
template<class, class T, unsigned int> __device__
T scanIncGenPad(volatile T*, const unsigned int, const unsigned int);
template<class, class T, unsigned int> __device__
T scanIncGen(volatile T*, const unsigned int);

template<class MapLambda, unsigned int BLOCK_Y>
__global__ void
mapVctKernel2(   typename MapLambda::InType*     d_in,
                 unsigned int*                   d_out,
                 typename MapLambda::TupleType*  d_out_chunk,
                 const unsigned int              d_height,
                 const unsigned int              CHUNK
    ) {
    unsigned int gidx = blockIdx.x*blockDim.x + threadIdx.x;
    
    if (gidx < d_height) {
        
#ifdef COMPACT_REPRESENTATION_MAP
#if MAX_CHUNK < 256
        typedef typename MapLambda::TupleType::SmallType M;
#else
        typedef typename MapLambda::TupleType::MediumType M;
#endif
#else
        typedef typename MapLambda::TupleType M;
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
            unsigned int res = MapLambda::apply(d_in[tmp_id]);
            d_out[tmp_id] = res;
            
#ifdef COMPACT_REPRESENTATION_MAP
            acc->increment(res);
#else
            (*acc)[res]++;
#endif
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
#ifdef COMPACT_REPRESENTATION_MAP
            d_out_chunk[gidx] = (typename MapLambda::TupleType)(*acc);
#else
            d_out_chunk[gidx] = (M)*acc;
#endif
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
writeMultiKernel(   typename OP:: InType* d_in,  unsigned int* cond_res, 
                    typename OP::TupleType* perm_chunk, 
                    typename OP:: InType* d_out, 
                    const unsigned int d_height, 
                    const unsigned int orig_size,
                    const unsigned int CHUNK
) {
    unsigned int gidx = blockIdx.x*blockDim.x + threadIdx.x;
    
    if (gidx < d_height) {
        
        typedef typename OP::InType    T;
        typedef typename OP::TupleType M;
        
        extern __shared__ char gen_sh_mem[];
#ifdef COMPACT_REPRESENTATION_WRITE
        typedef typename OP::TupleType::MediumType CompactM;
        volatile CompactM* ind_sh_mem = (volatile CompactM*) gen_sh_mem;
#else
        volatile M* ind_sh_mem = (volatile M*) gen_sh_mem;
#endif
        volatile T* elm_sh_mem = (volatile T*) gen_sh_mem;
        //volatile T* elm_sh_mem = (volatile T*)(ind_sh_mem + blockDim.x*(blockDim.y+1));
        unsigned int k, idx, padded_idx;
        
#ifdef COMPACT_REPRESENTATION_WRITE
        CompactM acc0;
#else
        M acc0;
#endif
        
        idx = threadIdx.x*blockDim.y+threadIdx.y;
        padded_idx = idx + idx/BLOCK_Y;
        // volatile M* acc1 = ind_sh_mem + padded_idx;
        
        // (*acc1) = M();
        
        // 1. vertically scan sequentially CHUNK elements, result in acc0
        unsigned int tmp_id = blockIdx.x*blockDim.x + threadIdx.y*CHUNK*d_height + threadIdx.x;
        
        for(k = 0; k < CHUNK; k++, tmp_id+=d_height) {
#ifdef COMPACT_REPRESENTATION_WRITE
            acc0.increment(cond_res[tmp_id]);
#else
            acc0[cond_res[tmp_id]]++;
#endif
            //(*acc1)[cond_res[tmp_id]]++;
        }
        
        ind_sh_mem[padded_idx] = acc0;
        __syncthreads();
        
        // 2. vertical warp-scan of the results from the seq scan step, put result back in acc0
        idx = threadIdx.y*blockDim.x+threadIdx.x;
        padded_idx = idx + idx/BLOCK_Y;
#ifdef COMPACT_REPRESENTATION_WRITE
        scanIncGenPad<Add<CompactM>,CompactM,BLOCK_Y>(ind_sh_mem, idx, padded_idx);
#else
        scanIncGenPad<Add<M>,M,BLOCK_Y>(ind_sh_mem, idx, padded_idx);
#endif
        __syncthreads();
        
        if (threadIdx.y > 0) {
            idx = threadIdx.x*blockDim.y+threadIdx.y-1;
            padded_idx = idx + idx/BLOCK_Y;
            acc0 = ind_sh_mem[padded_idx];
            
            // acc1 = ind_sh_mem + padded_idx;
        } else {
#ifdef COMPACT_REPRESENTATION_WRITE
            acc0 = CompactM();
#else
            acc0 = M();
#endif
            
            // idx = threadIdx.x*blockDim.y+blockDim.y-1;
            // padded_idx = idx + idx/BLOCK_Y;
            // acc1 = ind_sh_mem + padded_idx;
            // (*acc1) = M();
        }
        __syncthreads();
        // (*acc1) = acc0;
        // __syncthreads();
        
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
            
#ifdef COMPACT_REPRESENTATION_WRITE
            unsigned int shind = acc0.increment(iind);
#else
            unsigned int shind = acc0[iind]++;
#endif
            elm_sh_mem[myHash(shind)] = d_in[tmp_id];
            
            // unsigned int shind2 = (*acc1)[iind]++;
            // elm_sh_mem[myHash(shind2)] = d_in[tmp_id];
        }
        __syncthreads();
        
        // // Nonsense writes to global memory, so that everything doesn't get
        // // optimized away.
        // tmp_id = (blockIdx.x*blockDim.y*blockDim.x + threadIdx.x*blockDim.y + threadIdx.y)*CHUNK;
        // int tmp_id2 = (threadIdx.y*blockDim.x + threadIdx.x) * CHUNK;
        // for(unsigned int k = 0; k < CHUNK; k++, tmp_id++, tmp_id2++) {
        //     d_out[tmp_id] = elm_sh_mem[tmp_id2];
        // }
        // __syncthreads();
        
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
            // while( loc_ind >= arrtmp[tmp_id] && tmp_id < M::cardinal) {
            while(tmp_id < M::cardinal && loc_ind >= arrtmp[tmp_id]) {
                glb_ind += blk_vlst[tmp_id];
                loc_ind -= arrtmp[tmp_id];
                tmp_id++;
            }
            
            tmp_id = glb_ind + loc_ind + (blockIdx.x > 0) * 
                ((unsigned int*) (perm_chunk + blockIdx.x*blockDim.x - 1))[tmp_id]; // blk_beg;
            if(tmp_id < orig_size) 
                d_out[tmp_id] = elm_sh_mem[myHash(k)];
        }
    }
}

// // Completely sequential rows.
// template<class OP>
// __global__ void
// writeMultiKernel2(   typename OP:: InType* d_in,  unsigned int* cond_res, 
//                      typename OP::TupleType* perm_chunk, 
//                      typename OP:: InType* d_out, 
//                      const unsigned int d_height, 
//                      const unsigned int orig_size,
//                      const unsigned int CHUNK
// ) {
//     typedef typename OP:: InType T;
//     typedef typename OP::TupleType M;
    
//     extern __shared__ char gen_sh_mem[];
//     //volatile M* ind_sh_mem = (volatile M*) gen_sh_mem;
//     volatile T* elm_sh_mem = (volatile T*) gen_sh_mem;
//     unsigned int k;
    
//     M acc0;
    
//     unsigned int arrtmp[M::cardinal];
//     // 3. adjust acc0 to reflect block-level indices of the multi-index.
//     unsigned int tmp_id = blockIdx.x*blockDim.x;
//     OP::adjustRowToBlock(acc0, arrtmp, perm_chunk + tmp_id - 1, 
//                          perm_chunk + tmp_id + threadIdx.x - 1, 
//                          perm_chunk + min(tmp_id+blockDim.x,d_height) - 1 ); 
//     //__syncthreads();
    
//     // 4. performs an input-array traversal in which the elements are
//     //    recorded in shared mem (using the block-adjusted indices of acc0) 
//     tmp_id = blockIdx.x*blockDim.x + threadIdx.y*CHUNK*d_height + threadIdx.x;
//     for(unsigned int k = 0; k < CHUNK; k++, tmp_id+=d_height) {
//         unsigned int iind = cond_res[tmp_id];
//         unsigned int shind = acc0[iind]++;
//         elm_sh_mem[myHash(shind)] = d_in[tmp_id];
//     }
//     __syncthreads();
    
//     // 6. Finally, the shared memory is traverse in order and
//     //    and the filtered array is written to global memory;
//     //    Since all the elements of an equiv-class are contiguous
//     //    in shared memory (and global memory), the uncoalesced 
//     //    writes are minimized. 
//     unsigned int* blk_vlst= ((unsigned int*)(perm_chunk + d_height - 1)); // very last (row) scan result
//     k   = threadIdx.y*blockDim.x + threadIdx.x;
//     unsigned int total_len = blockDim.x*blockDim.y*CHUNK;
//     for( ; k < total_len; k+=blockDim.x*blockDim.y) {
//         unsigned int glb_ind = 0, loc_ind = k;
//         tmp_id = 0;
//         while( loc_ind >= arrtmp[tmp_id] && tmp_id < M::cardinal) {
//             glb_ind += blk_vlst[tmp_id];
//             loc_ind -= arrtmp[tmp_id];
//             tmp_id++;
//         }
        
//         tmp_id = glb_ind + loc_ind + (blockIdx.x > 0) * 
//             ((unsigned int*) (perm_chunk + blockIdx.x*blockDim.x - 1))[tmp_id]; // blk_beg;
//         if(tmp_id < orig_size) 
//             d_out[tmp_id] = elm_sh_mem[myHash(k)];
//     }
// }

#endif //MDISCR_KERNELS_OPTIMIZED
