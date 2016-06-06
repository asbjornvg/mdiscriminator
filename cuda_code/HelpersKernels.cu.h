#ifndef HELPERS_KERNELS
#define HELPERS_KERNELS

#include <cuda_runtime.h>

/***************************************/
/*** Scan Inclusive Helpers & Kernel ***/
/***************************************/
template<class OP, class T>
__device__ inline
T scanIncWarp( volatile T* ptr, const unsigned int idx ) {
    const unsigned int lane = idx & 31;

    // no synchronization needed inside a WARP,
    //   i.e., SIMD execution
    if (lane >= 1)  ptr[idx] = OP::apply(ptr[idx-1],  ptr[idx]);
    if (lane >= 2)  ptr[idx] = OP::apply(ptr[idx-2],  ptr[idx]);
    if (lane >= 4)  ptr[idx] = OP::apply(ptr[idx-4],  ptr[idx]);
    if (lane >= 8)  ptr[idx] = OP::apply(ptr[idx-8],  ptr[idx]);
    if (lane >= 16) ptr[idx] = OP::apply(ptr[idx-16], ptr[idx]);
    return const_cast<T&>(ptr[idx]);
}

template<class OP, class T>
__device__ inline
T scanIncBlock(volatile T* ptr, const unsigned int idx) {
    const unsigned int lane   = idx &  31;
    const unsigned int warpid = idx >> 5;

    T val = scanIncWarp<OP,T>(ptr,idx);
    __syncthreads();

    // place the end-of-warp results in
    //   the first warp. This works because
    //   warp size = 32, and 
    //   max block size = 32^2 = 1024
    if (lane == 31) { ptr[warpid] = const_cast<T&>(ptr[idx]); }
    __syncthreads();

    //
    if (warpid == 0) scanIncWarp<OP,T>(ptr, idx);
    __syncthreads();

    if (warpid > 0) {
        val = OP::apply(ptr[warpid-1], val);
    }

    return val;
}

template<class OP, class T>
__global__ void 
scanIncKernel(T* d_in, T* d_out, unsigned int d_size) {
    extern __shared__ char sh_mem1[];
    volatile T* sh_memT = (volatile T*)sh_mem1;
    const unsigned int tid = threadIdx.x;
    const unsigned int gid = blockIdx.x*blockDim.x + tid;
    T el    = (gid < d_size) ? d_in[gid] : OP::identity();
    sh_memT[tid] = el;
    __syncthreads();
    T res   = scanIncBlock < OP, T >(sh_memT, tid);
    if (gid < d_size) d_out [gid] = res; 
}

/*
 * Generalization of scanIncGen (below) in which padding can occur. So orig_idx
 * is the index into the non-padded array and is only used to determine the
 * lane of the calling thread, whereas padd_idx is the actual index into the
 * array. The padding should only occur between groups of cooperating threads
 * and never inside a group.
 */
template<class OP, class T, unsigned int NUM>
__device__ inline
T scanIncGenPad(volatile T* ptr,
                const unsigned int orig_idx,
                const unsigned int padded_idx) {
    
    const unsigned int lane = orig_idx & (NUM-1);
    unsigned int lower_bound;

#pragma unroll
    for (lower_bound = 1; lower_bound < NUM; lower_bound *= 2) {
        if (lane >= lower_bound) {
            ptr[padded_idx] = OP::apply(ptr[padded_idx-lower_bound],  ptr[padded_idx]);
        }
    }
    
    return const_cast<T&>(ptr[padded_idx]);
}

/*
 * Generalization of scanIncWarp so that the number of cooperating threads
 * doesn't need to be 32 (warp) but can be any power of 2 less than or equal to
 * 32, given by the template parameter NUM.
 */
template<class OP, class T, unsigned int NUM>
__device__ inline
T scanIncGen(volatile T* ptr, const unsigned int idx) {
    return scanIncGenPad<OP,T,NUM>(ptr,idx,idx);
}


/***********************************************************/
/*** Kernels to copy/distribute the end of block results ***/
/***********************************************************/

template<class T>
__global__ void 
copyEndOfBlockKernel(T* d_in, T* d_out, unsigned int d_out_size, unsigned int d_orig_size) {
    const unsigned int gid = blockIdx.x*blockDim.x + threadIdx.x;
    
    if(gid < d_out_size) {
        // d_out[gid] = d_in[ blockDim.x*(gid+1) - 1];
        d_out[gid] = d_in[ min(blockDim.x*(gid+1), d_orig_size) - 1];
    }
}

template<class OP, class T>
__global__ void 
distributeEndBlock(T* d_in, T* d_out, unsigned int d_size) {
    const unsigned int gid = blockIdx.x*blockDim.x + threadIdx.x;
    
    if(gid < d_size && blockIdx.x > 0)
        d_out[gid] = OP::apply(d_out[gid],d_in[blockIdx.x-1]);
}

/*************************************************/
/*************************************************/
/*** Segmented Inclusive Scan Helpers & Kernel ***/
/*************************************************/
/*************************************************/
template<class OP, class T, class F>
__device__ inline
T sgmScanIncWarp(volatile T* ptr, volatile F* flg, const unsigned int idx) {
    const unsigned int lane = idx & 31;

    // no synchronization needed inside a WARP,
    //   i.e., SIMD execution
    if (lane >= 1)  {
        if(flg[idx] == 0) { ptr[idx] = OP::apply(ptr[idx-1], ptr[idx]); }
        flg[idx] = flg[idx-1] | flg[idx];
    }
    if (lane >= 2)  {
        if(flg[idx] == 0) { ptr[idx] = OP::apply(ptr[idx-2], ptr[idx]); }
        flg[idx] = flg[idx-2] | flg[idx];
    }
    if (lane >= 4)  {
        if(flg[idx] == 0) { ptr[idx] = OP::apply(ptr[idx-4], ptr[idx]); }
        flg[idx] = flg[idx-4] | flg[idx];
    }
    if (lane >= 8)  {
        if(flg[idx] == 0) { ptr[idx] = OP::apply(ptr[idx-8], ptr[idx]); }
        flg[idx] = flg[idx-8] | flg[idx];
    }
    if (lane >= 16)  {
        if(flg[idx] == 0) { ptr[idx] = OP::apply(ptr[idx-16], ptr[idx]); }
        flg[idx] = flg[idx-16] | flg[idx];
    }

    return const_cast<T&>(ptr[idx]);
}

template<class OP, class T, class F>
__device__ inline
T sgmScanIncBlock(volatile T* ptr, volatile F* flg, const unsigned int idx) {
    const unsigned int lane   = idx &  31;
    const unsigned int warpid = idx >> 5;
    const unsigned int warplst= (warpid<<5) + 31;

    // 1a: record whether this warp begins with an ``open'' segment.
    bool warp_is_open = (flg[(warpid << 5)] == 0);
    __syncthreads();

    // 1b: intra-warp segmented scan for each warp
    T val = sgmScanIncWarp<OP,T>(ptr,flg,idx);

    // 2a: the last value is the correct partial result
    T warp_total = const_cast<T&>(ptr[warplst]);
    
    // 2b: warp_flag is the OR-reduction of the flags 
    //     in a warp, and is computed indirectly from
    //     the mindex in hd[]
    bool warp_flag = flg[warplst]!=0 || !warp_is_open;
    bool will_accum= warp_is_open && (flg[idx] == 0);

    __syncthreads();

    // 2c: the last thread in a warp writes partial results
    //     in the first warp. Note that all fit in the first
    //     warp because warp = 32 and max block size is 32^2
    if (lane == 31) {
        ptr[warpid] = warp_total; //ptr[idx]; 
        flg[warpid] = warp_flag;
    }
    __syncthreads();

    // 
    if (warpid == 0) sgmScanIncWarp<OP,T>(ptr, flg, idx);
    __syncthreads();

    if (warpid > 0 && will_accum) {
        val = OP::apply(ptr[warpid-1], val);
    }
    return val;
}

template<class OP, class T>
__global__ void 
sgmScanIncKernel(T* d_in, int* flags, T* d_out, 
                          int* f_rec, T* d_rec, unsigned int d_size) {
    extern __shared__ char sh_mem[];
    volatile T*   vals_sh = (volatile T*)sh_mem;
    volatile int* flag_sh = (int*) (vals_sh + blockDim.x);
    const unsigned int tid = threadIdx.x;
    const unsigned int gid = blockIdx.x*blockDim.x + tid;
    int fl;   
    if (gid < d_size) { vals_sh[tid] = d_in[gid];      fl = flags[gid]; }
    else              { vals_sh[tid] = OP::identity(); fl = 0;          }
    flag_sh[tid] = fl;
    __syncthreads();
    T res = sgmScanIncBlock <OP, T>(vals_sh, flag_sh, tid);
    if (gid < d_size) d_out [gid] = res; 

    // set the flags and data for the recursive step!
    if(tid == 0)  { f_rec[blockIdx.x] = 0; }
    __syncthreads();
    if(fl  >  0)  { f_rec[blockIdx.x] = 1; }
    if(tid == (blockDim.x - 1)) { d_rec[blockIdx.x] = res; }
}

template<class OP, class T>
__global__ void 
sgmDistributeEndBlock(T* d_rec_in, T* d_out, int* f_inds, unsigned int d_size) {
    const unsigned int gid = blockIdx.x*blockDim.x + threadIdx.x;
    
    if(gid < d_size && blockIdx.x > 0) {
        if(f_inds[gid] == 0)
            d_out[gid] = OP::apply(d_out[gid], d_rec_in[blockIdx.x-1]);
    }
}

template<class T>
__global__ void
sgmShiftRightByOne(T*            d_in,
                   int*          flags,
                   T*            d_out,
                   T             ne,
                   unsigned int  d_size
    ) {
    const unsigned int gid = blockIdx.x*blockDim.x + threadIdx.x;
    if(gid < d_size) {
        if      (flags[gid]!=0)  d_out[gid] = ne;
        else                     d_out[gid] = d_in[gid-1];
    }
}

/************************/
/*** TRANSPOSE Kernel ***/
/************************/
// blockDim.y = TILE; blockDim.x = TILE
// each block transposes a square TILE
template <class T, int TILE>
__global__ void matTransposeTiledPadKer(T* A, T* B, int heightA, int widthA, int orig_size, T padel) {
    
    __shared__ T tile[TILE][TILE+1];
    
    int block_rows = blockDim.y;
    int x, y, j, ind, yj;
    
    x = blockIdx.x * TILE + threadIdx.x;
    y = blockIdx.y * TILE + threadIdx.y;
    
    for (j = 0; j < TILE; j += block_rows) {
        yj = y + j;
        ind = yj * widthA + x;
        if( x < widthA && yj < heightA ) {
            tile[threadIdx.y + j][threadIdx.x] = (ind < orig_size) ? A[ind] : padel;
        }
    }
    
    __syncthreads();
    
    x = blockIdx.y * TILE + threadIdx.x;
    y = blockIdx.x * TILE + threadIdx.y;
    
    for (j = 0; j < TILE; j += block_rows) {
        yj = y + j;
        ind = yj * heightA + x;
        if( x < heightA && yj < widthA ) {
            B[ind] = tile[threadIdx.x][threadIdx.y + j];
        }
    }
}

/* template <class T, int TILE> */
/* __global__ void matTransposeTiledPadKer(T* A, T* B, int heightA, int widthA, int orig_size, T padel) { */

/*   __shared__ T tile[TILE][TILE+1]; */

/*   int x = blockIdx.x * TILE + threadIdx.x; */
/*   int y = blockIdx.y * TILE + threadIdx.y; */

/*   int ind = y*widthA+x; */
/*   if( x < widthA && y < heightA ) */
/*       tile[threadIdx.y][threadIdx.x] = (ind < orig_size) ? A[ind] : padel; */

/*   __syncthreads(); */

/*   x = blockIdx.y * TILE + threadIdx.x; */
/*   y = blockIdx.x * TILE + threadIdx.y; */

/*   ind = y*heightA + x; */
/*   if( x < heightA && y < widthA ) */
/*       B[ind] = tile[threadIdx.x][threadIdx.y]; */
/* } */

#endif //HELPERS_KERNELS
