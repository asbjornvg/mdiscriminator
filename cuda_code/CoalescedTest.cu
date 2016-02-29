#include <stdio.h>

#include "MdiscrKernels.cu.h"

#include <sys/time.h>
#include <time.h>

#define NUM 4

int nextMultOf(unsigned int x, unsigned int m) {
    if( x % m ) return x - (x % m) + m;
    else        return x;
}

int timeval_subtract(struct timeval *result, struct timeval *t2, struct timeval *t1)
{
    unsigned int resolution=1000000;
    long int diff = (t2->tv_usec + resolution * t2->tv_sec) - (t1->tv_usec + resolution * t1->tv_sec);
    result->tv_sec = diff / resolution;
    result->tv_usec = diff % resolution;
    return (diff<0);
}

__global__ void
coalescedKernel(  int*          d_in,
                  int*          d_out,
                  unsigned int  d_size
    ) {
    const unsigned int gid = blockIdx.x*blockDim.x + threadIdx.x;
    if(gid < d_size) {
        int k = d_in[gid];
        
        int tuple[NUM];
        for (int i = 0; i < NUM; i++) {
            tuple[i] = 0;
        }
        tuple[k] = 1;
        
        for (int i = 0; i < NUM; i++) {
            d_out[d_size*i+gid] = tuple[i];
        }
    }
}

__global__ void
nonCoalescedKernel(  int*          d_in,
                     int*          d_out,
                     unsigned int  d_size
    ) {
    const unsigned int gid = blockIdx.x*blockDim.x + threadIdx.x;
    if(gid < d_size) {
        int k = d_in[gid];
        
        int tuple[NUM];
        for (int i = 0; i < NUM; i++) {
            tuple[i] = 0;
        }
        tuple[k] = 1;
        
        for (int i = 0; i < NUM; i++) {
            d_out[NUM*gid+i] = tuple[i];
        }
    }
}

#define MAX_BLOCKS 65535

int main(int argc, char** argv) {
    if (argc != 2) {
        printf("The program takes <num_elems> as argument!\n");
        return EXIT_FAILURE;
    }
    
    struct timeval t_diff, t_start_coalesced, t_end_coalesced,
        t_start_non_coalesced, t_end_non_coalesced;
    unsigned long int elapsed_coalesced, elapsed_non_coalesced;
    
    const unsigned int num_elems = strtoul(argv[1], NULL, 10); //40000000; //50332001;
    unsigned int mem_size = num_elems * sizeof(int);
    
    // Allocate memory.
    int* h_in                  = (int*) malloc(mem_size);
    int* h_out_coalesced       = (int*) malloc(NUM*mem_size);
    int* h_out_non_coalesced   = (int*) malloc(NUM*mem_size);
    
    { // Initialize array.
        std::srand(33);
        for(unsigned int i = 0; i < num_elems; i++) {
            h_in[i] = std::rand() % NUM;
        }
    }
    
    int *d_in, *d_out_coalesced, *d_out_non_coalesced;
    { // Device allocation.
        cudaMalloc((void**)&d_in,                mem_size);
        cudaMalloc((void**)&d_out_coalesced,     NUM*mem_size);
        cudaMalloc((void**)&d_out_non_coalesced, NUM*mem_size);
        
        // Copy host memory to device.
        cudaMemcpy(d_in, h_in, mem_size, cudaMemcpyHostToDevice);
        cudaThreadSynchronize();
    }
    
    // Sizes for the kernels.
    unsigned int block_size, num_blocks;
    
    // Compute the sizes for the kernels.
    block_size = nextMultOf( (num_elems + MAX_BLOCKS - 1) / MAX_BLOCKS, 32 );
    block_size = (block_size < 256) ? 256 : block_size;
    num_blocks = (num_elems + block_size - 1) / block_size;
    
    // Call the kernels.
    gettimeofday(&t_start_coalesced, NULL);
    coalescedKernel<<<num_blocks, block_size>>>(d_in, d_out_coalesced, num_elems);
    cudaThreadSynchronize();
    gettimeofday(&t_end_coalesced, NULL);
    
    gettimeofday(&t_start_non_coalesced, NULL);
    nonCoalescedKernel<<<num_blocks, block_size>>>(d_in, d_out_non_coalesced, num_elems);
    cudaThreadSynchronize();
    gettimeofday(&t_end_non_coalesced, NULL);
    
    timeval_subtract(&t_diff, &t_end_coalesced, &t_start_coalesced);
    elapsed_coalesced = (t_diff.tv_sec*1e6+t_diff.tv_usec); 
    printf("Coalesced runtime: %lu microsecs\n", elapsed_coalesced);
    
    timeval_subtract(&t_diff, &t_end_non_coalesced, &t_start_non_coalesced);
    elapsed_non_coalesced = (t_diff.tv_sec*1e6+t_diff.tv_usec); 
    printf("Non-coalesced runtime: %lu microsecs\n", elapsed_non_coalesced);
    
    printf("Percent improvement: %.2f%%\n", (1.0-(float)elapsed_coalesced/elapsed_non_coalesced)*100.0);
    
    // Copy result back to host.
    cudaMemcpy(h_out_coalesced, d_out_coalesced, NUM*mem_size, cudaMemcpyDeviceToHost);
    cudaThreadSynchronize();
    cudaMemcpy(h_out_non_coalesced, d_out_non_coalesced, NUM*mem_size, cudaMemcpyDeviceToHost);
    cudaThreadSynchronize();
    
    // Validation.
    bool success = true;
    for (int gid = 0; gid < num_elems; gid++) {
        int k = h_in[gid];
        for (int i = 0; i < NUM; i++) {
            int element_coalesced = h_out_coalesced[num_elems*i+gid];
            int element_non_coalesced = h_out_non_coalesced[NUM*gid+i];
            if (i == k) {
                if (element_coalesced != 1) {
                    printf("Coalesced violation: %d should be 1, gid: %d, i: %d\n", element_coalesced, gid, i);
                    success = false;
                }
                if (element_non_coalesced != 1) {
                    printf("Non-coalesced violation: %d should be 1, gid: %d, i: %d\n", element_non_coalesced, gid, i);
                    success = false;
                }
            }
            else {
                if (element_coalesced != 0) {
                    printf("Coalesced violation: %d should be 0, gid: %d, i: %d\n", element_coalesced, gid, i);
                    success = false;
                }
                if (element_non_coalesced != 0) {
                    printf("Non-coalesced violation: %d should be 0, gid: %d, i: %d\n", element_non_coalesced, gid, i);
                    success = false;
                }
            }
        }
    }
    if (success) {
        printf("CoalescedTest on %d elems: VALID RESULT!\n", num_elems);
    }
    else {
        printf("CoalescedTest on %d elems: INVALID RESULT!\n", num_elems);
    }
    
    // Free device memory.
    cudaFree(d_in );
    cudaFree(d_out_coalesced);
    cudaFree(d_out_non_coalesced);
    
    // Cleanup memory.
    free(h_in );
    free(h_out_coalesced);
    free(h_out_non_coalesced);

    return 0;
    
}
