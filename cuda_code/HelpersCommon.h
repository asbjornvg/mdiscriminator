#ifndef HELPERS_COMMON
#define HELPERS_COMMON

#include<string>
#include<cstdio>
#include <sys/time.h>
#include <time.h>
#include <cassert>

#include "SeqMdiscr.h"

int nextMultOf(unsigned int x, unsigned int m) {
    if( x % m ) return x - (x % m) + m;
    else        return x;
}

// The maximum number of blocks in one dimension.
#define MAX_BLOCKS 65535

unsigned int getBlockSize(unsigned int num_elems) {
    unsigned int block_size;
    
    // Divide the elements into MAX_BLOCKS blocks, but round up to nearest
    // multiple of 32.
    block_size = nextMultOf( (num_elems + MAX_BLOCKS - 1) / MAX_BLOCKS, 32 );
    
    // We don't want really small blocks, however.
    block_size = (block_size < 256) ? 256 : block_size;
    
    // Cannot be greater than 1024. If the number of elements exceeds this
    // constraint, we must go 2D.
    assert(block_size <= 1024);
    
    printf("block_size: %d\n", block_size);
    
    return block_size;
}

unsigned int getNumBlocks(unsigned int num_elems, unsigned int block_size) {
    unsigned int num_blocks;
    
    // With the given number of elements and block size, how many blocks do
    // we need?
    num_blocks = (num_elems + block_size - 1) / block_size;
    
    printf("num_blocks: %d\n", num_blocks);
    
    return num_blocks;
}

int timeval_subtract(struct timeval *result, struct timeval *t2, struct timeval *t1)
{
    unsigned int resolution=1000000;
    long int diff = (t2->tv_usec + resolution * t2->tv_sec) - (t1->tv_usec + resolution * t1->tv_sec);
    result->tv_sec = diff / resolution;
    result->tv_usec = diff % resolution;
    return (diff<0);
}

/*
 * Populate an array of integers. The random number generator should be seeded
 * beforehand (by calling srand).
 */
void populateIntArray(const unsigned int  num_elems,
                      int*                in_array
    ) {
    for(unsigned int i = 0; i < num_elems; i++) {
        in_array[i] = std::rand() % 20;
    }
}

int getNextSegmentSize() {
    return (std::rand() % 50) + 25;
    //return (std::rand() % 5) + 3;
}

/*
 * Populate an array of sizes (flags). The random number generator should be
 * seeded beforehand (by calling srand).
 */
void populateSizesArray(const unsigned int  num_elems,
                        int*                in_array_sizes
    ) {
    
    // Size of the first segment.
    int current_size = getNextSegmentSize();
    if (current_size > num_elems) {
        current_size = num_elems;
    }
        
    // How far into the current segment are we?
    unsigned int j = 0;
        
    for(unsigned int i = 0; i < num_elems; i++) {
            
        if (j == 0) {
            // If we are at the segment start, write the size.
            in_array_sizes[i] = current_size;
        }
        else {
            // Otherwise, write a zero.
            in_array_sizes[i] = 0;
        }
            
        if (j == (current_size-1)) {
            // If we are at the last element of a segment, pick at new
            // size for the next segment.
            current_size = getNextSegmentSize();
            if (current_size > (num_elems-(i+1))) {
                current_size = num_elems-(i+1);
            }
            j = 0;
        }
        else {
            j++;
        }
    }
}

void printIntArray(unsigned int length, std::string title, int *arr) {
    printf("%-12s [", (title + ":").c_str());
    bool first = true;
    for(unsigned int i = 0; i < length; i++) {
        if (first) {
            printf("%2d", arr[i]);
            first = false;
        }
        else {
            printf(", %2d", arr[i]);
        }
    }
    printf("]\n");
}

template<class ModN>
bool validateOneSegment(typename ModN::InType*  in_array,
                        typename ModN::InType*  out_array,
                        int*                    sizes_array,
                        unsigned int            num_elems
    ) {
    
    bool success = true;
    
    // Allocate memory for the comparison (sequential) run.
    typename ModN::InType* in_array_cmp =
        (typename ModN::InType*) malloc(num_elems * sizeof(typename ModN::InType));
    typename ModN::InType* out_array_cmp =
        (typename ModN::InType*) malloc(num_elems * sizeof(typename ModN::InType));
    int* sizes_array_cmp = (int*) malloc(num_elems * sizeof(int));
    
    // Copy the in-array.
    memcpy(in_array_cmp, in_array, num_elems*sizeof(typename ModN::InType));
    
    // Run the sequential version.
    seqMdiscr<ModN, false>(num_elems, in_array_cmp, out_array_cmp, sizes_array_cmp);
    
    typename ModN::InType out_element, out_element_cmp;
    int size, size_cmp;
    for(unsigned int i = 0; i < num_elems; i++) {
        
        // Compare elements.
        out_element = out_array[i];
        out_element_cmp = out_array_cmp[i];
        if (out_element != out_element_cmp) {
            success = false;
            printf("Violation: %d should be %d (i = %d)\n", out_element, out_element_cmp, i);
        }
        
        // Compare sizes.
        size = sizes_array[i];
        size_cmp = sizes_array_cmp[i];
        if (size != size_cmp) {
            success = false;
            printf("Invalid size: %d should be %d (i = %d)\n", size, size_cmp, i);
        }
        
        // Only print the first few violations.
        if (!success && i > 9) {
            break;
        }
    }
    
    // Free memory.
    free(in_array_cmp);
    free(out_array_cmp);
    free(sizes_array_cmp);
    
    return success;
}

#endif //HELPERS_COMMON
