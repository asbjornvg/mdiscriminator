#ifndef HELPERS_COMMON
#define HELPERS_COMMON

#include<string>
#include<cstdio>
#include<sys/time.h>
#include<time.h>
#include<cassert>
#include<stdint.h>
#include<numeric>

#include "SeqMdiscr.h"

#ifdef __CUDACC__
#define CUDA_DEVICE_HOST __device__ __host__
#else
#define CUDA_DEVICE_HOST
#endif

/* #define BEGIN_CATCH_HANDLER try { */
/* #define END_CATCH_HANDLER } catch(std::exception& e) { \ */
/*         std::cerr << " error: " << e.what() << \ */
/*             " (file " << __FILE__ << ", line " << __LINE__ << ")" << std::endl; } */

int nextMultOf(unsigned int x, unsigned int m) {
    if( x % m ) return x - (x % m) + m;
    else        return x;
}

/* Greatest common divisor. */
int gcd(int a, int b)
{
    for (;;)
    {
        if (a == 0) return b;
        b %= a;
        if (b == 0) return a;
        a %= b;
    }
}

/* Least common multiple. */
int lcm(int a, int b)
{
    int temp = gcd(a, b);

    return temp ? (a / temp * b) : 0;
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
                      unsigned int*       in_array
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
                        unsigned int*       in_array_sizes
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

/*
 * Collection = Anything that overloads operator[].
 * T should be a type that you can index into (with operator[]) and get back
 * an integer (that can be used with printf's %d).
 */
template<typename T>
void printIntCollection(unsigned int length, std::string title, T collection) {
    printf("%-12s [", (title + ":").c_str());
    bool first = true;
    for(unsigned int i = 0; i < length; i++) {
        if (first) {
            printf("%2d", collection[i]);
            first = false;
        }
        else {
            printf(", %2d", collection[i]);
        }
    }
    printf("]\n");
}

void printIntArray(unsigned int length, std::string title, unsigned int *arr) {
    printIntCollection<unsigned int*>(length, title, arr);
}

/*
 * This generalized version is meant to accomodate the case where the element
 * type is an array-type. This way, we can validate one position of the inner
 * dimension at the time. The stride is the size of the inner dimension, and
 * the offset is which position we are currently validating.
 */
template<class T, unsigned int STRIDE>
bool compareSegmentToOtherGeneral(T*            out_array,
                                  T*            out_array_cmp,
                                  unsigned int  num_elems,
                                  std::string   description,
                                  unsigned int  offset) {
    
    bool success = true;
    
    T out_element, out_element_cmp;
    unsigned int num_violations = 0;
    for(unsigned int i = 0 + offset; i < num_elems * STRIDE; i+= STRIDE) {
        // if ((i-offset) % (num_elems*STRIDE / 10) == 0 ) {
        //     fprintf(stderr, "i = %d\n", i);
        // }
        
        // Compare elements.
        out_element = out_array[i];
        out_element_cmp = out_array_cmp[i];
        if (out_element != out_element_cmp) {
            success = false;
            num_violations++;
            fprintf(stderr, "Violation (%s): %d should be %d (i = %d)\n", description.c_str(), out_element, out_element_cmp, i);
        }
        
        // Only print the first few violations.
        if (!success && num_violations > 9) {
            break;
        }
    }
    
    if (success) {
        fprintf(stderr, (description + " array valid\n").c_str());
    }
    
    return success;
}

template<class T>
bool compareSegmentToOther(T*            out_array,
                           T*            out_array_cmp,
                           unsigned int  num_elems,
                           std::string   description) {
    return compareSegmentToOtherGeneral<T, 1>
        (out_array, out_array_cmp, num_elems, description, 0);
}

/*
 * This validation function takes an input array, an output array, and a size
 * array.
 */
template<class ModN>
bool validateOneSegment(typename ModN::InType*  in_array,
                        typename ModN::InType*  out_array,
                        unsigned int*           sizes_array,
                        unsigned int            num_elems
    ) {
    
    bool success = true;
    
    // Allocate memory for the comparison (sequential) run.
    typename ModN::InType* out_array_cmp =
        (typename ModN::InType*) malloc(num_elems * sizeof(typename ModN::InType));
    unsigned int* sizes_array_cmp = (unsigned int*) malloc(num_elems * sizeof(int));
    
    // Run the sequential version.
    seqMdiscr<ModN, false>(num_elems, in_array, out_array_cmp, sizes_array_cmp);
    
    if (!compareSegmentToOther<typename ModN::InType>(out_array, out_array_cmp, num_elems, "element")) {
        success = false;
    }
    if (!compareSegmentToOther<unsigned int>(sizes_array, sizes_array_cmp, num_elems, "size")) {
        success = false;
    }
    
    // typename ModN::InType out_element, out_element_cmp;
    // unsigned int size, size_cmp;
    // unsigned int num_violations = 0;
    // for(unsigned int i = 0; i < num_elems; i++) {
        
    //     // Compare elements.
    //     out_element = out_array[i];
    //     out_element_cmp = out_array_cmp[i];
    //     if (out_element != out_element_cmp) {
    //         success = false;
    //         num_violations++;
    //         printf("Violation: %d should be %d (i = %d)\n", out_element, out_element_cmp, i);
    //     }
        
    //     // Compare sizes.
    //     size = sizes_array[i];
    //     size_cmp = sizes_array_cmp[i];
    //     if (size != size_cmp) {
    //         success = false;
    //         num_violations++;
    //         printf("Invalid size: %d should be %d (i = %d)\n", size, size_cmp, i);
    //     }
        
    //     // Only print the first few violations.
    //     if (!success && num_violations > 9) {
    //         break;
    //     }
    // }
    
    // Free memory.
    free(out_array_cmp);
    free(sizes_array_cmp);
    
    return success;
}

template<typename TupleType>
void tupleToSizeArray(TupleType      sizes,
                      unsigned int*  sizes_array,
                      unsigned int   num_elems) {
    
    // Initialize all sizes to 0.
    memset(sizes_array, 0, num_elems * sizeof(unsigned int));
    
    // Fill in the sizes at the right positions.
    unsigned int size = 0;
    unsigned int i = 0;
    unsigned int j = 0;
    for ( ; i < num_elems; i += size, j++) {
        size = sizes[j];
        sizes_array[i] = size;
    }
}

/*
 * This validation function takes an input array, an output array, and a size
 * tuple.
 */
template<class ModN>
bool validateOneSegment(typename ModN::InType*    in_array,
                        typename ModN::InType*    out_array,
                        typename ModN::TupleType  sizes,
                        unsigned int              num_elems
    ) {
#ifdef PRINT
    printIntCollection<typename ModN::TupleType>(ModN::TupleType::cardinal, "sizes", sizes);
#endif
    
    // Allocate storage for an actual size-array.
    unsigned int* sizes_array = (unsigned int*) malloc(num_elems * sizeof(unsigned int));
    
    // Convert the tuple into a size-array.
    tupleToSizeArray<typename ModN::TupleType>(sizes, sizes_array, num_elems);
    
    // Call the overload that takes a size-array.
    bool success = validateOneSegment<ModN>(in_array, out_array, sizes_array, num_elems);
    
    // Clean up.
    free(sizes_array);
    
    return success;
}

/* /\* */
/*  * This validation function takes an input array, an output array, and a size */
/*  * tuple. */
/*  *\/ */
/* template<class ModN> */
/* bool validateOneSegment(typename ModN::InType*    in_array, */
/*                         typename ModN::InType*    out_array, */
/*                         typename ModN::TupleType  sizes, */
/*                         unsigned int              num_elems */
/*     ) { */
    
/*     unsigned int i, j; */
/*     bool success = true; */
    
/*     // Allocate memory for the comparison (sequential) run. */
/*     typename ModN::InType* out_array_cmp = */
/*         (typename ModN::InType*) malloc(num_elems * sizeof(typename ModN::InType)); */
/*     unsigned int* sizes_array_cmp = (unsigned int*) malloc(num_elems * sizeof(int)); */
    
/*     // Run the sequential version. */
/*     seqMdiscr<ModN, false>(num_elems, in_array, out_array_cmp, sizes_array_cmp); */
    
/*     unsigned int size = 0; */
/*     unsigned int size_cmp = 0; */
/*     i = 0; */
/*     j = 0; */
/*     for ( ; i < num_elems; i += size, j++) { */
/*         size = sizes[j]; */
/*         size_cmp = sizes_array_cmp[i]; */
/*         if (size != size_cmp) { */
/*             success = false; */
/*             fprintf(stderr, "Invalid size: %d should be %d (i = %d, j = %d)\n", size, size_cmp, i, j); */
/*         } */
        
/*         // Only print the first few violations. */
/*         if (!success && j > 9) { */
/*             break; */
/*         } */
/*     } */
    
/*     typename ModN::InType out_element, out_element_cmp; */
/*     for(i = 0; i < num_elems; i++) { */
        
/*         // Compare elements. */
/*         out_element = out_array[i]; */
/*         out_element_cmp = out_array_cmp[i]; */
/*         if (out_element != out_element_cmp) { */
/*             success = false; */
/*             fprintf(stderr, "Violation: %d should be %d (i = %d)\n", out_element, out_element_cmp, i); */
/*         } */
        
/*         // Only print the first few violations. */
/*         if (!success && i > 9) { */
/*             break; */
/*         } */
/*     } */
    
/*     // Free memory. */
/*     free(out_array_cmp); */
/*     free(sizes_array_cmp); */
    
/*     return success; */
/* } */

template<class T>
class Add {
public:
    typedef T BaseType;
    CUDA_DEVICE_HOST static inline T identity() {
        return T();
    }
    CUDA_DEVICE_HOST static inline T apply(const T t1, const T t2) {
        return t1 + t2;
    }
};

template<typename, unsigned int> class MyInt4Packed;

class MyInt4 {
public:
    unsigned int x; unsigned int y; unsigned int z; unsigned int w;
    static const unsigned int cardinal = 4;
    typedef MyInt4Packed<uint32_t, 8> SmallType;
    typedef MyInt4Packed<uint64_t, 16> MediumType;
    
    CUDA_DEVICE_HOST inline MyInt4()
        : x(0), y(0), z(0), w(0) {
    }
    CUDA_DEVICE_HOST inline MyInt4(const unsigned int& a,
                                   const unsigned int& b,
                                   const unsigned int& c,
                                   const unsigned int& d)
        : x(a), y(b), z(c), w(d) {
    }
    CUDA_DEVICE_HOST inline MyInt4(const MyInt4& i4)
        : x(i4.x), y(i4.y), z(i4.z), w(i4.w) {
    }
    CUDA_DEVICE_HOST inline MyInt4(const volatile MyInt4& i4)
        : x(i4.x), y(i4.y), z(i4.z), w(i4.w) {
    }
    CUDA_DEVICE_HOST volatile inline MyInt4& operator=(const MyInt4& i4) volatile {
        x = i4.x; y = i4.y; z = i4.z; w = i4.w;
        return *this;
    }
    CUDA_DEVICE_HOST inline MyInt4& operator=(volatile const MyInt4& i4) {
        x = i4.x; y = i4.y; z = i4.z; w = i4.w;
        return *this;
    }
    CUDA_DEVICE_HOST inline MyInt4& operator=(const MyInt4& i4) {
        x = i4.x; y = i4.y; z = i4.z; w = i4.w;
        return *this;
    }
    CUDA_DEVICE_HOST inline MyInt4& operator+=(const MyInt4& i4) {
        x += i4.x; y += i4.y; z += i4.z; w += i4.w;
        return *this;
    }
    CUDA_DEVICE_HOST inline friend MyInt4 operator+(const MyInt4& m1, const MyInt4& m2) {
        return MyInt4(m1.x+m2.x, m1.y+m2.y, m1.z+m2.z, m1.w+m2.w);
    }
    CUDA_DEVICE_HOST inline unsigned int& operator[](const unsigned int i) {
        assert(i < 4);
        if (i == 0) return x;
        else if (i == 1) return y;
        else if (i == 2) return z;
        else return w; // i == 3
    }
    CUDA_DEVICE_HOST volatile inline unsigned int& operator[](const unsigned int i) volatile {
        assert(i < 4);
        if (i == 0) return x;
        else if (i == 1) return y;
        else if (i == 2) return z;
        else return w; // i == 3
    }
    CUDA_DEVICE_HOST inline void zeroOut() {
        x = y = z = w = 0; 
    }
    CUDA_DEVICE_HOST inline void set(const volatile MyInt4& i4) {
        x = i4.x; y = i4.y; z = i4.z; w = i4.w;
    }
#ifdef __CUDACC__
    __device__ inline void adjustRowToBlock( unsigned int*        arrtmp,
                                             MyInt4*           blk_beg,
                                             MyInt4*           blk_prv,
                                             MyInt4*           blk_end
    ) {
        unsigned int beg_blk, tmp_diff = 0;
        
        beg_blk   = (blockIdx.x > 0) * blk_beg->x;
        x        += tmp_diff + (threadIdx.x > 0) * (blk_prv->x - beg_blk);
        arrtmp[0] = blk_end->x - beg_blk;
        tmp_diff += arrtmp[0];
        
        beg_blk   = (blockIdx.x > 0) * blk_beg->y;
        y        += tmp_diff + (threadIdx.x > 0) * (blk_prv->y - beg_blk);
        arrtmp[1] = blk_end->y - beg_blk;
        tmp_diff += arrtmp[1];
        
        beg_blk   = (blockIdx.x > 0) * blk_beg->z;
        z        += tmp_diff + (threadIdx.x > 0) * (blk_prv->z - beg_blk);
        arrtmp[2] = blk_end->z - beg_blk;
        tmp_diff += arrtmp[2];        
        
        beg_blk   = (blockIdx.x > 0) * blk_beg->w;
        w        += tmp_diff + (threadIdx.x > 0) * (blk_prv->w - beg_blk);
        arrtmp[3] = blk_end->w - beg_blk;
    }
#endif
};

template<typename uintXX_t, unsigned int bits>
class MyInt4Packed {
    uintXX_t p;
    static const unsigned int mask = (1u << bits) - 1;
public:
    CUDA_DEVICE_HOST inline MyInt4Packed() : p(0) {}
    CUDA_DEVICE_HOST inline MyInt4Packed(uintXX_t init) : p(init) {}
    
    // Post increment.
    CUDA_DEVICE_HOST inline unsigned int increment(unsigned int eq_class) volatile {
        assert(eq_class < 4);
        unsigned int tmp = (p >> bits*eq_class) & mask;
        p += (static_cast<uintXX_t>(1) << (bits*eq_class));
        return tmp;
    }
    
    CUDA_DEVICE_HOST inline operator MyInt4 () volatile {
        unsigned int x = p & mask;
        unsigned int y = (p >> bits) & mask;
        unsigned int z = (p >> bits*2) & mask;
        unsigned int w = (p >> bits*3) & mask;
        return MyInt4(x,y,z,w);
    }
    
    CUDA_DEVICE_HOST inline friend MyInt4Packed<uintXX_t, bits>
    operator+(const MyInt4Packed<uintXX_t, bits>& m1, const MyInt4Packed<uintXX_t, bits>& m2) {
        return MyInt4Packed<uintXX_t, bits>(m1.p + m2.p);
    }
    
    CUDA_DEVICE_HOST volatile inline MyInt4Packed<uintXX_t, bits>&
    operator=(const MyInt4Packed<uintXX_t, bits>& i4) volatile {
        p = i4.p;
        return *this;
    }
    
    CUDA_DEVICE_HOST inline MyInt4Packed<uintXX_t, bits>&
    operator=(volatile const MyInt4Packed<uintXX_t, bits>& i4) {
        p = i4.p;
        return *this;
    }
    
#ifdef __CUDACC__
    __device__ inline void adjustRowToBlock( unsigned int*        arrtmp,
                                             MyInt4*           blk_beg,
                                             MyInt4*           blk_prv,
                                             MyInt4*           blk_end
    ) {
        unsigned int beg_blk, tmp_diff = 0;
        unsigned int current = 0;
        
        beg_blk   = (blockIdx.x > 0) * blk_beg->x;
        current   = tmp_diff + (threadIdx.x > 0) * (blk_prv->x - beg_blk);
        p        += static_cast<uintXX_t>(current);
        arrtmp[0] = blk_end->x - beg_blk;
        tmp_diff += arrtmp[0];
        
        beg_blk   = (blockIdx.x > 0) * blk_beg->y;
        current   = tmp_diff + (threadIdx.x > 0) * (blk_prv->y - beg_blk);
        p        += (static_cast<uintXX_t>(current) << bits);
        arrtmp[1] = blk_end->y - beg_blk;
        tmp_diff += arrtmp[1];
        
        beg_blk   = (blockIdx.x > 0) * blk_beg->z;
        current   = tmp_diff + (threadIdx.x > 0) * (blk_prv->z - beg_blk);
        p        += (static_cast<uintXX_t>(current) << (bits*2));
        arrtmp[2] = blk_end->z - beg_blk;
        tmp_diff += arrtmp[2];
        
        beg_blk   = (blockIdx.x > 0) * blk_beg->w;
        current   = tmp_diff + (threadIdx.x > 0) * (blk_prv->w - beg_blk);
        p        += (static_cast<uintXX_t>(current) << (bits*3));
        arrtmp[3] = blk_end->w - beg_blk;
    }
#endif
};

class Mod4 {
public:
    typedef unsigned int         InType;
    // The out-type is always unsigned int for these discriminators.
    typedef MyInt4               TupleType;
    static const InType padelm = 3;
    CUDA_DEVICE_HOST static inline unsigned int apply(volatile InType x) {
        return x & 3;
    }
};

template<typename, unsigned int> class MyInt8Packed;

class MyInt8 {
public:
    unsigned int x0; unsigned int x1; unsigned int x2; unsigned int x3;
    unsigned int x4; unsigned int x5; unsigned int x6; unsigned int x7;
    static const unsigned int cardinal = 8;
    typedef MyInt8Packed<uint32_t, 8> SmallType;
    typedef MyInt8Packed<uint64_t, 16> MediumType;
    
    CUDA_DEVICE_HOST inline MyInt8()
        : x0(0), x1(0), x2(0), x3(0), x4(0), x5(0), x6(0), x7(0) {
    }
    CUDA_DEVICE_HOST inline MyInt8(const unsigned int& y0,
                                   const unsigned int& y1,
                                   const unsigned int& y2,
                                   const unsigned int& y3,
                                   const unsigned int& y4,
                                   const unsigned int& y5,
                                   const unsigned int& y6,
                                   const unsigned int& y7)
        : x0(y0), x1(y1), x2(y2), x3(y3), x4(y4), x5(y5), x6(y6), x7(y7) {
    }
    CUDA_DEVICE_HOST inline MyInt8(const MyInt8& i8)
        : x0(i8.x0), x1(i8.x1), x2(i8.x2), x3(i8.x3), x4(i8.x4), x5(i8.x5), x6(i8.x6), x7(i8.x7) {
    }
    CUDA_DEVICE_HOST inline MyInt8(const volatile MyInt8& i8)
        : x0(i8.x0), x1(i8.x1), x2(i8.x2), x3(i8.x3), x4(i8.x4), x5(i8.x5), x6(i8.x6), x7(i8.x7) {
    }
    CUDA_DEVICE_HOST volatile inline MyInt8& operator=(const MyInt8& i8) volatile {
        x0 = i8.x0; x1 = i8.x1; x2 = i8.x2; x3 = i8.x3; x4 = i8.x4; x5 = i8.x5; x6 = i8.x6; x7 = i8.x7;
        return *this;
    }
    CUDA_DEVICE_HOST inline MyInt8& operator=(volatile const MyInt8& i8) {
        x0 = i8.x0; x1 = i8.x1; x2 = i8.x2; x3 = i8.x3; x4 = i8.x4; x5 = i8.x5; x6 = i8.x6; x7 = i8.x7;
        return *this;
    }
    CUDA_DEVICE_HOST inline MyInt8& operator=(const MyInt8& i8) {
        x0 = i8.x0; x1 = i8.x1; x2 = i8.x2; x3 = i8.x3; x4 = i8.x4; x5 = i8.x5; x6 = i8.x6; x7 = i8.x7;
        return *this;
    }
    CUDA_DEVICE_HOST inline MyInt8& operator+=(const MyInt8& i8) {
        x0 += i8.x0; x1 += i8.x1; x2 += i8.x2; x3 += i8.x3; x4 += i8.x4; x5 += i8.x5; x6 += i8.x6; x7 += i8.x7;
        return *this;
    }
    CUDA_DEVICE_HOST inline friend MyInt8 operator+(const MyInt8& m1, const MyInt8& m2) {
        return MyInt8(m1.x0+m2.x0, m1.x1+m2.x1, m1.x2+m2.x2, m1.x3+m2.x3, m1.x4+m2.x4, m1.x5+m2.x5, m1.x6+m2.x6, m1.x7+m2.x7);
    }
    CUDA_DEVICE_HOST inline unsigned int& operator[](const unsigned int i) {
        assert(i < 8);
        if (i == 0) return x0;
        else if (i == 1) return x1;
        else if (i == 2) return x2;
        else if (i == 3) return x3;
        else if (i == 4) return x4;
        else if (i == 5) return x5;
        else if (i == 6) return x6;
        else return x7; // i == 7
    }
    CUDA_DEVICE_HOST volatile inline unsigned int& operator[](const unsigned int i) volatile {
        assert(i < 8);
        if (i == 0) return x0;
        else if (i == 1) return x1;
        else if (i == 2) return x2;
        else if (i == 3) return x3;
        else if (i == 4) return x4;
        else if (i == 5) return x5;
        else if (i == 6) return x6;
        else return x7; // i == 7
    }
    CUDA_DEVICE_HOST inline void zeroOut() {
        x0 = x1 = x2 = x3 = x4 = x5 = x6 = x7 = 0;
    }
    CUDA_DEVICE_HOST inline void set(const volatile MyInt8& i8) {
        x0 = i8.x0; x1 = i8.x1; x2 = i8.x2; x3 = i8.x3; x4 = i8.x4; x5 = i8.x5; x6 = i8.x6; x7 = i8.x7;
    }
#ifdef __CUDACC__
    __device__ inline void adjustRowToBlock( unsigned int*        arrtmp,
                                             MyInt8*           blk_beg,
                                             MyInt8*           blk_prv,
                                             MyInt8*           blk_end
    ) {
        unsigned int beg_blk, tmp_diff = 0;
        
        beg_blk   = (blockIdx.x > 0) * blk_beg->x0;
        x0       += tmp_diff + (threadIdx.x > 0) * (blk_prv->x0 - beg_blk);
        arrtmp[0] = blk_end->x0 - beg_blk;
        tmp_diff += arrtmp[0];
        
        beg_blk   = (blockIdx.x > 0) * blk_beg->x1;
        x1       += tmp_diff + (threadIdx.x > 0) * (blk_prv->x1 - beg_blk);
        arrtmp[1] = blk_end->x1 - beg_blk;
        tmp_diff += arrtmp[1];
        
        beg_blk   = (blockIdx.x > 0) * blk_beg->x2;
        x2       += tmp_diff + (threadIdx.x > 0) * (blk_prv->x2 - beg_blk);
        arrtmp[2] = blk_end->x2 - beg_blk;
        tmp_diff += arrtmp[2];
        
        beg_blk   = (blockIdx.x > 0) * blk_beg->x3;
        x3       += tmp_diff + (threadIdx.x > 0) * (blk_prv->x3 - beg_blk);
        arrtmp[3] = blk_end->x3 - beg_blk;
        tmp_diff += arrtmp[3];
        
        beg_blk   = (blockIdx.x > 0) * blk_beg->x4;
        x4       += tmp_diff + (threadIdx.x > 0) * (blk_prv->x4 - beg_blk);
        arrtmp[4] = blk_end->x4 - beg_blk;
        tmp_diff += arrtmp[4];
        
        beg_blk   = (blockIdx.x > 0) * blk_beg->x5;
        x5       += tmp_diff + (threadIdx.x > 0) * (blk_prv->x5 - beg_blk);
        arrtmp[5] = blk_end->x5 - beg_blk;
        tmp_diff += arrtmp[5];
        
        beg_blk   = (blockIdx.x > 0) * blk_beg->x6;
        x6       += tmp_diff + (threadIdx.x > 0) * (blk_prv->x6 - beg_blk);
        arrtmp[6] = blk_end->x6 - beg_blk;
        tmp_diff += arrtmp[6];
        
        beg_blk   = (blockIdx.x > 0) * blk_beg->x7;
        x7       += tmp_diff + (threadIdx.x > 0) * (blk_prv->x7 - beg_blk);
        arrtmp[7] = blk_end->x7 - beg_blk;
    }
#endif
};

template<typename uintXX_t, unsigned int bits>
class MyInt8Packed {
    uintXX_t p0;
    uintXX_t p1;
    static const unsigned int mask = (1u << bits) - 1;
public:
    CUDA_DEVICE_HOST inline MyInt8Packed() : p0(0), p1(0) {}
    CUDA_DEVICE_HOST inline MyInt8Packed(uintXX_t init0, uintXX_t init1)
        : p0(init0), p1(init1) {}
    
    // Post increment.
    CUDA_DEVICE_HOST inline unsigned int increment(unsigned int eq_class) volatile {
        assert(eq_class < 8);
        if (eq_class < 4) {
            unsigned int tmp = (p0 >> bits*eq_class) & mask;
            p0 += (static_cast<uintXX_t>(1) << (bits*eq_class));
            return tmp;
        }
        else {
            eq_class -= 4;
            unsigned int tmp = (p1 >> bits*eq_class) & mask;
            p1 += (static_cast<uintXX_t>(1) << (bits*eq_class));
            return tmp;
        }
    }
    
    CUDA_DEVICE_HOST inline operator MyInt8 () volatile {
        unsigned int x0 = p0 & mask;
        unsigned int x1 = (p0 >> bits) & mask;
        unsigned int x2 = (p0 >> bits*2) & mask;
        unsigned int x3 = (p0 >> bits*3) & mask;
        unsigned int x4 = p1 & mask;
        unsigned int x5 = (p1 >> bits) & mask;
        unsigned int x6 = (p1 >> bits*2) & mask;
        unsigned int x7 = (p1 >> bits*3) & mask;
        return MyInt8(x0,x1,x2,x3,x4,x5,x6,x7);
    }
    
    CUDA_DEVICE_HOST inline friend MyInt8Packed<uintXX_t,bits>
    operator+(const MyInt8Packed<uintXX_t,bits>& m1, const MyInt8Packed<uintXX_t,bits>& m2) {
        return MyInt8Packed<uintXX_t,bits>(m1.p0 + m2.p0, m1.p1 + m2.p1);
    }
    
    CUDA_DEVICE_HOST volatile inline MyInt8Packed<uintXX_t,bits>&
    operator=(const MyInt8Packed<uintXX_t,bits>& i8) volatile {
        p0 = i8.p0;
        p1 = i8.p1;
        return *this;
    }
    
    CUDA_DEVICE_HOST inline MyInt8Packed<uintXX_t,bits>&
    operator=(volatile const MyInt8Packed<uintXX_t,bits>& i8) {
        p0 = i8.p0;
        p1 = i8.p1;
        return *this;
    }
    
#ifdef __CUDACC__
    __device__ inline void adjustRowToBlock( unsigned int*        arrtmp,
                                             MyInt8*           blk_beg,
                                             MyInt8*           blk_prv,
                                             MyInt8*           blk_end
    ) {
        unsigned int beg_blk, tmp_diff = 0;
        unsigned int current = 0;
        
        beg_blk   = (blockIdx.x > 0) * blk_beg->x0;
        current   = tmp_diff + (threadIdx.x > 0) * (blk_prv->x0 - beg_blk);
        p0       += static_cast<uintXX_t>(current);
        arrtmp[0] = blk_end->x0 - beg_blk;
        tmp_diff += arrtmp[0];
        
        beg_blk   = (blockIdx.x > 0) * blk_beg->x1;
        current   = tmp_diff + (threadIdx.x > 0) * (blk_prv->x1 - beg_blk);
        p0       += (static_cast<uintXX_t>(current) << bits);
        arrtmp[1] = blk_end->x1 - beg_blk;
        tmp_diff += arrtmp[1];
        
        beg_blk   = (blockIdx.x > 0) * blk_beg->x2;
        current   = tmp_diff + (threadIdx.x > 0) * (blk_prv->x2 - beg_blk);
        p0       += (static_cast<uintXX_t>(current) << (bits*2));
        arrtmp[2] = blk_end->x2 - beg_blk;
        tmp_diff += arrtmp[2];
        
        beg_blk   = (blockIdx.x > 0) * blk_beg->x3;
        current   = tmp_diff + (threadIdx.x > 0) * (blk_prv->x3 - beg_blk);
        p0       += (static_cast<uintXX_t>(current) << (bits*3));
        arrtmp[3] = blk_end->x3 - beg_blk;
        tmp_diff += arrtmp[3];
        
        beg_blk   = (blockIdx.x > 0) * blk_beg->x4;
        current   = tmp_diff + (threadIdx.x > 0) * (blk_prv->x4 - beg_blk);
        p1       += static_cast<uintXX_t>(current);
        arrtmp[4] = blk_end->x4 - beg_blk;
        tmp_diff += arrtmp[4];
        
        beg_blk   = (blockIdx.x > 0) * blk_beg->x5;
        current   = tmp_diff + (threadIdx.x > 0) * (blk_prv->x5 - beg_blk);
        p1       += (static_cast<uintXX_t>(current) << bits);
        arrtmp[5] = blk_end->x5 - beg_blk;
        tmp_diff += arrtmp[5];
        
        beg_blk   = (blockIdx.x > 0) * blk_beg->x6;
        current   = tmp_diff + (threadIdx.x > 0) * (blk_prv->x6 - beg_blk);
        p1       += (static_cast<uintXX_t>(current) << (bits*2));
        arrtmp[6] = blk_end->x6 - beg_blk;
        tmp_diff += arrtmp[6];
        
        beg_blk   = (blockIdx.x > 0) * blk_beg->x7;
        current   = tmp_diff + (threadIdx.x > 0) * (blk_prv->x7 - beg_blk);
        p1       += (static_cast<uintXX_t>(current) << (bits*3));
        arrtmp[7] = blk_end->x7 - beg_blk;
    }
#endif
};

class Mod8 {
public:
    typedef unsigned int         InType;
    // The out-type is always unsigned int for these discriminators.
    typedef MyInt8               TupleType;
    static const InType padelm = 7;
    CUDA_DEVICE_HOST static inline unsigned int apply(volatile InType x) {
        return x & 7;
    }
};

template<unsigned int, typename, unsigned int> class MyIntPackedV1;
template<unsigned int, typename, typename, unsigned int> class MyIntPackedV2;
template<unsigned int, typename, unsigned int, unsigned int> class MyIntPackedV3;

template<unsigned int N>
class MyInt {
public:
    unsigned int arr[N];
    static const unsigned int cardinal = N;
    
#if defined PACKED_V1
    typedef MyIntPackedV1<N, uint32_t, 32/N> SmallType;
    typedef MyIntPackedV1<N, uint64_t, 64/N> MediumType;
    
#elif defined PACKED_V2
/*     typedef MyIntPackedV2<N, uint64_t, uint16_t, 4> MediumType; */
/* #if NUM_CLASSES <= 4 */
/*     typedef MyIntPackedV2<N, uint32_t, unsigned char, 4> SmallType; */
/* #else */
/*     typedef MyIntPackedV2<N, uint64_t, unsigned char, 8> SmallType; */
    typedef MyIntPackedV2<N, uint64_t, uint16_t, 2> MediumType;
#if NUM_CLASSES <= 4
    typedef MyIntPackedV2<N, uint32_t, unsigned char, 2> SmallType;
#else
    typedef MyIntPackedV2<N, uint64_t, unsigned char, 3> SmallType;
#endif
    
#elif defined PACKED_V3
/*     typedef MyIntPackedV3<N, uint64_t, 16, 4> MediumType; */
/* #if NUM_CLASSES <= 4 */
/*     typedef MyIntPackedV3<N, uint32_t, 8, 4> SmallType; */
/* #else */
/*     typedef MyIntPackedV3<N, uint64_t, 8, 8> SmallType; */
    typedef MyIntPackedV3<N, uint64_t, 16, 2> MediumType;
#if NUM_CLASSES <= 4
    typedef MyIntPackedV3<N, uint32_t, 8, 2> SmallType;
#else
    typedef MyIntPackedV3<N, uint64_t, 8, 3> SmallType;
#endif
#endif
    
    CUDA_DEVICE_HOST unsigned int& operator[](const unsigned int i) {
        assert(i < N);
        return arr[i];
    }
    CUDA_DEVICE_HOST volatile unsigned int& operator[](const unsigned int i) volatile {
        assert(i < N);
        return arr[i];
    }
    CUDA_DEVICE_HOST inline MyInt()
        : arr() {
    }
    CUDA_DEVICE_HOST inline MyInt(const MyInt<N>& other) {
        for (unsigned int i = 0; i < N; i++) {
            arr[i] = other.arr[i];
        }
    }
    CUDA_DEVICE_HOST inline MyInt(const volatile MyInt<N>& other) {
        for (unsigned int i = 0; i < N; i++) {
            arr[i] = other.arr[i];
        }
    }
    CUDA_DEVICE_HOST volatile inline MyInt<N>& operator=(const MyInt<N>& other) volatile {
        for (unsigned int i = 0; i < N; i++) {
            arr[i] = other.arr[i];
        }
        return *this;
    }
    CUDA_DEVICE_HOST inline MyInt<N>& operator=(volatile const MyInt<N>& other) {
        for (unsigned int i = 0; i < N; i++) {
            arr[i] = other.arr[i];
        }
        return *this;
    }
    CUDA_DEVICE_HOST inline MyInt<N>& operator=(const MyInt<N>& other) {
        for (unsigned int i = 0; i < N; i++) {
            arr[i] = other.arr[i];
        }
        return *this;
    }
    CUDA_DEVICE_HOST friend MyInt<N> operator+(const MyInt<N> &m1, const MyInt<N> &m2) {
        MyInt<N> m;
        for (unsigned int i = 0; i < N; i++) {
            m.arr[i] = m1.arr[i] + m2.arr[i];
        }
        return m;
    }
    CUDA_DEVICE_HOST inline void zeroOut() {
        for (unsigned int i = 0; i < N; i++) {
            arr[i] = 0;
        }
    }
    CUDA_DEVICE_HOST inline void set(const volatile MyInt<N>& other) {
        for (unsigned int i = 0; i < N; i++) {
            arr[i] = other.arr[i];
        }
    }
#ifdef __CUDACC__
    __device__ inline void adjustRowToBlock( unsigned int*  arrtmp,
                                             MyInt<N>*      blk_beg,
                                             MyInt<N>*      blk_prv,
                                             MyInt<N>*      blk_end
    ) {
        unsigned int beg_blk, tmp_diff = 0;
        
#pragma unroll
        for (unsigned int k = 0; k < N; k++) {
            beg_blk   = (blockIdx.x > 0) * blk_beg->arr[k];
            arr[k]   += tmp_diff + (threadIdx.x > 0) * (blk_prv->arr[k] - beg_blk);
            arrtmp[k] = blk_end->arr[k] - beg_blk;
            tmp_diff += arrtmp[k];
        }
    }
#endif
};

/*
 * The following only works for m = 4.
 */
template<unsigned int N, typename uintXX_t, unsigned int bits>
class MyIntPackedV1 {
    uintXX_t p;
    static const unsigned int mask = (1u << bits) - 1;
public:
    CUDA_DEVICE_HOST inline MyIntPackedV1() : p(0) {}
    CUDA_DEVICE_HOST inline MyIntPackedV1(uintXX_t init) : p(init) {}
    
    // Post increment.
    CUDA_DEVICE_HOST inline unsigned int increment(unsigned int eq_class) volatile {
        assert(eq_class < N);
        unsigned int tmp = (p >> bits*eq_class) & mask;
        p += (static_cast<uintXX_t>(1) << (bits*eq_class));
        return tmp;
    }
    
    CUDA_DEVICE_HOST inline operator MyInt<N> () volatile {
        MyInt<N> m;
        //unsigned int mask = (1u << bits) - 1;
        
        for (unsigned int i = 0; i < N; i++) {
            m[i] = (p >> bits*i) & mask;
        }
        
        return m;
    }
    
    CUDA_DEVICE_HOST inline friend MyIntPackedV1<N, uintXX_t, bits>
    operator+(const MyIntPackedV1<N, uintXX_t, bits>& m1, const MyIntPackedV1<N, uintXX_t, bits>& m2) {
        return MyIntPackedV1<N, uintXX_t, bits>(m1.p + m2.p);
    }
    
    CUDA_DEVICE_HOST volatile inline MyIntPackedV1<N, uintXX_t, bits>&
    operator=(const MyIntPackedV1<N, uintXX_t, bits>& other) volatile {
        p = other.p;
        return *this;
    }
    
    CUDA_DEVICE_HOST inline MyIntPackedV1<N, uintXX_t, bits>&
    operator=(volatile const MyIntPackedV1<N, uintXX_t, bits>& other) {
        p = other.p;
        return *this;
    }
    
#ifdef __CUDACC__
    __device__ inline void adjustRowToBlock( unsigned int*        arrtmp,
                                             MyInt<N>*            blk_beg,
                                             MyInt<N>*            blk_prv,
                                             MyInt<N>*            blk_end
    ) {
        unsigned int beg_blk, tmp_diff = 0;
        unsigned int current = 0;
        
        for (unsigned int k = 0; k < N; k++) {
            beg_blk   = (blockIdx.x > 0) * blk_beg->arr[k];
            current   = tmp_diff + (threadIdx.x > 0) * (blk_prv->arr[k] - beg_blk);
            p        += (static_cast<uintXX_t>(current) << (bits*k));
            arrtmp[k] = blk_end->arr[k] - beg_blk;
            tmp_diff += arrtmp[k];
        }
    }
#endif
};

/*
 * ratio: How many times does a uintXX_t fit into a uintLARGE_t.
 * log2ratio: We only use ratios that are powers of 2, log2ratio is the base 2
 * logarithm of the ratio.
 */
/* template<unsigned int N, typename uintLARGE_t, typename uintXX_t, unsigned int ratio> */
template<unsigned int N, typename uintLARGE_t, typename uintXX_t, unsigned int log2ratio>
class MyIntPackedV2 {
    static const unsigned int quotient = (N-1 >> log2ratio) + 1;
    static const unsigned int ratio = 1 << log2ratio;
    uintXX_t arr[quotient*ratio]; // Multiple of ratio
public:
    CUDA_DEVICE_HOST inline MyIntPackedV2() : arr() {}
    
    // Post increment.
    CUDA_DEVICE_HOST inline unsigned int increment(unsigned int eq_class) volatile {
        assert(eq_class < N);
        return arr[eq_class]++;
    }
    
    CUDA_DEVICE_HOST inline operator MyInt<N> () volatile {
        MyInt<N> m;
#pragma unroll
        for (unsigned int i = 0; i < N; i++) {
            m[i] = arr[i];
        }
        return m;
    }
    
    CUDA_DEVICE_HOST inline friend MyIntPackedV2<N,uintLARGE_t,uintXX_t,log2ratio>
    operator+(const MyIntPackedV2<N,uintLARGE_t,uintXX_t,log2ratio>& m1, const MyIntPackedV2<N,uintLARGE_t,uintXX_t,log2ratio>& m2) {
        MyIntPackedV2<N,uintLARGE_t,uintXX_t,log2ratio> m;
        uintLARGE_t* m_64 = reinterpret_cast<uintLARGE_t*>(m.arr);
        const uintLARGE_t* m1_64 = reinterpret_cast<const uintLARGE_t*>(m1.arr);
        const uintLARGE_t* m2_64 = reinterpret_cast<const uintLARGE_t*>(m2.arr);
#pragma unroll
        for (unsigned int i = 0; i < quotient; i++) {
            m_64[i] = m1_64[i] + m2_64[i];
        }
        return m;
    }
    
    CUDA_DEVICE_HOST volatile inline MyIntPackedV2<N,uintLARGE_t,uintXX_t,log2ratio>&
    operator=(const MyIntPackedV2<N,uintLARGE_t,uintXX_t,log2ratio>& other) volatile {
        volatile uintLARGE_t* arr_64 = reinterpret_cast<volatile uintLARGE_t*>(arr);
        const uintLARGE_t* other_64 = reinterpret_cast<const uintLARGE_t*>(other.arr);
#pragma unroll
        for (unsigned int i = 0; i < quotient; i++) {
            arr_64[i] = other_64[i];
        }
        return *this;
    }
    
    CUDA_DEVICE_HOST inline MyIntPackedV2<N,uintLARGE_t,uintXX_t,log2ratio>&
    operator=(volatile const MyIntPackedV2<N,uintLARGE_t,uintXX_t,log2ratio>& other) {
        uintLARGE_t* arr_64 = reinterpret_cast<uintLARGE_t*>(arr);
        volatile const uintLARGE_t* other_64 = reinterpret_cast<volatile const uintLARGE_t*>(other.arr);
#pragma unroll
        for (unsigned int i = 0; i < quotient; i++) {
            arr_64[i] = other_64[i];
        }
        return *this;
    }
    
#ifdef __CUDACC__
    __device__ inline void adjustRowToBlock( unsigned int*        arrtmp,
                                             MyInt<N>*            blk_beg,
                                             MyInt<N>*            blk_prv,
                                             MyInt<N>*            blk_end
    ) {
        unsigned int beg_blk, tmp_diff = 0;
        
#pragma unroll
        for (unsigned int k = 0; k < N; k++) {
            beg_blk   = (blockIdx.x > 0) * blk_beg->arr[k];
            arr[k]   += tmp_diff + (threadIdx.x > 0) * (blk_prv->arr[k] - beg_blk);
            arrtmp[k] = blk_end->arr[k] - beg_blk;
            tmp_diff += arrtmp[k];
        }
    }
#endif
};

/*
 * Here, instead of having an array of uint16_t which we then cast to uint64_t
 * when adding etc., it is the opposite: We have an array of uint64_t and then
 * we need to do bit operations on the appropriate array element, selected
 * using division and modulo.
 *
 * ratio: How many times do "bits" fit into a uintLARGE_t.
 */
template<unsigned int N, typename uintLARGE_t, unsigned int bits, unsigned int log2ratio>
class MyIntPackedV3 {
    static const unsigned int quotient = ((N-1) >> log2ratio) + 1;
    static const unsigned int ratio = 1 << log2ratio;
    uintLARGE_t arr[quotient];
    static const unsigned int mask = (1u << bits) - 1;
public:
    CUDA_DEVICE_HOST inline MyIntPackedV3() : arr() {}
    
    // Post increment.
    CUDA_DEVICE_HOST inline unsigned int increment(unsigned int eq_class) volatile {
        assert(eq_class < N);
        unsigned int tmp = (arr[eq_class >> log2ratio] >> bits*(eq_class & ratio-1)) & mask;
        arr[eq_class >> log2ratio] += static_cast<uintLARGE_t>(1) << bits*(eq_class & ratio-1);
        return tmp;
    }
    
    CUDA_DEVICE_HOST inline operator MyInt<N> () volatile {
        MyInt<N> m;
#pragma unroll
        for (unsigned int i = 0; i < N; i++) {
            m[i] = (arr[i >> log2ratio] >> bits*(i & (ratio-1))) & mask;
        }
        return m;
    }
    
    CUDA_DEVICE_HOST inline friend MyIntPackedV3<N,uintLARGE_t,bits,log2ratio>
    operator+(const MyIntPackedV3<N,uintLARGE_t,bits,log2ratio>& m1, const MyIntPackedV3<N,uintLARGE_t,bits,log2ratio>& m2) {
        MyIntPackedV3<N,uintLARGE_t,bits,log2ratio> m;
#pragma unroll
        for (unsigned int i = 0; i < quotient; i++) {
            m.arr[i] = m1.arr[i] + m2.arr[i];
        }
        return m;
    }
    
    CUDA_DEVICE_HOST volatile inline MyIntPackedV3<N,uintLARGE_t,bits,log2ratio>&
    operator=(const MyIntPackedV3<N,uintLARGE_t,bits,log2ratio>& other) volatile {
#pragma unroll
        for (unsigned int i = 0; i < quotient; i++) {
            arr[i] = other.arr[i];
        }
        return *this;
    }
    
    CUDA_DEVICE_HOST inline MyIntPackedV3<N,uintLARGE_t,bits,log2ratio>&
    operator=(volatile const MyIntPackedV3<N,uintLARGE_t,bits,log2ratio>& other) {
#pragma unroll
        for (unsigned int i = 0; i < quotient; i++) {
            arr[i] = other.arr[i];
        }
        return *this;
    }
    
#ifdef __CUDACC__
    __device__ inline void adjustRowToBlock( unsigned int*        arrtmp,
                                             MyInt<N>*            blk_beg,
                                             MyInt<N>*            blk_prv,
                                             MyInt<N>*            blk_end
    ) {
        unsigned int beg_blk, tmp_diff = 0;
        uintLARGE_t current = 0;
        
#pragma unroll
        for (unsigned int k = 0; k < N; k++) {
            beg_blk       = (blockIdx.x > 0) * blk_beg->arr[k];
            current       = tmp_diff + (threadIdx.x > 0) * (blk_prv->arr[k] - beg_blk);
            arr[k >> log2ratio] += current << bits*(k & (ratio-1));
            arrtmp[k]     = blk_end->arr[k] - beg_blk;
            tmp_diff     += arrtmp[k];
        }
    }
#endif
};

template<unsigned int N>
class Mod {
public:
    typedef unsigned int         InType;
    // The out-type is always unsigned int for these discriminators.
    typedef MyInt<N>             TupleType;
    static const InType padelm = N-1;
    CUDA_DEVICE_HOST static inline unsigned int apply(volatile InType x) {
        return x % N;
    }
};

#endif //HELPERS_COMMON
