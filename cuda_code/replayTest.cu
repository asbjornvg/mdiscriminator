#include <cuda_runtime.h>
#include<cassert>
#include<sys/time.h>
#include<time.h>
#include<stdio.h>
#include<string>
#include<sstream>

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
       fprintf(stderr,"GPUassert (%d): %s (%s, line %d)\n", code, cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

int timeval_subtract(struct timeval *result, struct timeval *t2, struct timeval *t1)
{
    unsigned int resolution=1000000;
    long int diff = (t2->tv_usec + resolution * t2->tv_sec) - (t1->tv_usec + resolution * t1->tv_sec);
    result->tv_sec = diff / resolution;
    result->tv_usec = diff % resolution;
    return (diff<0);
}

class MyInt8Tuple {
    unsigned int x0; unsigned int x1; unsigned int x2; unsigned int x3;
    unsigned int x4; unsigned int x5; unsigned int x6; unsigned int x7;
    
public:
    
    __device__ __host__ inline MyInt8Tuple()
        : x0(0), x1(0), x2(0), x3(0), x4(0), x5(0), x6(0), x7(0) {
    }
    
    __device__ __host__ inline unsigned int& operator[](const unsigned int i) {
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
};

class MyInt8Array {
    unsigned int arr[8];
    
public:
    
    __device__ __host__ inline MyInt8Array()
        : arr() {
    }
    
    __device__ __host__ unsigned int& operator[](const unsigned int i) {
        assert(i < 8);
        return arr[i];
    }
};

void populateIntArray(const unsigned int  num_elems,
                      unsigned int*       in_array
    ) {
    for(unsigned int i = 0; i < num_elems; i++) {
        in_array[i] = std::rand() % 20;
    }
}

template<typename T>
void printIntCollection(unsigned int length, std::string title, T collection) {
    fprintf(stderr, "%-12s [", (title + ":").c_str());
    bool first = true;
    for(unsigned int i = 0; i < length; i++) {
        if (first) {
            fprintf(stderr, "%2d", collection[i]);
            first = false;
        }
        else {
            fprintf(stderr, ", %2d", collection[i]);
        }
    }
    fprintf(stderr, "]\n");
}

#define NUM_ITERATIONS 50

template<typename IntType>
__global__ void testKernel(
    // IntType*            in_array,
    IntType*            out_array,
    unsigned int*       indices,
    const unsigned int  num_elems
    ) {
    const unsigned int gid = blockIdx.x*blockDim.x + threadIdx.x;
    if (gid < num_elems) {
        const unsigned int lane8 = threadIdx.x & 7;
        const unsigned int index = indices[lane8];
        
        IntType elem;
        for (unsigned int i = 0; i < NUM_ITERATIONS; i++) {
            elem[index]++;
        }
        out_array[gid] = elem;
    }
}

#define BLOCK_SIZE 512

template<typename IntType>
void test(const unsigned int num_elems,
          const unsigned int i0,
          const unsigned int i1,
          const unsigned int i2,
          const unsigned int i3,
          const unsigned int i4,
          const unsigned int i5,
          const unsigned int i6,
          const unsigned int i7
    ) {
    // fprintf(stderr, "sizeof(IntType) = %d\n", sizeof(IntType));
    
    struct timeval t_start, t_end, t_diff;
    unsigned long int elapsed;
    
    // IntType* in_array = (IntType*) malloc(num_elems * sizeof(IntType));
    IntType* out_array = (IntType*) malloc(num_elems * sizeof(IntType));
    
    // memset(in_array, 0, num_elems * sizeof(IntType));
    
    unsigned int indices[8] = {i0,i1,i2,i3,i4,i5,i6,i7};
    
    // IntType *in_array_d;
    IntType *out_array_d;
    unsigned int *indices_d;
    // cudaMalloc((void**)&in_array_d, num_elems * sizeof(IntType));
    gpuErrchk( cudaMalloc((void**)&out_array_d, num_elems * sizeof(IntType)) );
    gpuErrchk( cudaMalloc((void**)&indices_d, 8 * sizeof(unsigned int)) );
    
    // cudaMemcpy(in_array_d, in_array, num_elems * sizeof(IntType), cudaMemcpyHostToDevice);
    gpuErrchk( cudaMemset(out_array_d, 0, num_elems * sizeof(IntType)) );
    gpuErrchk( cudaMemcpy(indices_d, indices, 8 * sizeof(unsigned int), cudaMemcpyHostToDevice) );
    gpuErrchk( cudaThreadSynchronize() );
    
    const unsigned int num_blocks = (num_elems + BLOCK_SIZE - 1) / BLOCK_SIZE;
    // fprintf(stderr, "num_blocks = %d\n", num_blocks);
    
    gettimeofday(&t_start, NULL);
    
    testKernel<IntType><<<num_blocks, BLOCK_SIZE>>>(out_array_d, indices_d, num_elems);
    
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaThreadSynchronize() );
    gettimeofday(&t_end, NULL);
    
    gpuErrchk( cudaMemcpy(out_array, out_array_d, num_elems * sizeof(IntType), cudaMemcpyDeviceToHost) );
    
    // cudaFree(in_array_d);
    gpuErrchk( cudaFree(out_array_d) );
    gpuErrchk( cudaFree(indices_d) );
    
    timeval_subtract(&t_diff, &t_end, &t_start);
    elapsed = (t_diff.tv_sec*1e6+t_diff.tv_usec);
    fprintf(stderr, "Total runtime:  %6lu microsecs\n", elapsed);
    
    std::srand(time(NULL));
    unsigned int index = std::rand() % num_elems;
    std::stringstream sstm;
    sstm << "Random element (" << index << ")";
    std::string s = sstm.str();
#ifdef PRINT
    printIntCollection(8, s, out_array[index]);
#endif
    
    // free(in_array);
    free(out_array);
}

int main(int argc, char** argv) {
    if (argc != 10) {
        printf("The program takes <num_elems> and eight indices as arguments!\n");
        return EXIT_FAILURE;
    }
    const unsigned int num_elems = strtoul(argv[1], NULL, 10);
    const unsigned int i0 = strtoul(argv[2], NULL, 10);
    const unsigned int i1 = strtoul(argv[3], NULL, 10);
    const unsigned int i2 = strtoul(argv[4], NULL, 10);
    const unsigned int i3 = strtoul(argv[5], NULL, 10);
    const unsigned int i4 = strtoul(argv[6], NULL, 10);
    const unsigned int i5 = strtoul(argv[7], NULL, 10);
    const unsigned int i6 = strtoul(argv[8], NULL, 10);
    const unsigned int i7 = strtoul(argv[9], NULL, 10);
    
    fprintf(stderr, "MyInt8Tuple:\n");
    test<MyInt8Tuple>(num_elems,i0,i1,i2,i3,i4,i5,i6,i7);
    fprintf(stderr, "MyInt8Array:\n");
    test<MyInt8Array>(num_elems,i0,i1,i2,i3,i4,i5,i6,i7);
    
    return EXIT_SUCCESS;
}
