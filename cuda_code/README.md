# CUDA version of m-discriminator
There are two main versions of the program, a baseline version and an optimized version. The optimized version consists of the files

* MdiscrTest_optimized.cu
* MdiscrHost_optimized.cu.h
* MdiscrKernels_optimized.cu.h

These three files also come in a `_clean` variant (with most `#ifdef`'s and commented-out code etc. removed). It uses the datatypes and functions defined in `HelpersCommon.h`, `HelpersHost.cu.h`, `HelpersKernels.cu.h` and `MainCommon.h`.

The program expects several constants to be defined, e.g., on the command-line with `-D` option. The `_clean` variant needs these constants to be defined:

* NUM_CLASSES
* MAX_CHUNK
* MAP_X
* MAP_Y
* WRITE_X
* WRITE_Y

If the flag SPECIALIZED is defined, then we use the specialized tuple-types MyInt4 or MyInt8 (determined by NUM_CLASSES), otherwise we use the more general tuple-type MyInt<N> (with `N = NUM_CLASSES`). We can compile the program as follows:

```sh
nvcc -O3 -DNDEBUG -DSPECIALIZED -DNUM_CLASSES=8 -DMAX_CHUNK=96 -DMAP_X=64 -DMAP_Y=2 -DWRITE_X=32 -DWRITE_Y=8 -arch=sm_20 -o MdiscrTest_optimized_clean MdiscrTest_optimized_clean.cu
```

This compiles the program using the specialized MyInt8 tuples, with a MAX_CHUNK of 96, and with a block-size of `64 x 2` for the map-kernel and `32 x 8` for the write-kernel.

The non-cleaned optimized version further recognizes the `COMPACT_REPRESENTATION_MAP` and `COMPACT_REPRESENTATION_WRITE` flags. These make the program use the compact representation of the tuples in the map-kernel and the write-kernel, respectively. In the `_clean` variant of the program, it always uses the compact representation.

There are two more variants of the optimized version of the program: The `_tuples` and `_arrays` variants.

* In the `_tuples` variant, the input array conceptually consists of tuples of elements, but we are given a "tuple" of arrays (i.e., several arrays).
* In the `_arrays` variant, the input elements are themselves arrays, but the representation is flat, i.e., the elements are not delimitted. We know the inner dimension, and in order to get good memory accesses, we transpose the array so we basically end up with several arrays, just as in the case with the tuples.
