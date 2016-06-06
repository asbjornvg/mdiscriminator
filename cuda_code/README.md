# CUDA version of m-discriminator
There are two main versions of the program, a baseline version and an optimized version. The optimized version consists of the files

* MdiscrTest_optimized.cu
* MdiscrHost_optimized.cu.h
* MdiscrKernels_optimized.cu.h

These three files also come in a `_clean` variant (with most `#ifdef`'s etc. removed). Both versions use the datatypes and functions defined in `HelpersCommon.h`, `HelpersHost.cu.h`, `HelpersKernels.cu.h` and `MainCommon.h`.

The program expects several constants to be defined, e.g., on the command-line with `-D` option. The `_clean` variant needs these constants to be defined:

* NUM_CLASSES
* MAX_CHUNK
* MAP_X
* MAP_Y
* WRITE_X
* WRITE_Y

If the flag SPECIALIZED is defined, then we use specialized tuple-types (MyInt4 or MyInt8), otherwise we use the more general tuple-type MyInt<N> (with `N = NUM_CLASSES`).

The non-cleaned optimized version further recognizes the `COMPACT_REPRESENTATION_MAP` and `COMPACT_REPRESENTATION_WRITE` constants.

In addition to this, there are `_tuples` and `_arrays` variants of the code. Both these variants build upon the optimized version of the program.

##### Tuples
Here, the input array conceptually consists of tuples of elements, but we are given a "tuple" of arrays (i.e., several arrays).

##### Arrays
Here, the input elements are themselves arrays, but the representation is flat, i.e., the elements are not delimitted. We know the inner dimension, and in order to get good memory accesses, we transpose the array so we basically end up with several arrays, just as in the case with the tuples.
