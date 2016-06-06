#ifndef SEQ_HELPERS
#define SEQ_HELPERS

#include<cassert>
#include<cstdio>
#include<cstring>

#include "HelpersCommon.h"

// template<class T>
// class Add {
// public:
//     typedef T BaseType;
//     static inline T identity() {
//         return T();
//     }
//     static inline T apply(const T t1, const T t2) {
//         return t1 + t2;
//     }
// };

// class MyInt4 {
// public:
//     unsigned int x; unsigned int y; unsigned int z; unsigned int w;
//     static const unsigned int cardinal = 4;
    
//     inline MyInt4()
//         : x(0), y(0), z(0), w(0) {
//         //x = 0; y = 0; z = 0; w = 0;
//     }
//     inline MyInt4(unsigned int init)
//         : x(init), y(init), z(init), w(init) {
//         //x = init; y = init; z = init; w = init;
//     }
//     inline MyInt4(const unsigned int& a,
//                   const unsigned int& b,
//                   const unsigned int& c,
//                   const unsigned int& d)
//         : x(a), y(b), z(c), w(d) {
//         //x = a; y = b; z = c; w = d;
//     }
//     inline MyInt4(const MyInt4& i4)
//         : x(i4.x), y(i4.y), z(i4.z), w(i4.w) {
//         //x = i4.x; y = i4.y; z = i4.z; w = i4.w;
//     }
//     inline MyInt4(const volatile MyInt4& i4)
//         : x(i4.x), y(i4.y), z(i4.z), w(i4.w) {
//         //x = i4.x; y = i4.y; z = i4.z; w = i4.w;
//     }
//     volatile inline MyInt4& operator=(const MyInt4& i4) volatile {
//         x = i4.x; y = i4.y; z = i4.z; w = i4.w;
//         return *this;
//     }
//     inline MyInt4& operator=(const MyInt4& i4) {
//         x = i4.x; y = i4.y; z = i4.z; w = i4.w;
//         return *this;
//     }
//     friend MyInt4 operator+(const MyInt4 &m1, const MyInt4 &m2) {
//         return MyInt4(m1.x+m2.x, m1.y+m2.y, m1.z+m2.z, m1.w+m2.w);
//     }
//     unsigned int& operator[](const unsigned int i) {
//         assert(i >= 0 && i <= 3);
//         if (i == 0) return x;
//         else if (i == 1) return y;
//         else if (i == 2) return z;
//         else return w; // i == 3
//     }
//     operator int * () {
//         return reinterpret_cast<int *>(this);
//     }
//     inline void set(const volatile MyInt4& i4) {
//         x = i4.x; y = i4.y; z = i4.z; w = i4.w;
//     }
// };

// class Mod4 {
// public:
//     typedef unsigned int           InType;
//     // The out-type is always unsigned int for these discriminators.
//     typedef MyInt4        TupleType;
//     static inline unsigned int apply(volatile unsigned int x) {
//         return x & 3;
//     }
// };

// template<unsigned int N>
// class MyInt {
// public:
//     unsigned int arr[N];
//     static const unsigned int cardinal = N;
    
//     unsigned int& operator[](const unsigned int i) {
//         assert(i >= 0 && i < N);
//         return arr[i];
//     }
//     inline MyInt() {
//         for (unsigned int i = 0; i < N; i++) {
//             arr[i] = 0;
//         }
//     }
//     // inline MyInt(unsigned int init) {
//     //     for (unsigned int i = 0; i < N; i++) {
//     //         arr[i] = init;
//     //     }
//     // }
//     inline MyInt(const MyInt<N>& other) {
//         for (unsigned int i = 0; i < N; i++) {
//             //arr[i] = other[i];
//             arr[i] = other.arr[i];
//         }
//     }
//     inline MyInt(const volatile MyInt<N>& other) {
//         for (unsigned int i = 0; i < N; i++) {
//             //arr[i] = other[i];
//             arr[i] = other.arr[i];
//         }
//     }
//     volatile inline MyInt<N>& operator=(const MyInt<N>& other) volatile {
//         for (unsigned int i = 0; i < N; i++) {
//             //arr[i] = other[i];
//             arr[i] = other.arr[i];
//         }
//         return *this;
//     }
//     inline MyInt<N>& operator=(const MyInt<N>& other) {
//         for (unsigned int i = 0; i < N; i++) {
//             //arr[i] = other[i];
//             arr[i] = other.arr[i];
//         }
//         return *this;
//     }
//     friend MyInt<N> operator+(const MyInt<N> &m1, const MyInt<N> &m2) {
//         MyInt<N> m;
//         for (unsigned int i = 0; i < N; i++) {
//             //m[i] = m1[i] + m2[i];
//             m.arr[i] = m1.arr[i] + m2.arr[i];
//         }
//         return m;
//     }
//     // operator int * () {
//     //     return reinterpret_cast<int *>(this);
//     // }
//     inline void set(const volatile MyInt<N>& other) {
//         for (int i = 0; i < N; i++) {
//             arr[i] = other[i];
//         }
//     }
// };

// template<unsigned int N>
// class Mod {
// public:
//     typedef unsigned int   InType;
//     // The out-type is always unsigned int for these discriminators.
//     typedef MyInt<N>      TupleType;
//     static inline unsigned int apply(volatile unsigned int x) {
//         return x % N;
//     }
// };

#endif //SEQ_HELPERS
