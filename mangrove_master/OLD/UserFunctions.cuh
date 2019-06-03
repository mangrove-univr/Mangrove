#pragma once

template<typename T>
__HOST_DEVICE__ T add(T a, T b)		       { return a + b; }

template<typename T>
__HOST_DEVICE__ T minus(T a, T b)		   { return a - b; }

template<typename T>
__HOST_DEVICE__ T mul(T a, T b)			   { return a * b; }

template<typename T>
__HOST_DEVICE__ T divv(T a, T b)		   { return a / b; }

template<typename T>
__HOST_DEVICE__ T exp(T a, T b)        { return ::pow(a, b); }

template<typename T>
__HOST_DEVICE__ T min(T a, T b) { return (a <= b ? a : b); }

template<typename T>
__HOST_DEVICE__ T max(T a, T b) { return (a >= b ? a : b); }

template<typename T>
__HOST_DEVICE__ bool equal(T a, T b)	  { return a == b; }

template<typename T>
__HOST_DEVICE__ bool notEqualB(T a, T b)	 { return a == ~b; }

template<typename T>
__HOST_DEVICE__ bool notEqualN(T a, T b)	 { return a != b; }

template<typename T>
__HOST_DEVICE__ bool less(T a, T b)		   { return a < b; }

template<typename T>
__HOST_DEVICE__ bool lessEq(T a, T b)	  { return a <= b; }

template<typename T>
__HOST_DEVICE__ bool lessSqrt(T a, T b)
                                        { return a < ::sqrt(b); }
template<typename T>
__HOST_DEVICE__ bool equalLog(T a, T b)
                                        { return a == ::log2(b); }
template<typename T>
__HOST_DEVICE__ bool lessSucc(T a, T b)
                                         { return a < (b + 1); }
template<typename T>
__HOST_DEVICE__ bool equalTwice(T a, T b)
                                         { return a == (b * 2); }
template<typename T>
__HOST_DEVICE__ T root(T a)	             { return ::sqrt(a); }

template<typename T>
__HOST_DEVICE__ T log(T a)               { return ::log2(a); }

template<typename T>
__HOST_DEVICE__ T succ(T a)                { return a + 1; }

template<typename T>
__HOST_DEVICE__ T twice(T a)               { return a * 2; }

//------------------------------------------------------------------------------

template<typename T>
__HOST_DEVICE__ T AND(T a, T b)            { return a & b; }

template<typename T>
__HOST_DEVICE__ T NAND(T a, T b)        { return ~(a & b); }

template<typename T>
__HOST_DEVICE__ T OR(T a, T b)             { return a | b; }

template<typename T>
__HOST_DEVICE__ T NOR(T a, T b)         { return ~(a | b); }

template<typename T>
__HOST_DEVICE__ T XOR(T a, T b)            { return a ^ b; }

template<typename T>
__HOST_DEVICE__ T XNOR(T a, T b)        { return ~(a ^ b); }

template<typename T>
__HOST_DEVICE__ T IMPLY(T a, T b)       { return (~a) | b; }

template<typename T>
__HOST_DEVICE__ T R_IMPLY(T a, T b)     { return a | (~b); }

template<typename T>
__HOST_DEVICE__ T NOT_IMPLY(T a, T b)   { return a & (~b); }

template<typename T>
__HOST_DEVICE__ T NOT_R_IMPLY(T a, T b) { return (~a) & b; }
