#pragma once
#include <algorithm>

namespace mangrove {

template<typename T>
struct add {
    __HOST_DEVICE__ T operator()(T a, T b)    { return a + b; }
};

template<typename T>
struct Min {
    __HOST_DEVICE__ T operator()(T a, T b)    {
        using namespace std;
        return min(a, b);
    }
};

template<typename T>
struct Max {
    __HOST_DEVICE__ T operator()(T a, T b)    {
        using namespace std;
        return max(a, b);
    }
};

template<typename T>
struct mul {
    __HOST_DEVICE__ T operator()(T a, T b)    { return a * b; }
};

//------------------------------------------------------------------------------
// BINARY BITWISE

template<typename T>
struct AND {
    __HOST_DEVICE__ T operator()(T a, T b)    { return a & b; }
};

template<typename T>
struct NAND {
    __HOST_DEVICE__ T operator()(T a, T b)    { return ~(a & b); }
};

template<typename T>
struct OR {
    __HOST_DEVICE__ T operator()(T a, T b)    { return a | b; }
};

template<typename T>
struct NOR {
    __HOST_DEVICE__ T operator()(T a, T b)    { return ~(a | b); }
};

template<typename T>
struct XOR {
    __HOST_DEVICE__ T operator()(T a, T b)    { return a ^ b; }
};

template<typename T>
struct XNOR {
    __HOST_DEVICE__ T operator()(T a, T b)    { return ~(a ^ b); }
};

template<typename T>
struct IMPLY {
    __HOST_DEVICE__ T operator()(T a, T b)    { return (~a) | b; }
};

template<typename T>
struct R_IMPLY {
    __HOST_DEVICE__ T operator()(T a, T b)    { return a | (~b); }
};

template<typename T>
struct NOT_IMPLY {
    __HOST_DEVICE__ T operator()(T a, T b)    { return a & (~b); }
};

template<typename T>
struct NOT_R_IMPLY {
    __HOST_DEVICE__ T operator()(T a, T b)    { return (~a) & b; }
};

//==============================================================================
// UNARY FUNCTION

template<typename T>
struct identity {
    __HOST_DEVICE__ T operator()(T a)        { return a; }
};

template<typename T>
struct succ {
    __HOST_DEVICE__ T operator()(T a)        { return a + 1; }
};

template<typename T>
struct twice {
    __HOST_DEVICE__ T operator()(T a)        { return a * 2; }
};

template<typename T>
struct Sqrt {
    __HOST_DEVICE__ T operator()(T a)    { return std::sqrt(a); }
};

template<typename T>
struct Log {
    __HOST_DEVICE__ T operator()(T a)    { return std::log(a); }
};

template<typename T>
struct Exp {
    __HOST_DEVICE__ T operator()(T a)    { return std::exp(a); }
};

//==============================================================================
// COMPARE

template<typename T>
struct equal {
    __HOST_DEVICE__ bool operator()(T a, T b)	  { return a == b; }
};

template<typename T>
struct complement {
    __HOST_DEVICE__ bool operator()(T a, T b)	  { return a == ~b; }
};

template<typename T>
struct notEqual {
    __HOST_DEVICE__ bool operator()(T a, T b)	  { return a != b; }
};

template<typename T>
struct less {
    __HOST_DEVICE__ bool operator()(T a, T b)	  { return a < b; }
};

template<typename T>
struct lessEq {
    __HOST_DEVICE__ bool operator()(T a, T b)	  { return a <= b; }
};

} //@mangrove
