/*------------------------------------------------------------------------------
Copyright © 2016 by Nicola Bombieri

XLib is provided under the terms of The MIT License (MIT):

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
the Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
------------------------------------------------------------------------------*/
/**
 * @author Federico Busato
 * Univerity of Verona, Dept. of Computer Science
 * federico.busato@univr.it
 */
#pragma once

#include "Base/Device/Util/PTX.cuh"

#define Load_MACRO(CACHE_MOD, ptx_modifier)                                    \
                                                                               \
template<>                                                                     \
__device__ __forceinline__                                                     \
ulonglong2 LoadSupport<CACHE_MOD, ulonglong2>(ulonglong2* pointer) {           \
    ulonglong2 ret;                                                            \
    asm("ld."#ptx_modifier".v2.u64 {%0, %1}, [%2];"                            \
        : "=l"(ret.x), "=l"(ret.y) : ASM_PTR(pointer));                        \
    return ret;                                                                \
}                                                                              \
                                                                               \
template<>                                                                     \
__device__ __forceinline__                                                     \
uint4 LoadSupport<CACHE_MOD, uint4>(uint4* pointer) {                          \
    uint4 ret;                                                                 \
    asm("ld."#ptx_modifier".v4.u32 {%0, %1, %2, %3}, [%4];"                    \
        : "=r"(ret.x), "=r"(ret.y), "=r"(ret.z), "=r"(ret.w)                   \
        : ASM_PTR(pointer));                                                   \
    return ret;                                                                \
}                                                                              \
                                                                               \
template<>                                                                     \
__device__ __forceinline__                                                     \
uint2 LoadSupport<CACHE_MOD, uint2>(uint2* pointer) {                          \
    uint2 ret;                                                                 \
    asm("ld."#ptx_modifier".v2.u32 {%0, %1}, [%2];"                            \
        : "=r"(ret.x), "=r"(ret.y) : ASM_PTR(pointer));                        \
    return ret;                                                                \
}                                                                              \
                                                                               \
template<>                                                                     \
__device__ __forceinline__                                                     \
ushort4 LoadSupport<CACHE_MOD, ushort4>(ushort4* pointer) {                    \
    ushort4 ret;                                                               \
    asm("ld."#ptx_modifier".v4.u16 {%0, %1, %2, %3}, [%4];"                    \
        : "=h"(ret.x), "=h"(ret.y), "=h"(ret.z), "=h"(ret.w)                   \
        : ASM_PTR(pointer));                                                   \
    return ret;                                                                \
}                                                                              \
                                                                               \
template<>                                                                     \
__device__ __forceinline__                                                     \
ushort2 LoadSupport<CACHE_MOD, ushort2>(ushort2* pointer) {                    \
    ushort2 ret;                                                               \
    asm("ld."#ptx_modifier".v2.u16 {%0, %1}, [%2];"                            \
        : "=h"(ret.x), "=h"(ret.y) : ASM_PTR(pointer));                        \
    return ret;                                                                \
}                                                                              \
                                                                               \
template<>                                                                     \
__device__ __forceinline__                                                     \
longlong2 LoadSupport<CACHE_MOD, longlong2>(longlong2* pointer) {              \
    longlong2 ret;                                                             \
    asm("ld."#ptx_modifier".v2.s64 {%0, %1}, [%2];"                            \
        : "=l"(ret.x), "=l"(ret.y) : ASM_PTR(pointer));                        \
    return ret;                                                                \
}                                                                              \
                                                                               \
template<>                                                                     \
__device__ __forceinline__                                                     \
int4 LoadSupport<CACHE_MOD, int4>(int4* pointer) {                             \
    int4 ret;                                                                  \
    asm("ld."#ptx_modifier".v4.s32 {%0, %1, %2, %3}, [%4];"                    \
        : "=r"(ret.x), "=r"(ret.y), "=r"(ret.z), "=r"(ret.w)                   \
        : ASM_PTR(pointer));                                                   \
    return ret;                                                                \
}                                                                              \
                                                                               \
template<>                                                                     \
__device__ __forceinline__                                                     \
int2 LoadSupport<CACHE_MOD, int2>(int2* pointer) {                             \
    int2 ret;                                                                  \
    asm("ld."#ptx_modifier".v2.s32 {%0, %1}, [%2];"                            \
        : "=r"(ret.x), "=r"(ret.y) : ASM_PTR(pointer));                        \
    return ret;                                                                \
}                                                                              \
                                                                               \
template<>                                                                     \
__device__ __forceinline__                                                     \
short4 LoadSupport<CACHE_MOD, short4>(short4* pointer) {                       \
    short4 ret;                                                                \
    asm("ld."#ptx_modifier".v4.s16 {%0, %1, %2, %3}, [%4];"                    \
        : "=h"(ret.x), "=h"(ret.y), "=h"(ret.z), "=h"(ret.w)                   \
        : ASM_PTR(pointer));                                                   \
    return ret;                                                                \
}                                                                              \
                                                                               \
template<>                                                                     \
__device__ __forceinline__                                                     \
short2 LoadSupport<CACHE_MOD, short2>(short2* pointer) {                       \
    short2 ret;                                                                \
    asm("ld."#ptx_modifier".v2.s16 {%0, %1}, [%2];"                            \
        : "=h"(ret.x), "=h"(ret.y) : ASM_PTR(pointer));                        \
    return ret;                                                                \
}                                                                              \
                                                                               \
template<>                                                                     \
__device__ __forceinline__                                                     \
long long unsigned LoadSupport<CACHE_MOD, long long unsigned>                  \
    (long long unsigned* pointer) {                                            \
                                                                               \
    long long unsigned ret;                                                    \
    asm("ld."#ptx_modifier".u64 %0, [%1];" : "=l"(ret) : ASM_PTR(pointer));    \
    return ret;                                                                \
}                                                                              \
                                                                               \
template<>                                                                     \
__device__ __forceinline__                                                     \
unsigned LoadSupport<CACHE_MOD, unsigned>(unsigned* pointer) {                 \
    unsigned ret;                                                              \
    asm("ld."#ptx_modifier".u32 %0, [%1];" : "=r"(ret) : ASM_PTR(pointer));    \
    return ret;                                                                \
}                                                                              \
                                                                               \
template<>                                                                     \
__device__ __forceinline__                                                     \
unsigned short LoadSupport<CACHE_MOD, unsigned short>                          \
    (unsigned short* pointer) {                                                \
                                                                               \
    unsigned short ret;                                                        \
    asm("ld."#ptx_modifier".u16 %0, [%1];" : "=h"(ret) : ASM_PTR(pointer));    \
    return ret;                                                                \
}                                                                              \
                                                                               \
template<>                                                                     \
__device__ __forceinline__                                                     \
unsigned char LoadSupport<CACHE_MOD, unsigned char>(unsigned char* pointer) {  \
    unsigned short ret;                                                        \
    asm ("ld."#ptx_modifier".u8 %0, [%1];" :"=h"(ret) : ASM_PTR(pointer));      \
    return static_cast<unsigned char>(ret);                                    \
}                                                                              \
                                                                               \
template<>                                                                     \
__device__ __forceinline__                                                     \
    uchar2 LoadSupport<CACHE_MOD, uchar2>(uchar2* pointer) {                   \
    unsigned short ret = LoadSupport<CACHE_MOD>(                               \
                                reinterpret_cast<unsigned short*>(pointer));   \
    return reinterpret_cast<uchar2&>(ret);                                     \
}                                                                              \
                                                                               \
template<>                                                                     \
__device__ __forceinline__                                                     \
uchar4 LoadSupport<CACHE_MOD, uchar4>(uchar4* pointer) {                       \
    unsigned ret = LoadSupport<CACHE_MOD>(                                     \
                                reinterpret_cast<unsigned*>(pointer));         \
    return reinterpret_cast<uchar4&>(ret);                                     \
}                                                                              \
                                                                               \
template<>                                                                     \
__device__ __forceinline__                                                     \
long long int LoadSupport<CACHE_MOD, long long int>(long long int* pointer) {  \
    long long int ret;                                                         \
    asm("ld."#ptx_modifier".s64 %0, [%1];" : "=l"(ret) : ASM_PTR(pointer));    \
    return ret;                                                                \
}                                                                              \
                                                                               \
template<>                                                                     \
__device__ __forceinline__                                                     \
int LoadSupport<CACHE_MOD, int>(int* pointer) {                                \
    int ret;                                                                   \
    asm("ld."#ptx_modifier".s32 %0, [%1];" : "=r"(ret) : ASM_PTR(pointer));    \
    return ret;                                                                \
}                                                                              \
                                                                               \
template<>                                                                     \
__device__ __forceinline__                                                     \
short LoadSupport<CACHE_MOD, short>(short* pointer) {                          \
    short ret;                                                                 \
    asm("ld."#ptx_modifier".s16 %0, [%1];" : "=h"(ret) : ASM_PTR(pointer));    \
    return ret;                                                                \
}                                                                              \
                                                                               \
template<>                                                                     \
__device__ __forceinline__                                                     \
char LoadSupport<CACHE_MOD, char>(char* pointer) {                             \
    short ret;                                                                 \
    asm("ld."#ptx_modifier".s8 %0, [%1];" : "=h"(ret) : ASM_PTR(pointer));     \
    return static_cast<char>(ret);                                             \
}                                                                              \
                                                                               \
template<>                                                                     \
__device__ __forceinline__                                                     \
    char2 LoadSupport<CACHE_MOD, char2>(char2* pointer) {                      \
    short ret = LoadSupport<CACHE_MOD>(reinterpret_cast<short*>(pointer));     \
    return reinterpret_cast<char2&>(ret);                                      \
}                                                                              \
                                                                               \
template<>                                                                     \
__device__ __forceinline__                                                     \
char4 LoadSupport<CACHE_MOD, char4>(char4* pointer) {                          \
    int ret = LoadSupport<CACHE_MOD>(reinterpret_cast<int*>(pointer));         \
    return reinterpret_cast<char4&>(ret);                                      \
}                                                                              \
                                                                               \
template<>                                                                     \
__device__ __forceinline__                                                     \
double2 LoadSupport<CACHE_MOD, double2>(double2* pointer) {                    \
    double2 ret;                                                               \
    asm("ld."#ptx_modifier".v2.f64 {%0, %1}, [%2];"                            \
        : "=d"(ret.x), "=d"(ret.y) : "l"(pointer));                            \
    return ret;                                                                \
}                                                                              \
                                                                               \
template<>                                                                     \
__device__ __forceinline__                                                     \
float4 LoadSupport<CACHE_MOD, float4>(float4* pointer) {                       \
    float4 ret;                                                                \
    asm("ld."#ptx_modifier".v4.f32 {%0, %1, %2, %3}, [%4];"                    \
        : "=f"(ret.x), "=f"(ret.y), "=f"(ret.z), "=f"(ret.w)                   \
        : "l"(pointer));                                                       \
    return ret;                                                                \
}                                                                              \
                                                                               \
template<>                                                                     \
__device__ __forceinline__                                                     \
float2 LoadSupport<CACHE_MOD, float2>(float2* pointer) {                       \
    float2 ret;                                                                \
    asm("ld."#ptx_modifier".v2.f32 {%0, %1}, [%2];"                            \
        : "=f"(ret.x), "=f"(ret.y) : "l"(pointer));                            \
    return ret;                                                                \
}                                                                              \
                                                                               \
template<>                                                                     \
__device__ __forceinline__                                                     \
float LoadSupport<CACHE_MOD, float>(float* pointer) {                          \
    float ret;                                                                 \
    asm("ld."#ptx_modifier".f32 %0, [%1];" : "=f"(ret) : "l"(pointer));        \
    return ret;                                                                \
}                                                                              \
                                                                               \
template<>                                                                     \
__device__ __forceinline__                                                     \
double LoadSupport<CACHE_MOD, double>(double* pointer) {                       \
    double ret;                                                                \
    asm("ld."#ptx_modifier".f64 %0, [%1];" : "=d"(ret) : "l"(pointer));        \
    return ret;                                                                \
}

//==============================================================================

enum CacheLoadModifier { LOAD, LOAD_CA, LOAD_CG, LOAD_CS, LOAD_CV,
                         LOADNC, LOADNC_CA, LOADNC_CG, LOADNC_CS };

template<CacheLoadModifier M, typename T>
__device__ __forceinline__  T LoadSupport(T* pointer);

template<CacheLoadModifier MODIFIER = LOAD>
struct ThreadLoad {
    template<typename T>
    static __device__ __forceinline__ T OP(T* pointer) {
        printf("Thread Load Not Implemented");
    }
};

template<>
struct ThreadLoad<LOAD> {
    template<typename T>
    static __device__ __forceinline__ T OP(T* pointer) {
        return *pointer;
    }
    template<typename T>
    static __device__ __forceinline__ T OP(volatile T* pointer) {
        return *pointer;
    }
    template<typename T>
    static __device__ __forceinline__ T OP(const T* pointer) {
        return *pointer;
    }
    template<typename T>
    static __device__ __forceinline__ T OP(const volatile T* pointer) {
        return *pointer;
    }
};

template<>
struct ThreadLoad<LOADNC> {
    template<typename T>
    static __device__ __forceinline__ T OP(T* pointer) {
        return __ldg(pointer);
    }
    template<typename T>
    static __device__ __forceinline__ T OP(volatile T* pointer) {
        return __ldg(pointer);
    }
    template<typename T>
    static __device__ __forceinline__ T OP(const T* pointer) {
        return __ldg(pointer);
    }
    template<typename T>
    static __device__ __forceinline__ T OP(const volatile T* pointer) {
        return __ldg(pointer);
    }
};

#define LoadStruct_MACRO(CACHE_MOD)                                            \
                                                                               \
template<>                                                                     \
struct ThreadLoad<CACHE_MOD> {                                                 \
    template<typename T>                                                       \
    static __device__ __forceinline__ T OP(T* pointer) {                       \
        return LoadSupport<CACHE_MOD>(                                         \
               const_cast<typename std::remove_cv<T>::type*>(pointer));        \
    }                                                                          \
};

LoadStruct_MACRO(LOAD_CA)
LoadStruct_MACRO(LOAD_CG)
LoadStruct_MACRO(LOAD_CS)
LoadStruct_MACRO(LOAD_CV)
LoadStruct_MACRO(LOADNC_CA)
LoadStruct_MACRO(LOADNC_CG)
LoadStruct_MACRO(LOADNC_CS)

Load_MACRO(LOAD_CA, ca)
Load_MACRO(LOAD_CG, cg)
Load_MACRO(LOAD_CS, cs)
Load_MACRO(LOAD_CV, volatile)
Load_MACRO(LOADNC_CA, ca.nc)
Load_MACRO(LOADNC_CG, cg.nc)
Load_MACRO(LOADNC_CS, cs.nc)

#undef LoadStruct_MACRO
#undef Load_MACRO
