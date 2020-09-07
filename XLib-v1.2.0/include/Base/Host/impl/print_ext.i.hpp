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
#if defined(__NVCC__)
    #include <vector_types.h>    //int2
#endif
#include <type_traits>
#include <sstream>

namespace xlib {

template<class T, int SIZE>
void printArray(T (&Array)[SIZE], std::string text, char sep, T inf) {
    printArray(Array, SIZE, text, sep, inf);
}

template<class T>
void printArray(T* Array, int size, std::string text, char sep, T inf) {
    std::cout << text;
    for (int i = 0; i < size; i++) {
        if (Array[i] == inf)
            std::cout << "inf" << sep;
        else
            std::cout << Array[i] << sep;
    }
    std::cout << std::endl << std::endl;
}

template<class T>
void printMatrix(T** Matrix, int ROW, int COL, std::string text, T inf) {
    std::cout << text;
    for (int i = 0; i < ROW; i++)
        printArray(Matrix[i * COL], COL, "\n", true, inf, '\t');
    std::cout << std::endl << std::endl;
}

template<typename T>
__HOST_DEVICE__
typename std::enable_if<std::is_floating_point<T>::value>::type
_printArray(T* Array, int size) {
    for (int i = 0; i < size; i += 32) {
        for (int j = i; j < i + 32; j++)
           printf("%f ", Array[j]);
        printf("\n");
    }
    printf("\n");
}

template<typename T>
__HOST_DEVICE__
typename std::enable_if<std::is_integral<T>::value>::type
_printArray(T* Array, int size) {
    for (int i = 0; i < size; i += 32) {
        for (int j = i; j < i + 32; j++)
           printf("%d ", Array[j]);
        printf("\n");
    }
    printf("\n");
}

template<typename T>
__HOST_DEVICE__
typename std::enable_if<std::is_integral<T>::value>::type
printBits(T* Array, int size) {
    const int T_size = sizeof(T) * 8;
    for (int i = 0; i < size; i += T_size) {
        for (int j = i; j < i + static_cast<int>(T_size); j++)
           printf("%d", ( Array[j / T_size] &
                        (1 << (j % T_size)) ) ? 1 : 0 );
        printf(" ");
    }
    printf("\n");
}

#if defined(__NVCC__)

template<class T>
void printCudaArray(T* devArray, int size, std::string text, char sep, T inf) {
    T* hostArray = new T[size];
    cudaMemcpy(hostArray, devArray, size * sizeof(T), cudaMemcpyDeviceToHost);
    __CUDA_ERROR("Copy To Host");

    printArray(hostArray, size, text, sep, inf);
    delete[] hostArray;
}

#endif
} //@xlib
