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
#include "Base/Host/print_ext.hpp"

namespace xlib {

template<>
__HOST_DEVICE__
void _printArray<char>(char* Array, int size) {
    for (int i = 0; i < size; i += 32) {
        for (int j = i; j < i + 32; j++)
           printf("%c", Array[j]);
        printf(" ");
    }
     printf("\n");
}

template<>
__HOST_DEVICE__
void _printArray<unsigned char>(unsigned char* Array, int size) {
    for (int i = 0; i < size; i += 32) {
        for (int j = i; j < i + 32; j++)
           printf("%c", Array[j]);
        printf(" ");
    }
     printf("\n");
}

template<>
void printArray<char>(char* Array, int size, std::string text,
                      char sep, char inf) {
    std::cout << text;
    for (int i = 0; i < size; i++) {
        if (Array[i] == inf)
            std::cout << "inf" << sep;
        else
            std::cout << static_cast<int>(Array[i]) << sep;
    }
    std::cout << std::endl << std::endl;
}

template<>
void printArray<unsigned char>(unsigned char* Array, int size, std::string text,
                               char sep, unsigned char inf) {
    std::cout << text;
    for (int i = 0; i < size; i++) {
        if (Array[i] == inf)
            std::cout << "inf" << sep;
        else
            std::cout << static_cast<unsigned>(Array[i]) << sep;
    }
    std::cout << std::endl << std::endl;
}

#if defined(__NVCC__)

template<>
void printArray<int2>(int2* Array, int size, std::string text,
                      char sep, int2 inf) {
    std::cout << text;
    for (int i = 0; i < size; i++) {
        if (Array[i].x == inf.x)
            std::cout << "inf" << sep;
        else
            std::cout << Array[i].x << sep;
    }
    std::cout << std::endl;
    for (int i = 0; i < size; i++) {
        if (Array[i].y == inf.y)
            std::cout << "inf" << sep;
        else
            std::cout << Array[i].y << sep;
    }
    std::cout << std::endl << std::endl;
}

template<>
void printArray<int3>(int3* Array, int size, std::string text,
                      char sep, int3 inf) {
    std::cout << text;
    for (int i = 0; i < size; i++) {
        if (Array[i].x == inf.x)
            std::cout << "inf" << sep;
        else
            std::cout << Array[i].x << sep;
    }
    std::cout << std::endl;
    for (int i = 0; i < size; i++) {
        if (Array[i].y == inf.y)
            std::cout << "inf" << sep;
        else
            std::cout << Array[i].y << sep;
    }
    for (int i = 0; i < size; i++) {
        if (Array[i].z == inf.z)
            std::cout << "inf" << sep;
        else
            std::cout << Array[i].z << sep;
    }
    std::cout << std::endl << std::endl;
}

template<>
void printArray<int4>(int4* Array, int size, std::string text,
                      char sep, int4 inf) {
    std::cout << text;
    for (int i = 0; i < size; i++) {
        if (Array[i].x == inf.x)
            std::cout << "inf" << sep;
        else
            std::cout << Array[i].x << sep;
    }
    std::cout << std::endl;
    for (int i = 0; i < size; i++) {
        if (Array[i].y == inf.y)
            std::cout << "inf" << sep;
        else
            std::cout << Array[i].y << sep;
    }
    for (int i = 0; i < size; i++) {
        if (Array[i].z == inf.z)
            std::cout << "inf" << sep;
        else
            std::cout << Array[i].z << sep;
    }
    for (int i = 0; i < size; i++) {
        if (Array[i].w == inf.w)
            std::cout << "inf" << sep;
        else
            std::cout << Array[i].w << sep;
    }
    std::cout << std::endl << std::endl;
}

#endif
} //@printExt
