/*------------------------------------------------------------------------------
Copyright © 2016 by Nicola Bombieri

Mangrove is provided under the terms of The MIT License (MIT):

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
#include "Mining/Dictionary.hpp"

namespace mangrove {
/*
template<>
int generateDictionary<2>(dictionary_ptr_t<2> Dictionary, int vars, bool print){
    int K = 0;
    for (int i = 0; i < vars; i++) {
        for (int j = i + 1; j < vars; j++) {
            Dictionary[ K ][0] = static_cast<entry_t>(i);
            Dictionary[ K++ ][1] = static_cast<entry_t>(j);
            if (print)
                std::cout << i << '\t' << j << std::endl;
        }
    }
    return K;
}

template<>
int generateDictionary<3>(dictionary_ptr_t<3> Dictionary, int vars, bool print){
    int K = 0;
    for (int i = 0; i < vars; i++) {
        for (int j = 0; j < vars; j++) {
            if (j == i)
                continue;
            for (int x = j + 1; x < vars; x++) {
                if (x == i)
                    continue;
                Dictionary[ K ][0] = static_cast<entry_t>(i);
                Dictionary[ K ][1] = static_cast<entry_t>(j);
                Dictionary[ K++ ][2] = static_cast<entry_t>(x);
                if (print)
                    std::cout << i << '\t' << j << '\t' << x << std::endl;
            }
        }
    }
    return K;
}

template<>
int generateDictionary<4>(dictionary_ptr_t<4> Dictionary, int vars, bool print){
    int K = 0;
    for (int i = 0; i < vars; i++) {
        for (int j = 0; j < vars; j++) {
            if (j == i) continue;
            for (int x = j + 1; x < vars; x++) {
                if (x == i) continue;
                for (int y = x + 1; y < vars; y++) {
                    if (y == i) continue;
                    Dictionary[ K ][0] = static_cast<entry_t>(i);
                    Dictionary[ K ][1] = static_cast<entry_t>(j);
                    Dictionary[ K ][2] = static_cast<entry_t>(x);
                    Dictionary[ K++ ][3] = static_cast<entry_t>(y);
                    if (print) {
                        std::cout << i << '\t' << j << '\t' << x
                                  << "\t" << y << std::endl;
                    }
                }
            }
        }
    }
    return K;
}
*/
} //@Mangrove
