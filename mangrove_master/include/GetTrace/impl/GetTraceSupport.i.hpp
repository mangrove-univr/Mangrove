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
#include "XLib.hpp"

namespace mangrove {

inline float textToFloat(char*& input) {
    using namespace xlib;
    using namespace std;
    bool neg = false;
    if (*input == '-') {
        neg = true;
        ++input;
    }

    int len = 0;
    while (input[len] >= '0' && input[len] <= '9')
        ++len;

    int value = fastStringToInt(input, len);
    float result = static_cast<float>(value);

    input += len;
    if (*input == '.') {
        ++input;
        len = 0;
        while (input[len] >= '0' && input[len] <= '9')
            ++len;

        int len2 = min(len, 9);
        const float int_frac = static_cast<float>(fastStringToInt(input, len2));

        switch (len2) {
            case 1: result += int_frac *  (1 / 10.0f); break;
            case 2: result += int_frac *  (1 / 100.0f); break;
            case 3: result += int_frac *  (1 / 1000.0f); break;
            case 4: result += int_frac *  (1 / 10000.0f); break;
            case 5: result += int_frac *  (1 / 100000.0f); break;
            case 6: result += int_frac *  (1 / 1000000.0f); break;
            case 7: result += int_frac *  (1 / 10000000.0f); break;
            case 8: result += int_frac *  (1 / 100000000.0f); break;
            case 9: result += int_frac *  (1 / 1000000000.0f); break;
            default:;
        }
        input += len;
    }
    input++;
    return neg ? -result : result;
}

//------------------------------------------------------------------------------

template<unsigned SIZE, unsigned K>
struct textToBoolean {
    static inline void apply(const char* input, unsigned* ptrArray) {
        if (input[K] == '1')
            ptrArray[0] |= 1u << (K / 2u);
        textToBoolean<SIZE, K + 2>::apply(input, ptrArray);
    }
};
template<unsigned SIZE>
struct textToBoolean<SIZE, SIZE> {
    static inline void apply(const char*, unsigned*) {}
};

} //@mangrove
