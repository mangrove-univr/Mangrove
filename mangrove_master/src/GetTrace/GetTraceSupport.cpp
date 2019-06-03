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
#include <fstream>
#include "GetTrace/GetTraceSupport.hpp"
#include "XLib.hpp"

using namespace xlib;

namespace mangrove {

void booleanFilling(unsigned* host_trace, int sim_instants, int trace_length) {
    const unsigned last_index = static_cast<unsigned>(sim_instants - 1) / 32u;

    if (readBit(host_trace, sim_instants - 1)) {
        host_trace[last_index] |=
               ~( (1u << (static_cast<unsigned>(sim_instants - 1) % 32u)) - 1u);

        std::fill(host_trace + last_index + 1,
                  host_trace + trace_length, 0xFFFFFFFF);
    }
    else {
        host_trace[last_index] &=
                   (1u << (static_cast<unsigned>(sim_instants - 1) % 32u)) - 1u;
        std::fill(host_trace + last_index + 1,
                  host_trace + trace_length, 0x00000000);
    }
}


void textToBooleanDyn(const char* input, unsigned* ptrArray, unsigned size) {
    const unsigned n_of_elements = size / 2u;
    switch (n_of_elements) {
        case 0: break;
        case 1: textToBoolean<2>::apply(input, ptrArray); break;
        case 2: textToBoolean<4>::apply(input, ptrArray); break;
        case 3: textToBoolean<6>::apply(input, ptrArray); break;
        case 4: textToBoolean<8>::apply(input, ptrArray); break;
        case 5: textToBoolean<10>::apply(input, ptrArray); break;
        case 6: textToBoolean<12>::apply(input, ptrArray); break;
        case 7: textToBoolean<14>::apply(input, ptrArray); break;
        case 8: textToBoolean<16>::apply(input, ptrArray); break;
        case 9: textToBoolean<18>::apply(input, ptrArray); break;
        case 10: textToBoolean<20>::apply(input, ptrArray); break;
        case 11: textToBoolean<22>::apply(input, ptrArray); break;
        case 12: textToBoolean<24>::apply(input, ptrArray); break;
        case 13: textToBoolean<26>::apply(input, ptrArray); break;
        case 14: textToBoolean<28>::apply(input, ptrArray); break;
        case 15: textToBoolean<30>::apply(input, ptrArray); break;
        case 16: textToBoolean<32>::apply(input, ptrArray); break;
        case 17: textToBoolean<34>::apply(input, ptrArray); break;
        case 18: textToBoolean<36>::apply(input, ptrArray); break;
        case 19: textToBoolean<38>::apply(input, ptrArray); break;
        case 20: textToBoolean<40>::apply(input, ptrArray); break;
        case 21: textToBoolean<42>::apply(input, ptrArray); break;
        case 22: textToBoolean<44>::apply(input, ptrArray); break;
        case 23: textToBoolean<46>::apply(input, ptrArray); break;
        case 24: textToBoolean<48>::apply(input, ptrArray); break;
        case 25: textToBoolean<50>::apply(input, ptrArray); break;
        case 26: textToBoolean<52>::apply(input, ptrArray); break;
        case 27: textToBoolean<54>::apply(input, ptrArray); break;
        case 28: textToBoolean<56>::apply(input, ptrArray); break;
        case 29: textToBoolean<58>::apply(input, ptrArray); break;
        case 30: textToBoolean<60>::apply(input, ptrArray); break;
        case 31: textToBoolean<62>::apply(input, ptrArray); break;
        default:;
    }
}

} //@mangrove
