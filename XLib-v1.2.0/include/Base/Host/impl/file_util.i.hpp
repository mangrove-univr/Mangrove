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
#include "Base/Host/fUtil.hpp"

namespace xlib {

template<typename, typename, typename... Ts>
void Batch::writeBinary(void*) {}

template<typename T1, typename T2, typename... Ts>
void Batch::writeBinary(void* pointer, T1* Data, T2 size, Ts... args) {
    static_assert(std::is_integral<T2>::value,
                  PRINT_ERR("writeBinary: not integral type"));
    std::copy(Data, Data + size, (T1*) pointer);
    partial += size * sizeof(T1);
    progress.perCent(partial);

    writeBinary((T1*) pointer + size, args...);
}

template<typename, typename, typename... Ts>
void Batch::readBinary(void*) {}

template<typename T1, typename T2, typename... Ts>
void Batch::readBinary(void* pointer, T1* Data, T2 size, Ts... args) {
    static_assert(std::is_integral<T2>::value,
                  PRINT_ERR("readBinary: not integral type"));
    std::copy((T1*) pointer, (T1*) pointer + size, Data);
    partial += static_cast<std::size_t>(size * sizeof(T1));
    progress.perCent(partial);

    readBinary(static_cast<T1*>(pointer) + static_cast<std::size_t>(size),
               args...);
}

} //@xlib
