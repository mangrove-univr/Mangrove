/*------------------------------------------------------------------------------
Copyright © 2015 by Nicola Bombieri

H-BF is provided under the terms of The MIT License (MIT):

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
#include "Base/Host/Timer.hpp"

namespace timer {

//-------------------------- HOST ----------------------------------------------

template<>
void Timer<HOST>::start() {
    startTime = std::chrono::system_clock::now();
}

template<>
void Timer<HOST>::stop() {
    endTime = std::chrono::system_clock::now();
    time_elapsed = std::chrono::duration<float>(endTime - startTime).count();
}

//-------------------------- CPU -----------------------------------------------

template<>
void Timer<CPU>::start() {
    c_start = std::clock();
}

template<>
void Timer<CPU>::stop() {
    c_end = std::clock();
    time_elapsed = static_cast<float>(c_end - c_start) / CLOCKS_PER_SEC;
}

//-------------------------- SYS -----------------------------------------------

#if defined(__linux__)

template<>
void Timer<SYS>::start() {
    startTime = std::chrono::system_clock::now();
    ::times(&startTMS);
}

template<>
void Timer<SYS>::stop() {
    endTime = std::chrono::system_clock::now();
    ::times(&endTMS);
}
#endif

} //@timer
