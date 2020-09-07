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
#include <iomanip>                // set precision cout
#include <ratio>

namespace timer {

template<typename _ChronoPrecision>
void Timer_cuda::print(std::string str) {
    static_assert(is_duration<_ChronoPrecision>::value,
               PRINT_ERR("Wrong type : typename is not std::chrono::duration"));
    std::cout __ENABLE_COLOR(<< this->defaultColor)
              << std::right << std::setw(this->space - 2) << str << "  "
              << std::fixed << std::setprecision(this->decimals)
              << this->duration<_ChronoPrecision>()
              << _ChronoPrecision()
              __ENABLE_COLOR(<< xlib::Color::FG_DEFAULT)
              << std::endl;
}

template<typename _ChronoPrecision>
void Timer_cuda::getTime(std::string str) {
    static_assert(is_duration<_ChronoPrecision>::value,
               PRINT_ERR("Wrong type : typename is not std::chrono::duration"));
    this->stop();
    this->print(str);
}

template<typename _ChronoPrecision>
void Timer_cuda::getTimeA(std::string str) {
    this->getTime(str);
    std::cout << std::endl;
}

template<typename _ChronoPrecision>
float Timer_cuda::duration() {
    using float_milliseconds = typename std::chrono::duration<float,
                                                              std::milli>;
    static_assert(is_duration<_ChronoPrecision>::value,
                  "Wrong type : typename is not std::chrono::duration");
    return std::chrono::duration_cast<_ChronoPrecision>(
                float_milliseconds(time_elapsed)).count();
}

template<typename _ChronoPrecision>
void Timer_cuda::getTimeError(std::string str, const char* file, int line) {
    this->getTime<_ChronoPrecision>(str);
    cudaDeviceSynchronize();
    xlib::__getLastCudaError(str.c_str(), file, line);
}

} //@timer
