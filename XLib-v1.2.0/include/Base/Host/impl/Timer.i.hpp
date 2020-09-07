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
#include <stdexcept>
#include <iostream>
#include <iomanip>                // set precision cout
#include <chrono>
#include <ratio>
#include <type_traits>

namespace std {

template<class Rep, std::intmax_t Num, std::intmax_t Denom>
std::ostream& operator<<(std::ostream& os,
                         const std::chrono::duration
                            <Rep, std::ratio<Num, Denom>>&) {
    if (Num == 3600 && Denom == 1)        return os << " h";
    else if (Num == 60 && Denom == 1)    return os << " min";
    else if (Num == 1 && Denom == 1)    return os << " s";
    else if (Num == 1 && Denom == 1000)    return os << " ms";
    else return os << " Unsupported";
}

} //@std

namespace timer {

template<typename>
struct is_duration : std::false_type {};

template<typename T, typename R>
struct is_duration<std::chrono::duration<T, R>> : std::true_type {};

//-------------------------- GENERIC -------------------------------------------

template<timer_type type>
Timer<type>::Timer(std::ostream& _outStream, int _decimals) :
             outStream(_outStream), decimals(_decimals), time_elapsed(0) {}

#if defined(__COLOR)

template<timer_type type>
Timer<type>::Timer(int _decimals, int _space, xlib::Color _color) :
                    decimals(_decimals), space(_space), time_elapsed(0),
                   defaultColor(_color), startTime(), endTime(),
                   c_start(0), c_end(0)
#if defined(__linux__)
                 , startTMS(), endTMS()
#endif
                    {}

#else

template<timer_type type>
Timer<type>::Timer(int _decimals, int _space) :
                    decimals(_decimals), space(_space), time_elapsed(0)
#if defined(__linux__)
                 , startTMS(), endTMS()
#endif
                   {}

#endif

template<timer_type type>
Timer<type>::~Timer() {}

template<timer_type type>
template<typename _ChronoPrecision>
void Timer<type>::print(std::string str) {
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

template<timer_type type>
template<typename _ChronoPrecision>
void Timer<type>::getTime(std::string str) {
    static_assert(is_duration<_ChronoPrecision>::value,
               PRINT_ERR("Wrong type : typename is not std::chrono::duration"));
    this->stop();
    this->print(str);
}

template<timer_type type>
template<typename _ChronoPrecision>
void Timer<type>::getTimeA(std::string str) {
    getTime(str);
    std::cout << std::endl;
}

template<timer_type type>
template<typename _ChronoPrecision>
float Timer<type>::duration() {
    using float_seconds = std::chrono::duration<float>;
    static_assert(is_duration<_ChronoPrecision>::value,
                  "Wrong type : typename is not std::chrono::duration");
    return std::chrono::duration_cast<_ChronoPrecision>(
                float_seconds(time_elapsed)).count();
}

//==============================================================================

//-------------------------- SYS -----------------------------------------------
#if defined(__linux__)

template<>
template<typename _ChronoPrecision>
float Timer<SYS>::duration() {
    throw std::runtime_error( "Timer<SYS>::duration() is unsupported" );
}

template<>
template<typename _ChronoPrecision>
void Timer<SYS>::print(std::string str) {
    using float_seconds = std::chrono::duration<float>;
    static_assert(is_duration<_ChronoPrecision>::value,
               PRINT_ERR("Wrong type : typename is not std::chrono::duration"));
    auto wall_time = std::chrono::duration_cast<_ChronoPrecision>(
                                                  endTime - startTime ).count();
    auto user_time = std::chrono::duration_cast<_ChronoPrecision>(
                float_seconds(
                    static_cast<float>(endTMS.tms_utime - startTMS.tms_utime) /
                ::sysconf(_SC_CLK_TCK) ) ).count();
    auto sys_time = std::chrono::duration_cast<_ChronoPrecision>(
                float_seconds(
                    static_cast<float>(endTMS.tms_stime - startTMS.tms_stime) /
                ::sysconf(_SC_CLK_TCK) ) ).count();

    std::cout __ENABLE_COLOR(<< defaultColor)
              << std::setw(space) << str
              << "  Elapsed time: [user " << user_time << ", system "
              << sys_time << ", real "
              << wall_time << " " << _ChronoPrecision() << "]"
              __ENABLE_COLOR(<< xlib::Color::FG_DEFAULT)
              << std::endl;
}
#endif
} //@timer
