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

#include <string>
#include <ostream>            // color
#include <unordered_map>
#include <cstdint>

#if !defined(__NVCC__)
    #define PRINT_ERR(ERR) "\n\n\033[91m--> "#ERR "\033[97m\n"
    #define PRINT_MSG(MSG) "\n\n\033[96m--> "#MSG "\033[97m\n"
#else
    #define PRINT_ERR(ERR) "\n\n--> "#ERR "\n"
    #define PRINT_MSG(MSG) "\n\n--> "#MSG "\n"
#endif

#define __ENABLE(VAL, EXPR) {                                                  \
    if (VAL)  {                                                                \
        EXPR                                                                   \
    }                                                                          \
}

#define __PRINT(msg)  {                                                        \
    std::cout << msg << std::endl;                                             \
}

#if !defined(__NVCC__)
#define __ERROR(msg)  {                                                        \
    std::cerr << std::endl << " ! ERROR : " << msg << std::endl << std::endl;  \
    std::exit(EXIT_FAILURE);                                                   \
}
#else
#define __ERROR(msg)  {                                                        \
    std::cerr << std::endl << " ! ERROR : " << msg << std::endl << std::endl;  \
    cudaDeviceReset();                                                         \
    std::exit(EXIT_FAILURE);                                                   \
}
#endif

#define __ERROR_LINE(msg)    {                                                  \
                std::cerr << std::endl << " ! ERROR : " << msg                 \
                          << " in " << __FILE__ << " : " << __func__           \
                          << " (line: " << __LINE__ << ")" << endl << endl;    \
                std::exit(EXIT_FAILURE);                                       \
            }

// =============================================================================

#if defined(_WIN32) || defined(__CYGWIN__)
    namespace std {
        template <typename T>
        std::string to_string(const T& n){
            std::ostringstream stm;
            stm << n;
            return stm.str() ;
        }
    }
#endif

namespace xlib {


using byte_t = uint8_t;

constexpr int operator""            _BIT ( unsigned long long int value );
constexpr std::size_t operator""     _KB ( unsigned long long int value );
constexpr std::size_t operator""     _MB ( unsigned long long int value );

const int ZERO = 0;

/**
 * @namespace StreamModifier provide modifiers and support methods for
 * std:ostream
 */
//namespace StreamModifier {
/**
 * @enum Color change the color of the output stream
 */
enum class Color {
                       /** <table border=0><tr><td><div> Red </div></td><td><div style="background:#FF0000;width:20px;height:20px;border:1px solid #000"></div></td></tr></table> */
    FG_RED       = 31, /** <table border=0><tr><td><div> Green </div></td><td><div style="background:#008000;width:20px;height:20px;border:1px solid #000"></div></td></tr></table> */
    FG_GREEN     = 32, /** <table border=0><tr><td><div> Yellow </div></td><td><div style="background:#FFFF00;width:20px;height:20px;border:1px solid #000"></div></td></tr></table> */
    FG_YELLOW    = 33, /** <table border=0><tr><td><div> Blue </div></td><td><div style="background:#0000FF;width:20px;height:20px;border:1px solid #000"></div></td></tr></table> */
    FG_BLUE      = 34, /** <table border=0><tr><td><div> Magenta </div></td><td><div style="background:#FF00FF;width:20px;height:20px;border:1px solid #000"></div></td></tr></table> */
    FG_MAGENTA   = 35, /** <table border=0><tr><td><div> Cyan </div></td><td><div style="background:#00FFFF;width:20px;height:20px;border:1px solid #000"></div></td></tr></table> */
    FG_CYAN      = 36, /** <table border=0><tr><td><div> Light Gray </div></td><td><div style="background:#D3D3D3;width:20px;height:20px;border:1px solid #000"></div></td></tr></table> */
    FG_L_GRAY    = 37, /** <table border=0><tr><td><div> Dark Gray </div></td><td><div style="background:#A9A9A9;width:20px;height:20px;border:1px solid #000"></div></td></tr></table> */
    FG_D_GREY    = 90, /** <table border=0><tr><td><div> Light Red </div></td><td><div style="background:#DC143C;width:20px;height:20px;border:1px solid #000"></div></td></tr></table> */
    FG_L_RED     = 91, /** <table border=0><tr><td><div> Light Green </div></td><td><div style="background:#90EE90;width:20px;height:20px;border:1px solid #000"></div></td></tr></table> */
    FG_L_GREEN   = 92, /** <table border=0><tr><td><div> Light Yellow </div></td><td><div style="background:#FFFFE0;width:20px;height:20px;border:1px solid #000"></div></td></tr></table> */
    FG_L_YELLOW  = 93, /** <table border=0><tr><td><div> Light Blue </div></td><td><div style="background:#ADD8E6;width:20px;height:20px;border:1px solid #000"></div></td></tr></table> */
    FG_L_BLUE    = 94, /** <table border=0><tr><td><div> Light Magenta </div></td><td><div style="background:#EE82EE;width:20px;height:20px;border:1px solid #000"></div></td></tr></table> */
    FG_L_MAGENTA = 95, /** <table border=0><tr><td><div> Light Cyan </div></td><td><div style="background:#E0FFFF;width:20px;height:20px;border:1px solid #000"></div></td></tr></table> */
    FG_L_CYAN    = 96, /** <table border=0><tr><td><div> White </div></td><td><div style="background:#FFFFFF;width:20px;height:20px;border:1px solid #000"></div></td></tr></table> */
    FG_WHITE     = 97, /** Default */
    FG_DEFAULT   = 39
};

/**
 * @enum Emph
 */
enum class Emph {
    SET_BOLD      = 1,
    SET_DIM       = 2,
    SET_UNDERLINE = 4,
    SET_RESET     = 0,
};

/// @cond
std::ostream& operator<<(std::ostream& os, const Color& mod);
std::ostream& operator<<(std::ostream& os, const Emph& mod);
//struct myseps;
/// @endcond

struct myseps : std::numpunct<char> {
private:
    char do_thousands_sep() const { return ','; }    // use space as separator
    std::string do_grouping() const { return "\3"; }    // digits are grouped by 3 digits each
};

class ThousandSep {
private:
    myseps* sep;
    ThousandSep(const ThousandSep&) = delete;
    void operator=(const ThousandSep&) = delete;
public:
    ThousandSep();
    ~ThousandSep();
};


void fixedFloat();
void scientificFloat();

template <typename>
struct get_arity;

template<typename T>
struct Type64;

template<class iterator_t>
struct Type64_it;

template <class T>
std::string type_name(T Obj);

//http://stackoverflow.com/posts/20170989/revisions
template <class T>
std::string type_name();

bool isDigit(std::string str);

template<class FUN_T, typename... T>
inline void Funtion_TO_multiThreads(bool MultiCore, FUN_T FUN, T... Args);

/**
 * @namespace return the old value if exits
 */
template<typename T, typename R = T>
class UniqueMap : public std::unordered_map<T, R> {
public:
    virtual ~UniqueMap();
    R insertValue(T id);
};

//------------------------------------------------------------------------------

void memInfoPrint(size_t total, size_t free, size_t Req);
void memInfoHost(size_t Req);

#if (__linux__)

#include <sys/resource.h>

class stackManagement {
private:
    struct rlimit ActualLimit;
public:
    stackManagement();
    ~stackManagement();
    void setLimit(std::size_t size = RLIM_INFINITY);
    void restore();
    void checkUnlimited();
};

void ctrlC_Handle();

#else

class stackManagement {
    stackManagement();
    ~stackManagement();
    void setLimit(std::size_t size = 0);
    void restore();
    void checkUnlimited();
};
#endif
//------------------------------------------------------------------------------

template<bool FAULT = true, class iteratorA_t, class iteratorB_t>
bool equal(iteratorA_t start_A, iteratorA_t end_A, iteratorB_t start_B);

template<bool FAULT = true, class iteratorA_t, class iteratorB_t>
bool equal(iteratorA_t start_A, iteratorA_t end_A, iteratorB_t start_B,
        bool (*equalFunction)(
                typename std::iterator_traits<iteratorA_t>::value_type,
                typename std::iterator_traits<iteratorB_t>::value_type));

template<bool FAULT = true, class iteratorA_t, class iteratorB_t>
bool equalSorted(iteratorA_t start_A, iteratorA_t end_A, iteratorB_t start_B);

} //@xlib

#include "impl/fUtil.i.hpp"
