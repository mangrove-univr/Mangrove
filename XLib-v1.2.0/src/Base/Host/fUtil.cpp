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
 #include <iostream>
#include <fstream>
#include <locale>
#include <iomanip>
#include <stdexcept>
#include "Base/Host/fUtil.hpp"
#include "Base/Host/numeric.hpp"

#if __linux__
    #include <unistd.h>
#endif

namespace xlib {
/// @cond

std::ostream& operator<<(std::ostream& os, const Color& mod) {
    return os << "\033[" << (int) mod << "m";
}

std::ostream& operator<<(std::ostream& os, const Emph& mod) {
    return os << "\033[" << (int) mod << "m";
}

ThousandSep::ThousandSep() : sep(NULL) {
    sep = new myseps;
    std::cout.imbue(std::locale(std::locale(), sep));
}

ThousandSep::~ThousandSep() {
    std::cout.imbue(std::locale());
}

void fixedFloat() {
    std::cout.setf(std::ios::fixed, std::ios::floatfield);
}

void scientificFloat() {
    std::cout.setf(std::ios::scientific, std::ios::floatfield);
}
/// @endcond
//} //@StreamModifier

#if __linux__

void memInfoHost(std::size_t Req) {
    unsigned pages = static_cast<unsigned>(::sysconf(_SC_PHYS_PAGES));
    unsigned page_size = static_cast<unsigned>(::sysconf(_SC_PAGE_SIZE));
    memInfoPrint(pages * page_size, pages * page_size - 100u * (1u << 20u),
                 Req);
}
#endif

void memInfoPrint(std::size_t total, std::size_t free, std::size_t Req) {
    std::cout    << "  Total Memory:\t" << (total >> 20)    << " MB" << std::endl
                << "   Free Memory:\t" << (free >> 20)    << " MB" << std::endl
                << "Request memory:\t" << (Req >> 20)    << " MB" << std::endl
                << "   Request (%):\t" << ((Req >> 20) * 100) / (total >> 20)
                << " %" << std::endl << std::endl;
    if (Req > free)
        throw std::runtime_error(" ! Memory too low");
}

bool isDigit(std::string str) {
    return str.find_first_not_of("0123456789") == std::string::npos;
}

#if (__linux__)

#include <sys/resource.h>
#include <errno.h>

stackManagement::stackManagement() {
    if (getrlimit(RLIMIT_STACK, &ActualLimit))
        __ERROR("stackManagement::stackManagement() -> getrlimit()");
}

stackManagement::~stackManagement() {
    this->restore();
}

void stackManagement::setLimit(std::size_t size) {
    struct rlimit RL;
    RL.rlim_cur = size;
    RL.rlim_max = size;
    if (setrlimit(RLIMIT_STACK, &RL)) {
        if (errno == EFAULT)
            std::cout << "EFAULT" << std::endl;
        else if (errno == EINVAL)
            std::cout << "EINVAL" << std::endl;
        else if (errno == EPERM)
            std::cout << "EPERM" << std::endl;
        else if (errno == ESRCH)
            std::cout << "ESRCH" << std::endl;
        else
            std::cout << "?" << std::endl;
        __ERROR("stackManagement::setLimit() -> setrlimit()");
    }

}

void stackManagement::restore() {
    if (setrlimit(RLIMIT_STACK, &ActualLimit))
        __ERROR("stackManagement::restore() -> setrlimit()");
}

void stackManagement::checkUnlimited() {
    if (ActualLimit.rlim_cur != RLIM_INFINITY)
        __ERROR("stack size != unlimited (" << ActualLimit.rlim_cur << ")");
}

#include <signal.h> //  our new library

namespace {
    void ctrlC_HandleFun(int) {
        #if defined(__NVCC__)
            cudaDeviceReset();
        #endif
        std::exit(EXIT_FAILURE);
    }
}

void ctrlC_Handle() {
    signal(SIGINT, ctrlC_HandleFun);
}

#else

stackManagement::~stackManagement() {}
stackManagement::stackManagement() {}
void stackManagement::setLimit(std::size_t size) {}
void stackManagement::restore() {}
void ctrlC_Handle() {}

#endif

} //@xlib
