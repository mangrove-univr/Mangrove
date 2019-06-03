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
#pragma once

#include <limits>
#include "XLib.hpp"
#include "config.cuh"

namespace mangrove {

class AutoTuningClass {

private:
    static const int N_OF_BLOCK_CONFIGURATION =
                                        xlib::LOG2<xlib::MAX_BLOCKSIZE>::value -
                                        xlib::LOG2<xlib::WARP_SIZE>::value + 1;
    static const int N_OF_STEP_CONFIGURATION =
                                             xlib::LOG2<MAX_STEP>::value + 1;
    static const int N_CONF = N_OF_BLOCK_CONFIGURATION *
                              N_OF_STEP_CONFIGURATION;

    static int problem_size;
    static float ExecDuration[N_CONF];

public:
    static void Init(const int _problem_size) {
        problem_size = _problem_size;
        std::fill(ExecDuration, ExecDuration + N_CONF,
                  std::numeric_limits<float>::max());
    }

    template<int BLOCK_SIZE, int STEP, class FUN_T, typename... Ts>
    static void Exec(FUN_T FUN, Ts... Args) {
        using namespace timer;
        using namespace xlib;

        int grid_dim = gridConfig(FUN, BLOCK_SIZE, 0, problem_size);
        if (grid_dim == 0) return;
        //grid_dim = problem_size;

        Timer_cuda TM;
        TM.start();

        FUN<<<grid_dim, BLOCK_SIZE>>>(Args...);

        TM.stop();
        const int index = (LOG2<BLOCK_SIZE>::value -
                                LOG2<xlib::WARP_SIZE>::value)
                          * N_OF_STEP_CONFIGURATION + LOG2<STEP>::value;
        ExecDuration[index] = TM.duration();
        __CUDA_ERROR("AutoTuning");
    }

    static void computeResults() {
        auto it = std::min_element(ExecDuration, ExecDuration + N_CONF);
        int index = 0;
        for (unsigned B = xlib::WARP_SIZE; B <= xlib::MAX_BLOCKSIZE; B *= 2u) {
            for (int S = 1; S <= MAX_STEP; S *= 2) {
                const float val = ExecDuration[index++];
                if (val == std::numeric_limits<float>::max()) continue;
                if (val != *it) {
                    std::cout << std::setprecision(2) << std::fixed << std::left
                              << std::setw(8) << B << std::setw(8) << S
                              << std::setw(8) << val << std::endl;
                }
                else {
                    std::cout << xlib::Color::FG_RED << std::setprecision(2)
                              << std::fixed << std::left
                              << std::setw(8) << B << std::setw(8) << S
                              << std::setw(8) << val
                              << xlib::Color::FG_DEFAULT << std::endl;
                }
            }
        }
        std::cout << std::endl;
    }
};

//------------------------------------------------------------------------------

#define AutoTuning(FUN)                                                        \
                                                                               \
template<typename _PROP, int BLOCK_SIZE = xlib::WARP_SIZE, int STEP = 1>       \
struct AutoTuning##FUN {                                                       \
                                                                               \
    template<typename... Ts>                                                   \
    static void Apply(Ts... Args) {                                            \
        using PROP = PROPERTY<BLOCK_SIZE, STEP, _PROP::TEMPL_INDEX>;           \
        AutoTuningClass::Exec<BLOCK_SIZE, STEP>( FUN <PROP>, Args...);         \
                                                                               \
        AutoTuning##FUN<_PROP, BLOCK_SIZE, STEP * 2>::Apply(Args...);          \
    }                                                                          \
};                                                                             \
                                                                               \
template<typename _PROP, int BLOCK_SIZE>                                       \
struct AutoTuning##FUN<_PROP, BLOCK_SIZE, MAX_STEP> {                          \
                                                                               \
    template<typename... Ts>                                                   \
    static void Apply(Ts... Args) {                                            \
        using PROP = PROPERTY<BLOCK_SIZE, MAX_STEP, _PROP::TEMPL_INDEX>;       \
        AutoTuningClass::Exec<BLOCK_SIZE, MAX_STEP>(FUN<PROP>, Args...);       \
                                                                               \
        AutoTuning##FUN<_PROP, BLOCK_SIZE * 2, 1>::Apply(Args...);             \
    }                                                                          \
};                                                                             \
                                                                               \
template<typename _PROP>                                                       \
struct AutoTuning##FUN<_PROP, xlib::MAX_BLOCKSIZE, MAX_STEP> {                 \
                                                                               \
    template<typename... Ts>                                                   \
    static void Apply(Ts... Args) {                                            \
        using PROP = PROPERTY<xlib::MAX_BLOCKSIZE, MAX_STEP,                   \
                              _PROP::TEMPL_INDEX>;                             \
        AutoTuningClass::Exec<xlib::MAX_BLOCKSIZE, MAX_STEP>                   \
            (FUN<PROP>, Args...);                                              \
        AutoTuningClass::computeResults();                                     \
    }                                                                          \
};

//------------------------------------------------------------------------------

#define AutoTuningGen(FUN)                                                     \
                                                                               \
template<typename _PROP, typename R,                                           \
         int BLOCK_SIZE = xlib::WARP_SIZE, int STEP = 1>                       \
struct AutoTuningGen##FUN {                                                    \
                                                                               \
    template<typename... Ts>                                                   \
    static void Apply(Ts... Args) {                                            \
        using PROP = PROPERTY<BLOCK_SIZE, STEP, _PROP::TEMPL_INDEX>;           \
        AutoTuningClass::Exec<BLOCK_SIZE, STEP>( FUN <PROP, R>, Args...);      \
                                                                               \
        AutoTuningGen##FUN<_PROP, R, BLOCK_SIZE, STEP * 2>::Apply(Args...);    \
    }                                                                          \
};                                                                             \
                                                                               \
template<typename _PROP, typename R, int BLOCK_SIZE>                           \
struct AutoTuningGen##FUN<_PROP, R, BLOCK_SIZE, MAX_STEP> {                    \
                                                                               \
    template<typename... Ts>                                                   \
    static void Apply(Ts... Args) {                                            \
        using PROP = PROPERTY<BLOCK_SIZE, MAX_STEP, _PROP::TEMPL_INDEX>;       \
        AutoTuningClass::Exec<BLOCK_SIZE, MAX_STEP>(FUN<PROP, R>, Args...);    \
                                                                               \
        AutoTuningGen##FUN<_PROP, R, BLOCK_SIZE * 2, 1>::Apply(Args...);       \
    }                                                                          \
};                                                                             \
                                                                               \
template<typename _PROP, typename R>                                           \
struct AutoTuningGen##FUN<_PROP, R, xlib::MAX_BLOCKSIZE, MAX_STEP> {           \
                                                                               \
    template<typename... Ts>                                                   \
    static void Apply(Ts... Args) {                                            \
        using PROP = PROPERTY<xlib::MAX_BLOCKSIZE, MAX_STEP,                   \
                              _PROP::TEMPL_INDEX>;                             \
        AutoTuningClass::Exec<xlib::MAX_BLOCKSIZE, MAX_STEP>                   \
            (FUN<PROP, R>, Args...);                                           \
        AutoTuningClass::computeResults();                                     \
    }                                                                          \
};

} //@mangrove
