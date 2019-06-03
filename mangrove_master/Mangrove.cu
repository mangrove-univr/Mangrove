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
#include <iostream>
#include "XLib.hpp"

#include "mangrove.hpp"
#include "Utility.hpp"

using namespace mangrove;
using namespace mangrove_support;

template<typename T>
void Processing(GenerateParamStr& GenerateParam, ReadParamStr& ReadParam,
                HostParamStr& HostParam, GPUParamStr& GPUParam,
                ParameterStr& Parameters);

//------------------------------------------------------------------------------

int main(int argc, char *argv[]) {
    GenerateParamStr GenerateParam;
    ReadParamStr ReadParam;
    HostParamStr HostParam;
    GPUParamStr GPUParam;
    ParameterStr Parameters;

    setParameters(GenerateParam, ReadParam, HostParam, GPUParam, Parameters,
                  argc, argv);

    if (Parameters.mining_type == MINING_TYPE::BOOL) {
        std::cout << " Boolean Mining" << std::endl;
        Processing<bool>(GenerateParam, ReadParam, HostParam,
                         GPUParam, Parameters);
    }
    else if (Parameters.mining_type == MINING_TYPE::NUMERIC) {
        std::cout << " Numeric Mining" << std::endl;
        Processing<numeric_t>(GenerateParam, ReadParam, HostParam,
                              GPUParam, Parameters);
    }
#if defined(__NVCC__)
    cudaDeviceReset();
#endif
}

//------------------------------------------------------------------------------

template<typename T>
void Processing(GenerateParamStr& GenerateParam, ReadParamStr& ReadParam,
                HostParamStr& HostParam, GPUParamStr& GPUParam,
                ParameterStr& Parameters) {

    xlib::ThousandSep TS;
    TracePropSTR<T> TraceProp;

    if (ReadParam.trace_file != nullptr)
        readTrace(ReadParam, TraceProp);
    else
        generateMain(GenerateParam, TraceProp);

    // Reading variables' names
    if (Parameters.host)
        getVariables(Parameters.var_file, TraceProp.vars, HostParam);
    else
        getVariables(Parameters.var_file, TraceProp.vars, GPUParam);

    std::cout << std::endl
              << "               N. of threads  "
              << std::thread::hardware_concurrency() << std::endl
              << "             Trace variables  "
              << TraceProp.vars << std::endl
              << "         Simulation Instants  "
              << TraceProp.sim_instants << std::endl;
    if (std::is_same<T, bool>::value) {
        std::cout << "                Trace length  "
                  << TraceProp.trace_length << std::endl << std::endl;
    }

    if (Parameters.host)    HostMiningFixed(HostParam, TraceProp);
#if defined(__NVCC__)
    if (Parameters.gpu)     GPUMiningFixed(GPUParam, TraceProp);
#endif
    std::cout << std::endl;
}
