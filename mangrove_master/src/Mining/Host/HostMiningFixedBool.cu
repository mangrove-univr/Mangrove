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

#include "Inference/Generator.hpp"
#include "Inference/ResultCollector.hpp"
#include "Printer/InvariantPrinter.hpp"

#include "Mining/Mining.hpp"
#include "Mining/Host/HostMiningFixed.hpp"
#include "Mining/Host/HostMiningGeneric.hpp"

#include "TemplateConf.cuh"

using namespace timer;
using namespace xlib;

namespace mangrove {

template<>
void HostMiningFixed<bool>(const HostParamStr& HostParam,
                           const TracePropSTR<bool>& TraceProp) {

    ResultCollector<bool> results_boolean(TraceProp.vars);
    Generator<bool> generator(TraceProp.vars, TraceProp.sim_instants32 * 32);

    Timer<HOST> TM(1, 55, Color::FG_L_CYAN);

    // -------------------------------------------------------------------------
    bitCounter_t* Result = static_cast<bitCounter_t *>(
                                          results_boolean.getVectorResult<1>());

    TM.start();

    Funtion_TO_multiThreads(HostParam.multi, support::HostUnaryBitCounting,
                            Result, std::cref(TraceProp));

    TM.getTimeA(" Host Kernel - (Boolean) Unary Bit Counting");

    // -------------------------------------------------------------------------
    int dictionary_size = generator.generator<2>(results_boolean);
    dictionary_ptr_t<2> Dictionary2 = (dictionary_ptr_t<2>) generator.dictionary;
    std::cout << "(Boolean)" << std::setw(30) << " Binary Dictionary size : "
              << dictionary_size << std::endl << std::endl;

    Result = static_cast<bitCounter_t*>(results_boolean.getVectorResult<2>());

    TM.start();

    Funtion_TO_multiThreads(HostParam.multi, support::HostBinaryBitCounting,
                            Result, Dictionary2, dictionary_size,
                            std::cref(TraceProp));

    TM.getTimeA(" Host Kernel - (Boolean) Binary Bit Counting");

    // -------------------------------------------------------------------------

    dictionary_size = generator.generator<3>(results_boolean);
    dictionary_ptr_t<3> Dictionary3 = (dictionary_ptr_t<3>)generator.dictionary;
    std::cout << "(Boolean)" << std::setw(30) << " Ternary Dictionary size : "
              << dictionary_size << std::endl << std::endl;

    result_t* Result3 = static_cast<result_t*>(
                                          results_boolean.getVectorResult<3>());

    TM.start();

    Funtion_TO_multiThreads(HostParam.multi,
                            MultiThreadMiner<BTtemplate, bool, 3>,
                            Result3, Dictionary3, dictionary_size,
                            TraceProp.sim_instants32, std::cref(TraceProp));

    TM.getTimeA(" Host Kernel - (Boolean) Ternary");

    // -------------------------------------------------------------------------
    std::ofstream stream;
    if (HostParam.output_file)
        stream.open(HostParam.output_file);

    InvariantPrinter<bool> printer_boolean (HostParam, TraceProp.vars,
                                            TraceProp.sim_instants32 * 32);
    std::cout << "  Unary Result: "
              << printer_boolean.unaryInvariants(results_boolean, stream)
              << std::endl;
    std::cout << " Binary Result: "
              << printer_boolean.binaryInvariants(results_boolean, stream)
              << std::endl;
    std::cout << "Ternary Result: "
              << printer_boolean.ternaryInvariants(results_boolean, stream)
              << std::endl;
}

namespace support {

void HostUnaryBitCounting(bitCounter_t* Result,
                          const TracePropSTR<bool>& TraceProp,
                          int thread_index, int concurrency) {

    for (int V = thread_index; V < TraceProp.vars; V += concurrency) {
        unsigned* host_trace_var = TraceProp.host_trace +
                                   TraceProp.trace_length * V;
        bitCounter_t local_result = 0;
        for (int i = 0; i < TraceProp.sim_instants32; i++)
            local_result += __builtin_popcount(host_trace_var[i]);
        Result[V] = local_result;
    }
}

void HostBinaryBitCounting(bitCounter_t* Result,
                           dictionary_ptr_t<2> Dictionary2,
                           int dictionary_size,
                           const TracePropSTR<bool>& TraceProp,
                           int thread_index, int concurrency) {

    for (int D = thread_index; D < dictionary_size; D += concurrency) {
        bitCounter_t local_result = 0;
        unsigned* TraceA = TraceProp.host_trace +
                           Dictionary2[D][0] * TraceProp.trace_length;
        unsigned* TraceB = TraceProp.host_trace +
                           Dictionary2[D][1] * TraceProp.trace_length;
        for (int i = 0; i < TraceProp.sim_instants32; i++)
             local_result += __builtin_popcount(TraceA[i] & TraceB[i]);
        Result[D] = static_cast<bitCounter_t>(local_result);
    }
}

} //@support
} //@mangrove
