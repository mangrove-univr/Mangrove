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
#include "Printer/InvariantPrinter.hpp"
#include "Inference/ResultCollector.hpp"

#include "Mining/Mining.hpp"

#include "Mining/Device/GPUMiningFixed.cuh"
#include "Mining/Device/impl/GPUHelper.cuh"
#include "Mining/Device/Kernels/BooleanCounting.cuh"

#include "XLib.hpp"

using namespace xlib;
using namespace timer;

namespace mangrove {

AutoTuning(GPUUnaryBitCounting)
AutoTuning(GPUBinaryBitCounting)

template<>
void GPUMiningFixed<bool>(const GPUParamStr& GPUParam,
                          const TracePropSTR<bool>& TraceProp) {

    result_t* devResultPTR;
    cudaGetSymbolAddress((void**)&devResultPTR, devResult);

    using PROP_BASE = PROPERTY<256>;
    using PROP_TERNARY = PROPERTY<256, 1, BT>;
    Timer_cuda TM(1, 55, Color::FG_L_RED);

    const size_t length = TraceProp.getRoundTraceLength<PROP_BASE>()
                          / sizeof(unsigned);

    ResultCollector<bool> results_boolean(TraceProp.vars);
    Generator<bool> generator(TraceProp.vars, length * 32);

    // Unary Bit counting
    //==========================================================================
    devTraceSTR<bool> devProp1(TraceProp);

#if defined(AUTO_TUNING)
    AutoTuningClass::Init(TraceProp.vars);
    AutoTuningGPUUnaryBitCounting<PROP_BASE>::Apply(devProp1);
#endif
    cudaMemsetAsync(devResultPTR, 0x0, static_cast<size_t>(TraceProp.vars) *
                                sizeof(result_t));

    unsigned gridDim = gridConfig(GPUUnaryBitCounting<PROP_BASE>,
                                  PROP_BASE::BlockSize, 0, TraceProp.vars);
    TM.start();

    GPUUnaryBitCounting<PROP_BASE>
                       <<<gridDim, PROP_BASE::BlockSize>>>(devProp1);

    TM.getTime(" GPU Kernel - (Boolean) Unary Bit Counting");
    __CUDA_ERROR("GPUUnaryBitCounting");
    support::bitCountingCheckUnary(devResultPTR, TraceProp, length,
                                   GPUParam.check_result);
    cudaMemcpyFromSymbol(results_boolean.getVectorResult<1>(), devResult,
                         static_cast<size_t>(TraceProp.vars) *
                            sizeof(bitCounter_t));

    // Binary Bit counting
    //==========================================================================

    int dictionary_size = generator.generator<2>(results_boolean);
    std::cout << "(Boolean)" << std::setw(30) << " Binary Dictionary size : "
              << dictionary_size << std::endl << std::endl;

    cudaMemcpyToSymbolAsync(devDictionary, generator.dictionary,
                            static_cast<size_t>(dictionary_size * 2) *
                                sizeof(entry_t));

    devTraceSTR<bool> devProp2(TraceProp, dictionary_size);

#if defined(AUTO_TUNING)
    AutoTuningClass::Init(dictionary_size);
    AutoTuningGPUBinaryBitCounting<PROP_BASE>::Apply(devProp2);
#endif
    cudaMemsetAsync(devResultPTR, 0,
                static_cast<size_t>(dictionary_size)* sizeof(bitCounter_t));

    gridDim = gridConfig(GPUUnaryBitCounting<PROP_BASE>,
                         PROP_BASE::BlockSize, 0, dictionary_size);

    TM.start();

    GPUBinaryBitCounting<PROP_BASE><<<gridDim, PROP_BASE::BlockSize>>>
        (devProp2);

    TM.getTime(" GPU Kernel - (Boolean) Binary Bit Counting");

    __CUDA_ERROR("GPUBinaryBitCounting");
    support::bitCountingCheckBinary(devResultPTR,
                                    (dictionary_ptr_t<2>) generator.dictionary,
                                    dictionary_size, TraceProp, length,
                                    GPUParam.check_result);
    cudaMemcpyFromSymbol(results_boolean.getVectorResult<2>(),
                         devResult, static_cast<size_t>(dictionary_size) *
                                    sizeof(result_t));

    // Ternary Mining
    //==========================================================================

    dictionary_size = generator.generator<3>(results_boolean);
    std::cout << "(Boolean)" << std::setw(30) << " Ternary Dictionary size : "
              << dictionary_size << std::endl << std::endl;

    entry_t* dictionary = generator.dictionary;
    result_t* result3 = static_cast<result_t *>(
                            results_boolean.getVectorResult<3>());

    if (dictionary_size != 0) {
        GPUHelper<PROP_TERNARY>(result3, dictionary,
                                dictionary_size, TraceProp);
    }

    // Print Invariants
    //==========================================================================
    std::ofstream stream;
    if (GPUParam.output_file)
        stream.open(GPUParam.output_file);

    InvariantPrinter<bool> printer_boolean (GPUParam,
                                            TraceProp.vars, length * 32);
    std::cout << "  Unary Result: "
              << printer_boolean.unaryInvariants(results_boolean, stream)
              << std::endl;
    std::cout << " Binary Result: "
              << printer_boolean.binaryInvariants(results_boolean, stream)
              << std::endl;
    std::cout << "Ternary Result: "
              << printer_boolean.ternaryInvariants(results_boolean, stream)
              << std::endl;

    stream.setstate(std::ios_base::goodbit);
}

//------------------------------------------------------------------------------

namespace support {
//slow
void bitCountingCheckUnary(result_t* devResult,
                           const TracePropSTR<bool>& TraceProp,
                           int length, bool check_result) {
    if (check_result) {
        bitCounter_t* UnaryCounter = new bitCounter_t[TraceProp.vars];
        cudaMemcpy(UnaryCounter, devResult,
                   static_cast<size_t>(TraceProp.vars) * sizeof(result_t),
                   cudaMemcpyDeviceToHost);

        unsigned* host_trace_var = TraceProp.host_trace;
        for (int V = 0; V < TraceProp.vars; V++) {
            bitCounter_t counter = 0;
            for (unsigned i = 0; i < static_cast<unsigned>(length); i++)
                counter += __builtin_popcount(host_trace_var[i]);

            if (UnaryCounter[V] != counter) {
                __ERROR(__func__ << "  V: "  << V << " -> " << UnaryCounter[V]
                        << "   " << counter);
            }
            host_trace_var += TraceProp.trace_length;
        }
        std::cout << __func__ << ": OK" << std::endl << std::endl;
        delete[] UnaryCounter;
    }
}

void bitCountingCheckBinary(result_t* devResult, dictionary_ptr_t<2> Dictionary,
                            int dictionary_size,
                            const TracePropSTR<bool>& TraceProp,
                            int length, bool check_result) {
    if (check_result) {
        bitCounter_t* BinaryCounter = new bitCounter_t[dictionary_size];
        cudaMemcpy(BinaryCounter, devResult,
                   static_cast<size_t>(dictionary_size) * sizeof(result_t),
                   cudaMemcpyDeviceToHost);

        for (int D = 0; D < dictionary_size; D++) {
            unsigned* TraceA = TraceProp.host_trace + Dictionary[D][0] *
                               TraceProp.trace_length;
            unsigned* TraceB = TraceProp.host_trace + Dictionary[D][1] *
                               TraceProp.trace_length;
            bitCounter_t counter_AB = 0;
            for (unsigned i = 0; i < static_cast<unsigned>(length); i++)
                counter_AB += __builtin_popcount(TraceA[i] & TraceB[i]);

            if (BinaryCounter[D] != counter_AB) {
                __ERROR(__func__ << "  A & B    D: "  << D << " -> "
                        << BinaryCounter[D] << "   " << counter_AB);
            }
        }
        std::cout << __func__ << ": OK" << std::endl << std::endl;
        delete[] BinaryCounter;
    }
}

} //@support
} //@mangrove
