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
#include "Inference/CheckBinaryMonotony.hpp"
#include "Inference/CheckTernaryMonotony.hpp"

#include <iostream>
#include "Mining/Mining.hpp"
#include "Mining/Host/HostMiningFixed.hpp"
#include "Mining/Host/HostMiningGeneric.hpp"

using namespace timer;
using namespace xlib;

namespace mangrove {

template <>
void HostMiningFixed<numeric_t>(const HostParamStr& HostParam,
                                const TracePropSTR<numeric_t>& TraceProp) {

    ResultCollector<numeric_t> results_numeric(TraceProp.vars);
    Generator<numeric_t> generator(TraceProp.vars);

    Timer<HOST> TM(1, 30, Color::FG_L_CYAN);

    // Unary Mining
    //==========================================================================
    num2_t* unaryResult = static_cast<num2_t*>(
                                          results_numeric.getVectorResult<1>());
    TM.start();
    Funtion_TO_multiThreads(HostParam.multi, support::HostUnaryNumericRange,
                            unaryResult, std::cref(TraceProp));
    TM.getTimeA("Unary Range");
    // -------------------------------------------------------------------------

#if defined(NUMERIC_INFERENCE)
    // Monotony checking
    //==========================================================================
    TM.start();
    CheckBinaryMonotony<NBtemplate::size>  (results_numeric, TraceProp.vars);
    CheckTernaryMonotony<NTtemplate::size> (results_numeric, TraceProp.vars);
    TM.getTimeA("Monotony checking");
    // -------------------------------------------------------------------------
#endif

    // Binary Mining
    //==========================================================================
    // Generate FORWARD binary dictionary
    generator.setForwardDirection(true);
    results_numeric.setForwardDirection(true);

    int forward_dictionary_size = generator.generator<2>(results_numeric);
    dictionary_ptr_t<2> dictionary2F = (dictionary_ptr_t<2>)
                                        generator.dictionary;

    std::cout << "(Numeric)" << std::setw(30)
              << "Forward Dictionary size : "
              << forward_dictionary_size << std::endl << std::endl;

    // launch FORWARD binary mining
    result_t *binaryResultF = static_cast<result_t*>(
                                results_numeric.getVectorResult<2>());
    TM.start();

    Funtion_TO_multiThreads(HostParam.multi,
                            MultiThreadMiner<NBtemplate, numeric_t, 2>,
                            binaryResultF, dictionary2F,
                            forward_dictionary_size, TraceProp.sim_instants,
                            std::cref(TraceProp));

    TM.getTimeA("Forward Binary Range");

    // Generate BACKWARD binary dictionary
    generator.setForwardDirection(false);
    results_numeric.setForwardDirection(true);

    int backward_dictionary_size = generator.generator<2>(results_numeric);
    dictionary_ptr_t<2> dictionary2B = (dictionary_ptr_t<2>)
                                            (generator.dictionary +
                                             halfNumericDictionary);

    std::cout << "(Numeric)" << std::setw(30)
              << "Backward Dictionary size : "
              << backward_dictionary_size << std::endl << std::endl;

    // launch BACKWARD binary mining
    results_numeric.setForwardDirection(false);
    result_t *binaryResultB = static_cast<result_t*>(
                                          results_numeric.getVectorResult<2>());
    TM.start();

    Funtion_TO_multiThreads(HostParam.multi,
                            MultiThreadMiner<NBtemplate, numeric_t, 2>,
                            binaryResultB, dictionary2B,
                            backward_dictionary_size, TraceProp.sim_instants,
                            std::cref(TraceProp));

    TM.getTimeA("Backward Binary Range");

    generator.setForwardDirection(true);
    results_numeric.setForwardDirection(true);
    //--------------------------------------------------------------------------

#if defined(NUMERIC_INFERENCE)
    // Equivalence sets Mining
    support::MiningEquivalenceSets(results_numeric, dictionary2F,
                                   forward_dictionary_size);
#endif

    // Ternary Mining
    //==========================================================================
    int dictionary_size = generator.generator<3>(results_numeric);
    dictionary_ptr_t<3> dictionary3 = (dictionary_ptr_t<3>)generator.dictionary;

    std::cout << "(Numeric)" << std::setw(30)
              << "Ternary Dictionary size : " << dictionary_size
              << std::endl << std::endl;

    // launch ternary mining
    result_t *ternaryResult = (result_t*) results_numeric.getVectorResult<3>();
    TM.start();

    Funtion_TO_multiThreads(HostParam.multi,
                            MultiThreadMiner<NTtemplate, numeric_t, 3>,
                            ternaryResult, dictionary3,
                            dictionary_size, TraceProp.sim_instants,
                            std::ref(TraceProp));

    TM.getTime("Ternary Range");

    // Print Invariants
    //==========================================================================
    std::ofstream stream;
    if (HostParam.output_file)
        stream.open(HostParam.output_file);

    InvariantPrinter<numeric_t> printer_numeric (HostParam, TraceProp.vars);
    std::cout << " Unary Result: "
              << printer_numeric.unaryInvariants(results_numeric, stream)
              << std::endl;
    std::cout << " Binary Result: "
              << printer_numeric.binaryInvariants(results_numeric, stream)
              << std::endl;
    std::cout << "Ternary Result: "
              << printer_numeric.ternaryInvariants(results_numeric, stream)
              << std::endl;
}

namespace support {

void HostUnaryNumericRange(num2_t* Result,
                                  const TracePropSTR<numeric_t>& TraceProp,
                                  int th_index, int concurrency) {

    for (int V = th_index; V < TraceProp.vars; V += concurrency) {
        numeric_t *host_trace_var = TraceProp.host_trace +
                                (V * TraceProp.trace_length);

        numeric_t maxValue = *std::max_element(host_trace_var,
                            host_trace_var + TraceProp.trace_length);
        numeric_t minValue = *std::min_element(host_trace_var,
                            host_trace_var + TraceProp.trace_length);

        Result[V][0] = minValue;
        Result[V][1] = maxValue;
    }
}

void MiningEquivalenceSets(ResultCollector<numeric_t>& results,
                           dictionary_ptr_t<2> dictionary2F,
                           int forward_dictionary_size) {

    int* equivalence_sets = results.getEquivalenceSets();
    result_t* binaryResult = (result_t *) results.getVectorResult<2>();

    const result_t EQ_Idx = GET_POSITION<NBtemplate,mangrove::equal>::value;

    for (int index = 0; index < forward_dictionary_size; ++index) {
        result_t result = binaryResult[index];
        if ((result & EQ_Idx) != 1) continue;

        int first  = dictionary2F[index][0];
        int second = dictionary2F[index][1];

        //std::cout << first << "\t" << second << std::endl;
        // 1° step:
        // find the parent of first variable
        int first_parent = first;
        while (first_parent != equivalence_sets[first_parent])
            first_parent = equivalence_sets[first_parent];

        // 2° step:
        // path compression
        int first_brother = first;
        while (first_brother != equivalence_sets[first_brother]) {
            int tmp = equivalence_sets[first_brother];
            equivalence_sets[first_brother] = first_parent;
            first_brother = tmp;
        }

        // 1° step:
        // find the parent of second variable
        int second_parent = second;
        while (second_parent != equivalence_sets[second_parent])
            second_parent = equivalence_sets[second_parent];

        // 2° step:
        // path compression
        int second_brother = second;
        while (second_brother != equivalence_sets[second_brother]) {
            int tmp = equivalence_sets[second_brother];
            equivalence_sets[second_brother] = second_parent;
            second_brother = tmp;
        }

        equivalence_sets[second] = first_parent;
    }
}

} //@support
} //@mangrove
