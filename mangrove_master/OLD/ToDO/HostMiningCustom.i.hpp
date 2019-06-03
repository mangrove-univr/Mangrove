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

#include "Mining/Host/HostGenericMining.hpp"
#include "Mining/Dictionary.hpp"
#include "Mining/TemplateEngine.cuh"

#include "XLib.hpp"

namespace mangrove {

namespace {

template<int TEMPL_SIZE, int TEMPL_INDEX = 0>
struct testAllHost {
    template<typename T>
    static void Apply(TracePropSTR<T>& TraceProp, const bool MultiCore) {
        using namespace timer;
        using templ = typename GET_TEMPL<TEMPL_INDEX, TemplateSET>::templ_t;
        using T2 = typename templ::type;
        Timer<HOST> TM(23, 2, StreamModifier::FG_CYAN);

        const int length = std::is_same<bool, T>::value ?
                           Div(TraceProp.traceLength, 32) :
                           TraceProp.traceLength;

        entry_t CurrentDictionary[MAX_DICTIONARY_ENTRIES][templ::ARITY];
        const int dictionarySize = generateDictionary(CurrentDictionary,
                                                 TraceProp.Vars, false);

        T2* DictionaryPTR[MAX_DICTIONARY_ENTRIES][templ::ARITY];
        DictionaryToPTR(DictionaryPTR, CurrentDictionary, dictionarySize);
        result_t Result[MAX_DICTIONARY_ENTRIES];

        std::cout << TemplStrT<typename templ::type>::NAME
                  << std::left << std::setw(30)
                  << " " << TemplStrA<templ::ARITY>::NAME
                  << " Dictionary size : "
                  << dictionarySize << std::endl << std::endl;

        TM.start();

        Funtion_TO_multiThreads(MultiCore,
                                MultiThreadMiner<TEMPL_INDEX, T2, templ::ARITY>,
                                Result, DictionaryPTR,
                                dictionarySize, length);

        TM.getTime(TemplStrA<templ::ARITY>::NAME);
        testAllHost<TEMPL_SIZE, TEMPL_INDEX + 1>::Apply(TraceProp, MultiCore);
    }
};

template<int TEMPL_SIZE>
struct testAllHost<TEMPL_SIZE, TEMPL_SIZE> {
    template<typename T>
    static void Apply(TracePropSTR<T>&, const bool) {}
};

} //@anonymous

//------------------------------------------------------------------------------

template<typename T>
void HostMiningUser(TracePropSTR<T>& TraceProp, const bool MultiCore) {
    testAllHost<TemplateSET::size>::Apply(TraceProp, MultiCore);
}

} //@mangrove
