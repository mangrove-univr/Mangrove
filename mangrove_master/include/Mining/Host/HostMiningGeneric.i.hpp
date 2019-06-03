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

#include <type_traits>
#include <thread>
#include "Mining/TemplateEngine.cuh"

namespace mangrove {

namespace {

template<typename Invariant, int INV_SIZE = Invariant::arity - 1, int F_INDEX = 0>
struct recursiveInvariant {

    template<typename T, int SIZE>
    inline static
    typename std::conditional<F_INDEX == 0, bool, T>::type
    eval(T* (&Queue)[SIZE], int K) {

        return APPLY_FUN<F_INDEX, Invariant>::eval( Queue[F_INDEX][K],
                    recursiveInvariant<Invariant, INV_SIZE,
                                      F_INDEX + 1>::eval(Queue, K));
    }
};

template<typename Invariant, int INV_SIZE>
struct recursiveInvariant<Invariant, INV_SIZE, INV_SIZE> {
    template<typename T, int SIZE>
    inline static
    typename std::conditional<INV_SIZE == 0, bool, T>::type
    eval(T* (&Queue)[SIZE], int K) {
        return Queue[INV_SIZE][K];
    }
};

//==============================================================================

template<typename Template, unsigned INV_INDEX = Template::size - 1>
struct recursiveTemplate {

    template<typename T, int SIZE>
    inline static void
    eval(result_t& InvariantFlags, T* (&Queue)[SIZE], int K) {
        using Invariant = typename GET_INVARIANT<INV_INDEX, Template>::type;
        if (InvariantFlags & (1u << INV_INDEX)) {
            if (!recursiveInvariant<Invariant>::eval(Queue, K))
                InvariantFlags &= ~(1u << INV_INDEX);
        }

        recursiveTemplate<Template, INV_INDEX - 1>
            ::eval(InvariantFlags, Queue, K);
    }
};

template<typename Template>
struct recursiveTemplate<Template, static_cast<unsigned>(-1)> {
    template<typename T, int SIZE>
    inline static void eval(result_t&, T* (&)[SIZE], int) {}
};

}//@anonimous

//==============================================================================

template<typename Template, typename T, int ARITY>
void MultiThreadMiner(result_t* host_result,
                      dictionary_ptr_t<ARITY> Dictionary,
                      int dictionary_size, int length,
                      const TracePropSTR<T>& TraceProp,
                      int thread_index, int concurrency) {

    for (int D = thread_index; D < dictionary_size; D += concurrency) {
        result_t InvariantFlags = (1u << Template::size) - 1u;

        typename Template::type* Entry[Template::arity];
        for (int i = 0; i < Template::arity; i++) {
            Entry[i] = TraceProp.host_trace +
                         Dictionary[D][i] * TraceProp.trace_length;
        }

        for (int K = 0; K < length; K++) {
            recursiveTemplate<Template>::eval(InvariantFlags, Entry, K);
            if (InvariantFlags == 0)
                break;
        }
        host_result[D] = InvariantFlags;
    }
}

} //@mangrove
