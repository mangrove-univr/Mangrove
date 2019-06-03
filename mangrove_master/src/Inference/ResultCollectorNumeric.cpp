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
 * @author Alessandro Danese
 * Univerity of Verona, Dept. of Computer Science
 * alessandro.danese@univr.it
 */

#include "Inference/ResultCollector.hpp"

namespace mangrove
{
    // /////////////////////////////////////////////////////////////////////////////
    // specialization of the class ResultCollector for the FLOAT variables
    // /////////////////////////////////////////////////////////////////////////////

    ResultCollector<numeric_t>::ResultCollector(int variables)
      : ResultCollectorGuide<numeric_t>(variables),
        associativeArrayForBinaryResult(),
        _forwardDirection(true)
    {
        _ternaryResults = new result_t[DictionarySize<numeric_t, MAX_VARS, 3>::value];
        ternaryAssociativeArray = new int [MAX_VARS * MAX_VARS * MAX_VARS];

        std::fill(ternaryAssociativeArray,
                  ternaryAssociativeArray + (MAX_VARS * MAX_VARS * MAX_VARS),
                  -1);

        for (int i = 0; i < variables; ++i) _equivalence_sets[i] = i;
    }

    int ResultCollector<numeric_t>::getTAindex(int left, int right_1, int right_2) {
        return left * (_vars * _vars) + _vars * right_1 + right_2;
    }

    ResultCollector<numeric_t>::~ResultCollector() {
        delete[] _ternaryResults;
        delete[] ternaryAssociativeArray;
    }

    template <>
    void * ResultCollector<numeric_t>::getVectorResult<1>() {
        return static_cast<void *>(_unaryResults);
    }

    template <>
    void * ResultCollector<numeric_t>::getVectorResult<2>() {
      if (_forwardDirection)
        return static_cast<void *>(_binaryResults);
      else
        return static_cast<void *>(_binaryResults + halfNumericResult);
    }

    template <>
    void * ResultCollector<numeric_t>::getVectorResult<3>() {
        return static_cast<void *>(_ternaryResults);
    }

    int * ResultCollector<numeric_t>::getMonotony() {
        return _monotony;
    }

    int * ResultCollector<numeric_t>::getTernaryMonotony() {
        return _ternaryMonotony;
    }

    int * ResultCollector<numeric_t>::getTernaryCommutative() {
        return _ternaryCommutative;
    }

    int * ResultCollector<numeric_t>::getEquivalenceSets() {
        return _equivalence_sets;
    }

    void ResultCollector<numeric_t>::setForwardDirection(bool direction) {
        _forwardDirection = direction;
    }
}
