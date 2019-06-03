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
    // specialization of the class ResultCollector for the BOOLEAN variables
    // /////////////////////////////////////////////////////////////////////////////

    ResultCollector<bool>::ResultCollector(int variables)
      : ResultCollectorGuide<bool>(variables),
        associativeArrayForBinaryResult()
    {
      _ternaryResults = new result_t[DictionarySize<bool, MAX_VARS, 3>::value];
    }

    ResultCollector<bool>::~ResultCollector()
    {
      delete[] _ternaryResults;
    }

    template <>
    void * ResultCollector<bool>::getVectorResult<1>()
    {
      return static_cast<void *>(& _unaryResults[0]);
    }

    template <>
    void * ResultCollector<bool>::getVectorResult<2>()
    {
      return static_cast<void *>(& _binaryResults[0]);
    }

    template <>
    void * ResultCollector<bool>::getVectorResult<3>()
    {
      return static_cast<void *>(& _ternaryResults[0]);
    }
}
