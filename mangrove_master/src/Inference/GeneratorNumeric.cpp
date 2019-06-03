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

#include <iostream>
#include "Inference/Generator.hpp"
// binary Inference
#include "Inference/RecursiveMatchRanges.cuh"
// ternary inference
#include "Inference/RecursiveMatchRangesTernary.cuh"

using namespace std;

namespace mangrove
{
    // /////////////////////////////////////////////////////////////////////////////
    // specialization of the class Generator for the BOOLEAN variables
    // /////////////////////////////////////////////////////////////////////////////

    Generator<numeric_t>::Generator(int vars)
      : GeneratorGuide(vars),
        _forwardDirection(true) {
        // ntd
    }

    Generator<numeric_t>::~Generator() {
        // ntd
    }

    void Generator<numeric_t>::setForwardDirection(bool direction) {
        _forwardDirection = direction;
    }

#ifndef NUMERIC_INFERENCE

    template <>
    int Generator<numeric_t>::generator<2>(ResultCollector<numeric_t> &results) {

        num2_t* unaryResult = static_cast<num2_t*>(
                                              results.getVectorResult<1>());

        // index of the dictionary array
        int index = 0;

        for (int first = 0; first < _vars; ++first) {

            #ifdef NOCONSTANT
                if (unaryResult[first][0] == unaryResult[first][1]) continue;
            #endif

            for (int second = first + 1; second < _vars; ++second) {

                #ifdef NOCONSTANT
                    if (unaryResult[second][0] == unaryResult[second][1]) continue;
                #endif

                if (_forwardDirection) {
                  dictionary[index] = static_cast<entry_t>(first);
                  dictionary[index + 1] = static_cast<entry_t>(second);
                }
                else {
                  dictionary[halfNumericDictionary + index] = static_cast<entry_t>(second);
                  dictionary[halfNumericDictionary + index + 1] = static_cast<entry_t>(first);
                }

                index = index + 2;
            }
        }

        return index / 2;
    }

    template <>
    int Generator<numeric_t>::generator<3>(ResultCollector<numeric_t> & results) {

      num2_t* unaryResult = static_cast<num2_t*>(
                                              results.getVectorResult<1>());

        // index of the dictionary array
      int index = 0;

      // iterator for the variable left
      for (int left = 0; left < _vars; ++left) {

          #ifdef NOCONSTANT
              if (unaryResult[left][0] == unaryResult[left][1]) continue;
          #endif
          // iterator for the variable right_1
          for (int right_1 = 0; right_1 < _vars; ++right_1) {
              if (right_1 == left) continue;

              #ifdef NOCONSTANT

                  if (unaryResult[right_1][0] == unaryResult[right_1][1]) continue;
              #endif

              // iterator for the variable right_2
              for (int right_2 = right_1 + 1; right_2 < _vars; ++right_2) {
                  if (right_2 == left) continue;

                  #ifdef NOCONSTANT
                      if (unaryResult[right_2][0] == unaryResult[right_2][1]) continue;
                  #endif

                  // A op f(b,c)
                  dictionary[index++] = static_cast<entry_t>(left);
                  dictionary[index++] = static_cast<entry_t>(right_1);
                  dictionary[index++] = static_cast<entry_t>(right_2);

                  // A op f(c,b)
                  dictionary[index++] = static_cast<entry_t>(left);
                  dictionary[index++] = static_cast<entry_t>(right_2);
                  dictionary[index++] = static_cast<entry_t>(right_1);
              }
          }
      }

      return index / 3;
    }

#else

    // this function considers forwards and backwards
    template <>
    int Generator<numeric_t>::generator<2>(ResultCollector<numeric_t> & results) {

        num2_t* unaryResult = static_cast<num2_t*>(
                                              results.getVectorResult<1>());

        results.setForwardDirection(true);
        result_t *binaryResult = static_cast<result_t *>(results.getVectorResult<2>());

        // index of the dictionary array
        int index = 0;

        // first op inv(second)
        for (int first = 0; first < _vars; ++first) {

            #ifdef NOCONSTANT
                if (unaryResult[first][0] == unaryResult[first][1]) continue;
            #endif

            for (int second = first + 1; second < _vars; ++second) {

                #ifdef NOCONSTANT
                    if (unaryResult[second][0] == unaryResult[second][1]) continue;
                #endif

                // FORWARD INFERENCE
                //--------------------------------------------------------------
                if (_forwardDirection &&
                    !RecursiveMatchRanges<NBtemplate::size>::ApplyT(results, first, second))
                        continue;

                // BACKWARD INFERENCE
                //--------------------------------------------------------------
                int trip2two = results.getIndexFromTriplet2Pair(first, second);
                int resultIndex = results.associativeArrayForBinaryResult[trip2two];
                result_t result = binaryResult[resultIndex];

                const result_t EQ_Idx = GET_POSITION<NBtemplate,
                                                     mangrove::equal>::value;
                if (!_forwardDirection &&
                     ( (result & EQ_Idx) ||
                       (!RecursiveMatchRanges<NBtemplate::size>::ApplyT(results, second, first)) )
                   )   continue;

                if (_forwardDirection) {
                    results.associativeArrayForBinaryResult[trip2two] = index / 2;
                    dictionary[index] = static_cast<entry_t>(first);
                    dictionary[index + 1] = static_cast<entry_t>(second);
                }
                else {
                    results.associativeArrayForBinaryResult[halfNumericResult + trip2two] = halfNumericResult + (index / 2);
                    dictionary[halfNumericDictionary + index] = static_cast<entry_t>(second);
                    dictionary[halfNumericDictionary + index + 1] = static_cast<entry_t>(first);
                }
                index = index + 2;
            }
        }

        return index / 2;
    }

    template <>
    int Generator<numeric_t>::generator<3>(ResultCollector<numeric_t> & results) {

      num2_t* unaryResult = static_cast<num2_t*>(results.getVectorResult<1>());
      int* ternaryCommutative = results.getTernaryCommutative();
      int* equivalence_sets = results.getEquivalenceSets();

      bool areAllCommutativeInvs = true;
      for (int i = 0; i < NTtemplate::size; ++i)
           areAllCommutativeInvs &= (ternaryCommutative[i] == CommutativeProperty::YES);

      // index of the dictionary array
      int index = 0;

      // iterator for the variable left
      for (int left = 0; left < _vars; ++left) {

          #ifdef NOCONSTANT
              if (unaryResult[left][0] == unaryResult[left][1]) continue;
          #endif

          // iterator for the variable right_1
          for (int right_1 = 0; right_1 < _vars; ++right_1) {
              if (right_1 == left) continue;

              #ifdef NOCONSTANT
                  if (unaryResult[right_1][0] == unaryResult[right_1][1]) continue;
              #endif

              // iterator for the variable right_2
              for (int right_2 = right_1 + 1; right_2 < _vars; ++right_2) {
                  if (right_2 == left) continue;

                  #ifdef NOCONSTANT
                      if (unaryResult[right_2][0] == unaryResult[right_2][1]) continue;
                  #endif

                  int left_p    = equivalence_sets[left];
                  int right_1_p = equivalence_sets[right_1];
                  int right_2_p = equivalence_sets[right_2];

                  int indexAssoc = results.getTAindex(left_p, right_1_p, right_2_p);
                  if (results.ternaryAssociativeArray[indexAssoc] != -1) continue;


                  results.ternaryAssociativeArray[indexAssoc] = index / 3;

                  // A op f(b,c)?
                  if (RecursiveMatchRangesTernary<NTtemplate::size>::ApplyT(results, left, right_1, right_2))
                  {
                      dictionary[index++] = static_cast<entry_t>(left);
                      dictionary[index++] = static_cast<entry_t>(right_1);
                      dictionary[index++] = static_cast<entry_t>(right_2);
                  }

                  if (areAllCommutativeInvs) continue;

                  indexAssoc = results.getTAindex(left_p, right_2_p, right_1_p);
                  results.ternaryAssociativeArray[indexAssoc] = index / 3;

                  // A op f(c,b)?
                  if (RecursiveMatchRangesTernary<NTtemplate::size>::ApplyT(results, left, right_2, right_1))
                  {
                      dictionary[index++] = static_cast<entry_t>(left);
                      dictionary[index++] = static_cast<entry_t>(right_2);
                      dictionary[index++] = static_cast<entry_t>(right_1);
                  }
              }
          }
      }

      return index / 3;
    }

#endif
}
