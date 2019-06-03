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

#include "Inference/Generator.hpp"

namespace mangrove
{
    // /////////////////////////////////////////////////////////////////////////////
    // specialization of the class Generator for the BOOLEAN variables
    // /////////////////////////////////////////////////////////////////////////////

    Generator<bool>::Generator(int vars, int traceLength)
      : GeneratorGuide(vars),
        _traceLength(traceLength)
    {

    }

    Generator<bool>::~Generator()
    {

    }

#ifndef BOOL_INFERENCE

    template <>
    int Generator<bool>::generator<2>(ResultCollector<bool> & results)
    {
        // index of the dictionary array
        int index = 0;

        // iterator for the variable var_1
        for (int first = 0; first < _vars; ++first)
        {
            //iterator for the variable var_2
            for (int second = first + 1; second < _vars; ++second)
            {
                int trip2two = results.getIndexFromTriplet2Pair(first, second);
                results.associativeArrayForBinaryResult[trip2two] = index / 2;

                dictionary[index++] = static_cast<entry_t>(first);
                dictionary[index++] = static_cast<entry_t>(second);
            }
        }

        return (_vars * (_vars - 1)) >> 1;
    }

    template <>
    int Generator<bool>::generator<3>(ResultCollector<bool> & )
    {
        // index of the dictionary array
        int index = 0;

        // iterator for the variable left
        for (int left = 0; left < _vars; ++left)
        {
            // iterator for the variable right_1
            for (int right_1 = 0; right_1 < _vars; ++right_1)
            {
                if (right_1 == left) continue;

                // iterator for the variable right_2
                for (int right_2 = right_1 + 1; right_2 < _vars; ++right_2)
                {
                    if (right_2 == left) continue;

                    dictionary[index++] = static_cast<entry_t>(left);
                    dictionary[index++] = static_cast<entry_t>(right_1);
                    dictionary[index++] = static_cast<entry_t>(right_2);
                }
            }
        }

        return ((_vars * (_vars - 1)) >> 1) * (_vars - 2);
    }

#else

    template <>
    int Generator<bool>::generator<2>(ResultCollector<bool> & results)
    {
        bitCounter_t * unaryResult = static_cast<bitCounter_t *>(results.getVectorResult<1>());

        // index of the dictionary array
        int index = 0;

        // iterator for the variable var_1
        for (int first = 0; first < _vars; ++first)
        {
            // is it a constant variable ?
            if (unaryResult[first] == 0 || unaryResult[first] == _traceLength)
                continue;

            //iterator for the variable var_2
            for (int second = first + 1; second < _vars; ++second)
            {
                // is it a constant variable ?
                if (unaryResult[second] == 0 || unaryResult[second] == _traceLength)
                    continue;

                int trip2two = results.getIndexFromTriplet2Pair(first, second);
                results.associativeArrayForBinaryResult[trip2two] = index / 2;

                dictionary[index++] = static_cast<entry_t>(first);
                dictionary[index++] = static_cast<entry_t>(second);
            }
        }

        return index / 2;
    }

    template <>
    int Generator<bool>::generator<3>(ResultCollector<bool> & results)
    {
        bitCounter_t * unaryResult  = static_cast<bitCounter_t *> (results.getVectorResult<1>());
        bitCounter_t * binaryResult = static_cast<bitCounter_t *> (results.getVectorResult<2>());

        // number of generated triplets
        int number_of_triplets = 0;

        // index of the dictionary array
        int index = 0;

        // iterator for the variable left
        for (int left = 0; left < _vars; ++left)
        {
            // is it a constant variable ?
            if (unaryResult[left] == 0 || unaryResult[left] == _traceLength)
                continue;

            // iterator for the variable right_1
            for (int right_1 = 0; right_1 < _vars; ++right_1)
            {
                if (right_1 == left) continue;

                // is it a constant variable ?
                if (unaryResult[right_1] == 0 || unaryResult[right_1] == _traceLength)
                    continue;

                // iterator for the variable right_2
                for (int right_2 = right_1 + 1; right_2 < _vars; ++right_2)
                {
                    if (right_2 == left) continue;

                    // is it a constant variable ?
                    if (unaryResult[right_2] == 0 || unaryResult[right_2] == _traceLength)
                        continue;

                    // we have to have neither != nor = between right_1 and right_2
                    // =  iff |right_1| = |right_2| = |right_1 & right_2|
                    // != iff |!right_1 & right_2| = |T|
                    int trip2two = results.getIndexFromTriplet2Pair(right_1, right_2);
                    int rigth12_index = results.associativeArrayForBinaryResult[trip2two];

                    // we have either != or = between right_1 and right_2
                    if ( (unaryResult[right_1] == unaryResult[right_2]
                          && unaryResult[right_1] == binaryResult[rigth12_index]) ||
                         (binaryResult[rigth12_index] == 0
                          && (unaryResult[right_1] + unaryResult[right_2]) == _traceLength )
                       )
                       continue;

                    //( (unaryResult[right_2] - binaryResult[rigth12_index]) == _traceLength ) )
                    //continue;

                    // 0   -----         8 A = !B and !C
                    // 1    A = B & C    9 A = !(B xor C)
                    // 2    A = B & !C   10  -----
                    // 3   -----         11 A = B or !C
                    // 4    A = !B & C   12  -----
                    // 5   -----         13 A = !B or C
                    // 6    A = B xor C     14 A = !B or !C
                    // 7    A = B or C     15  -----

                    // |A|' = |T| - |A|
                    int left_prime = _traceLength - unaryResult[left];
                    // |B & !C| = |C| - |B & C|
                    int b_and_not_c = unaryResult[right_2] - binaryResult[rigth12_index];
                    // |!B & C| = |B| - |B & C|
                    int not_b_and_c = unaryResult[right_1] - binaryResult[rigth12_index];

                    if ( //-1 |A| = |B and C|
                        unaryResult[left] != binaryResult[rigth12_index] &&
                        //-2 |A| = |B and !C|
                        unaryResult[left] != b_and_not_c &&
                        //-4 |A| = |!B and C|
                        unaryResult[left] != not_b_and_c &&
                        //-6 |A| = |!B and C| + |B and !C|
                        unaryResult[left] != (not_b_and_c + b_and_not_c) &&
                        //-7 |A| = |B| + |C| - |B and C|
                        unaryResult[left] != (unaryResult[right_1] + unaryResult[right_2] - binaryResult[rigth12_index]) &&
                        //-8 |A|' = |B| + |C| - |B and C|
                        left_prime != (unaryResult[right_1] + unaryResult[right_2] - binaryResult[rigth12_index]) &&
                        //-9 |A|' = |!B and C| + |B and !C|
                        left_prime != (not_b_and_c + b_and_not_c) &&
                        //-11 |A|' = |!B and C|
                        left_prime != not_b_and_c &&
                        //-13 |A|' = |B and !C|
                        left_prime != b_and_not_c &&
                        //-14 |A|' = |B and C|
                        left_prime != binaryResult[rigth12_index])
                        continue;


                    dictionary[index++] = static_cast<entry_t>(left);
                    dictionary[index++] = static_cast<entry_t>(right_1);
                    dictionary[index++] = static_cast<entry_t>(right_2);

                    ++number_of_triplets;
                }
            }
        }

        return number_of_triplets;
    }

#endif
}
