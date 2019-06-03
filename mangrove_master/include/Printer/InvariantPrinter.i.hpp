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

#pragma once
#include "InvariantPrinter.hpp"
#include <string>
#include <cstring>

namespace mangrove
{
    template <typename T>
    InvariantPrinterGuide<T>::InvariantPrinterGuide(const MiningParamStr& variables, int vars) :
      _vars(vars),
      _variables() {

      for (int i = 0; i < vars; ++i) {
        const size_t length = std::strlen(variables.name[i]) + 1;
        _variables.name[i] = new char[length]();
        std::copy(variables.name[i], variables.name[i] + length, _variables.name[i]);
      }

    }

    template <typename T>
    InvariantPrinterGuide<T>::~InvariantPrinterGuide() {
      // ntd
    }

    template<int F_INDEX>
    struct RecursiveNumericPrinter {

        // check: left op inv(right_1,right_2)
        inline static void ApplyT (ResultCollector<numeric_t> & results, int first, int second,
                                   result_t &result) {

            using function = typename GET_INVARIANT<F_INDEX-1, NBtemplate>::type::unary;
            using cmp = typename GET_INVARIANT<F_INDEX-1, NBtemplate>::type::first;

            int *monotony = results.getMonotony();
            num2_t *unaryResult = static_cast<num2_t *>(results.getVectorResult<1>());

            if (monotony[F_INDEX-1] == MonotonyProperty::INCREASING ||
                monotony[F_INDEX-1] == MonotonyProperty::DECREASING) {

                numeric_t f_max_second = function()(unaryResult[second][1]);
                numeric_t f_min_second = function()(unaryResult[second][0]);

                if (f_max_second == f_min_second) {
                    if (std::is_same<cmp, equal<numeric_t> >::value &&
                        unaryResult[first][0] == unaryResult[first][1] &&
                        unaryResult[first][0] == f_min_second)
                    { result |= 1 << (F_INDEX-1); }
                }

                if (std::is_same<cmp, notEqual<numeric_t> >::value &&
                    (unaryResult[first][0] > std::max(f_min_second, f_max_second) ||
                     unaryResult[first][1] < std::min(f_min_second, f_max_second)) )
                    { result |= 1 << (F_INDEX-1); }

                if (std::is_same<cmp, less<numeric_t> >::value &&
                    unaryResult[first][1] < std::min(f_min_second, f_max_second) )
                    { result |= 1 << (F_INDEX-1); }

                if (std::is_same<cmp, lessEq<numeric_t> >::value &&
                    unaryResult[first][1] <= std::min(f_min_second, f_max_second) )
                    { result |= 1 << (F_INDEX-1); }
            }

            return RecursiveNumericPrinter<F_INDEX-1>::ApplyT(results, first, second, result);
        }
    };

    template<>
    struct RecursiveNumericPrinter<0> {
        inline static void ApplyT (ResultCollector<numeric_t> & , int , int ,
                                   result_t &) { }
    };


    template<int F_INDEX>
    struct TernaryRecursiveNumericPrinter {

        // check: left op inv(right_1,right_2)
        inline static void ApplyT (ResultCollector<numeric_t> & results, int left, int right_1, int right_2,
                                   result_t &result) {
                                       
           using invariant = typename GET_INVARIANT<F_INDEX-1, NTtemplate>::type;
           using cmp = typename GET_INVARIANT<F_INDEX-1, NTtemplate>::type::first;

           int *monotony = results.getTernaryMonotony();
           num2_t *unaryResult = static_cast<num2_t *>(results.getVectorResult<1>());

           if (monotony[F_INDEX-1] == MonotonyProperty::INCREASING ||
               monotony[F_INDEX-1] == MonotonyProperty::DECREASING) {

               numeric_t f_max_second = APPLY_FUN<1, invariant>::eval(unaryResult[right_1][1],
                                                                      unaryResult[right_2][1]);

               numeric_t f_min_second = APPLY_FUN<1, invariant>::eval(unaryResult[right_1][0],
                                                                      unaryResult[right_2][0]);

              if (std::is_same<cmp, notEqual<numeric_t> >::value &&
                  (unaryResult[left][0] > std::max(f_min_second, f_max_second) ||
                   unaryResult[left][1] < std::min(f_min_second, f_max_second)) )
                { result |= 1 << (F_INDEX-1); }

              if (std::is_same<cmp, less<numeric_t> >::value &&
                  unaryResult[left][1] < std::min(f_min_second, f_max_second) )
                { result |= 1 << (F_INDEX-1); }

              if ( std::is_same<cmp, lessEq<numeric_t>>::value &&
                   unaryResult[left][1] <= std::min(f_min_second, f_max_second) )
                { result |= 1 << (F_INDEX-1); }
           }

           return TernaryRecursiveNumericPrinter<F_INDEX-1>::ApplyT(results, left, right_1,  right_2, result);
        }
    };

    template<>
    struct TernaryRecursiveNumericPrinter<0> {

        // check: first op inv(second)
        inline static void ApplyT (ResultCollector<numeric_t> &, int, int, int,
                                   result_t &) { }
    };
}
