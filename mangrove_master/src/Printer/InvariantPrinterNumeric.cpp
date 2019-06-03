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

#include <sstream>
#include <string>
#include <iostream>

//#include "config.cuh"
#include "Printer/InvariantPrinter.hpp"
//#include "Inference/InvariantChecker.cuh"
#include "Inference/RecursiveMatchRanges.cuh"
#include "Inference/RecursiveMatchRangesTernary.cuh"


using namespace std;

namespace mangrove
{
    // /////////////////////////////////////////////////////////////////////////////
    // specialization of the class Generator for the float variables - Constructor
    // /////////////////////////////////////////////////////////////////////////////

    InvariantPrinter<numeric_t>::InvariantPrinter(const MiningParamStr &variables, int vars)
    :  InvariantPrinterGuide<numeric_t>(variables, vars) {
        // ntd
    }

    InvariantPrinter<numeric_t>::~InvariantPrinter() {
        // ntd
    }

    // ///////////////////////////////////////////////////////////////////////////////////
    // specialization of the class Generator for the float variables - Printing methods
    // ///////////////////////////////////////////////////////////////////////////////////

    std::string InvariantPrinter<numeric_t>::_printUnaryNumericInvariant(int indexVar,
                                                                         num2_t *unaryResult) {
        std::stringstream ss;
        ss << std::left << std::setw(6) << _variables.name[indexVar] << " = " << unaryResult[indexVar][0];
        return ss.str();
    }

    std::string InvariantPrinter<numeric_t>::_printBinaryNumericInvariant(int indexVarA,
                                                                          int indexVarB,
                                                                          result_t result) {
        std::stringstream ss;
        ss  << "(" << _variables.name[indexVarA] << "; " << _variables.name[indexVarB] << ") ";

        const result_t EQ_Idx = GET_POSITION<NBtemplate,mangrove::equal>::value;
        const result_t NEQ_Idx = GET_POSITION<NBtemplate,
                                              mangrove::notEqual>::value;
        const result_t L_Idx = GET_POSITION<NBtemplate, mangrove::less>::value;
        const result_t LE_Idx = GET_POSITION<NBtemplate,
                                             mangrove::lessEq>::value;

        if (result & EQ_Idx) ss << "( = )";
        if (result & NEQ_Idx) ss << "( != )";
        if (result & L_Idx) ss << "( < )";
        if (result & LE_Idx) ss << "( <= ) ";

        for (result_t i = 5, index = 16; i < 32; ++i, index = index * 2)
          if (result & index)
              ss << "(INV " << i << ") ";

        return ss.str();
    }

    std::string InvariantPrinter<numeric_t>::_printTernaryNumericInvariant(int indexVarLeft,
                                                                           int indexVarRight1,
                                                                           int indexVarRight2,
                                                                           result_t result) {
        std::stringstream tmp;
        std::stringstream ss;

        tmp  << "(" << _variables.name[indexVarLeft] << "; " << _variables.name[indexVarRight1]
             << "; " << _variables.name[indexVarRight2] << ") ";

        ss << std::left << std::setw(20) << tmp.str() << std::left;

        for (result_t i = 1, index = 1; i < 32; ++i, index = index * 2)
          if (result & index)
              ss << "(INV " << i << ") ";

        return ss.str();
    }

    // ///////////////////////////////////////////////////////////////////////////////////
    // specialization of the class Generator for the float variables - Support methods
    // ///////////////////////////////////////////////////////////////////////////////////

    bool InvariantPrinter<numeric_t>::_isConstat(int varA,
                                                 ResultCollector<numeric_t> & results) {
        num2_t * unaryResult = static_cast<num2_t *> (results.getVectorResult<1>());
        return unaryResult[varA][0] == unaryResult[varA][1];
    }

    // /////////////////////////////////////////////////////////////////////////////////
    // specialization of the class Generator for the float variables - Public methods
    // /////////////////////////////////////////////////////////////////////////////////

    int InvariantPrinter<numeric_t>::unaryInvariants(ResultCollector<numeric_t> & results,
                                                     std::ostream & out) {
        num2_t *unaryResult = static_cast<num2_t *> (results.getVectorResult<1>());

        // number of unary invaraints
        int number_unaryInvariants = 0;

        // iterator for the variable
        for (int var_index = 0; var_index < _vars; ++var_index) {
            // is it a constant ?
            if ( _isConstat(var_index, results) ) {
                out << _printUnaryNumericInvariant(var_index, unaryResult) << std::endl;
                ++number_unaryInvariants;
            }
        }

        return number_unaryInvariants;
    }

#ifndef NUMERIC_INFERENCE

    int InvariantPrinter<numeric_t>::binaryInvariants(ResultCollector<numeric_t> & results,
                                                      std::ostream & out) {

        num2_t* unaryResult = static_cast<num2_t*>(
                                              results.getVectorResult<1>());
        // results array
        results.setForwardDirection(true);
        result_t *binaryResult = static_cast<result_t *> (results.getVectorResult<2>());

        // number of binary invaraints
        int numInvariants = 0;

        int index = 0;
        for (int first = 0; first < _vars; ++first) {

            #ifdef NOCONSTANT
                if (unaryResult[first][0] == unaryResult[first][1]) continue;
            #endif

            for (int second = first + 1; second < _vars; ++second) {

                #ifdef NOCONSTANT
                    if (unaryResult[second][0] == unaryResult[second][1]) continue;
                #endif

                result_t result = binaryResult[index];

                if (result != 0) {
                   numInvariants += __builtin_popcount(result);
                   out << _printBinaryNumericInvariant(first, second, result) << std::endl;
                }

                result = binaryResult[halfNumericResult + index];

                if (result != 0) {
                   numInvariants += __builtin_popcount(result);
                   out << _printBinaryNumericInvariant(second, first, result) << std::endl;
                }

                ++index;
            }
        }

        return numInvariants;
    }

    int InvariantPrinter<numeric_t>::ternaryInvariants(ResultCollector<numeric_t> & results,
                                                       std::ostream & out) {
        num2_t* unaryResult = static_cast<num2_t*>(
                                             results.getVectorResult<1>());

        result_t * ternaryResult = static_cast<result_t *> (results.getVectorResult<3>());

        // number of ternary invaraints
        int number_ternaryInvariants = 0;

        // index of the result array
        int index = 0;

        // iterator for the variable left
        for (int left = 0; left < _vars; ++left)
        {
            #ifdef NOCONSTANT
                if (unaryResult[left][0] == unaryResult[left][1]) continue;
            #endif

            // iterator for the variable right_1
            for (int right_1 = 0; right_1 < _vars; ++right_1)
            {
                if (right_1 == left) continue;

                #ifdef NOCONSTANT
                    if (unaryResult[right_1][0] == unaryResult[right_1][1]) continue;
                #endif

                // iterator for the variable right_2
                for (int right_2 = right_1 + 1; right_2 < _vars; ++right_2)
                {
                    if (right_2 == left) continue;

                    #ifdef NOCONSTANT
                        if (unaryResult[right_2][0] == unaryResult[right_2][1]) continue;
                    #endif

                    result_t result = ternaryResult[index++];

                    if (result != 0) {
                        number_ternaryInvariants += __builtin_popcount(result);
                        out << _printTernaryNumericInvariant(left, right_1, right_2, result) << std::endl;
                    }

                    result = ternaryResult[index++];

                    if (result != 0) {
                        number_ternaryInvariants += __builtin_popcount(result);
                        out << _printTernaryNumericInvariant(left, right_2, right_1, result) << std::endl;
                    }
                }
            }
        }

        return number_ternaryInvariants;
    }

#else

    int InvariantPrinter<numeric_t>::binaryInvariants(ResultCollector<numeric_t> & results, std::ostream & out)
    {
        num2_t* unaryResult = static_cast<num2_t*>(
                                              results.getVectorResult<1>());

        // results array
        results.setForwardDirection(true);
        result_t *binaryResult = (result_t *)results.getVectorResult<2>();

        // number of binary invaraints
        int numInvariants = 0;

        for (int first = 0; first < _vars; ++first) {

            #ifdef NOCONSTANT
                if (unaryResult[first][0] == unaryResult[first][1]) continue;
            #endif

            for (int second = first + 1; second < _vars; ++second) {

                #ifdef NOCONSTANT
                    if (unaryResult[second][0] == unaryResult[second][1]) continue;
                #endif

                int trip2two = results.getIndexFromTriplet2Pair(first, second);
                int resultIndex = results.associativeArrayForBinaryResult[trip2two];
                result_t result = 0;

                // Forward
                if (!RecursiveMatchRanges<NBtemplate::size>::ApplyT(results, first, second)) {
                    #ifdef GET_ALL_INVARIANTS
                        RecursiveNumericPrinter<NBtemplate::size>::ApplyT(results, first, second, result);
                        if (result != 0) {
                            out << _printBinaryNumericInvariant(first, second, result) << endl;
                            numInvariants += __builtin_popcount(result);
                        }
                    #endif
                }
                else {
                    result = binaryResult[resultIndex];

                    if (result != 0) {
                        out << _printBinaryNumericInvariant(first, second, result) << endl;
                        numInvariants += __builtin_popcount(result);
                    }
                }
                const result_t EQ_Idx = GET_POSITION<NBtemplate,
                                        mangrove::equal>::value;
                if (result & EQ_Idx) {
                    #ifdef GET_ALL_INVARIANTS
                        out << _printBinaryNumericInvariant(second, first, result) << endl;
                        numInvariants += __builtin_popcount(result);
                    #endif
                    continue;
                }

                // Backward
                if (!RecursiveMatchRanges<NBtemplate::size>::ApplyT(results, second, first)) {
                    #ifdef GET_ALL_INVARIANTS
                        result_t result = 0;
                        RecursiveNumericPrinter<NBtemplate::size>::ApplyT(results, second, first, result);
                        if (result != 0) {
                            out << _printBinaryNumericInvariant(second, first, result) << endl;
                            numInvariants += __builtin_popcount(result);
                        }
                    #endif
                }
                else {
                    int resultIndex2 = results.associativeArrayForBinaryResult[halfNumericResult + trip2two];
                    result = binaryResult[resultIndex2];

                    if (result != 0) {
                        out << _printBinaryNumericInvariant(second, first, result) << endl;
                        numInvariants += __builtin_popcount(result);
                    }
                }
            }
        }

        return numInvariants;
    }


    int InvariantPrinter<numeric_t>::ternaryInvariants(ResultCollector<numeric_t> & results,
                                                       std::ostream &  out) {

       int* equivalence_sets = results.getEquivalenceSets();
       num2_t* unaryResult = static_cast<num2_t*>(results.getVectorResult<1>());
       int* ternaryCommutative = results.getTernaryCommutative();

       bool areAllCommutativeInvs = true;
       for (int i = 0; i < NTtemplate::size; ++i)
            areAllCommutativeInvs &= (ternaryCommutative[i] == CommutativeProperty::YES);

        result_t * ternaryResult = static_cast<result_t *> (results.getVectorResult<3>());

        // number of ternary invaraints
        int numInvariants = 0;

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
                    index = results.ternaryAssociativeArray[indexAssoc];

                    result_t result = 0;

                    if (RecursiveMatchRangesTernary<NTtemplate::size>::ApplyT(results, left, right_1, right_2)) {
                        result = ternaryResult[index];

                        if (result != 0) {
                            out << _printTernaryNumericInvariant(left, right_1, right_2, result) << std::endl;
                            numInvariants += __builtin_popcount(result);
                        }
                    }
                    else {
                        #ifdef GET_ALL_INVARIANTS
                            TernaryRecursiveNumericPrinter<NTtemplate::size>::ApplyT(results, left, right_1, right_2, result);
                            if (result != 0) {
                                out << _printTernaryNumericInvariant(left, right_1, right_2, result) << std::endl;
                                numInvariants += __builtin_popcount(result);
                            }
                        #endif
                    }

                    if (areAllCommutativeInvs) {
                        #ifdef GET_ALL_INVARIANTS
                        if (result != 0) {
                            out << _printTernaryNumericInvariant(left, right_2, right_1, result) << std::endl;
                            numInvariants += __builtin_popcount(result);
                        }
                        #endif
                        continue;
                    }

                    result = 0;

                    indexAssoc = results.getTAindex(left_p, right_2_p, right_1_p);
                    index = results.ternaryAssociativeArray[indexAssoc];

                    if (RecursiveMatchRangesTernary<NTtemplate::size>::ApplyT(results, left, right_2, right_1)) {
                        result_t result = ternaryResult[index];

                        if (result != 0) {
                            out << _printTernaryNumericInvariant(left, right_2, right_1, result) << std::endl;
                            numInvariants += __builtin_popcount(result);
                        }
                    }
                    else {
                        #ifdef GET_ALL_INVARIANTS
                            result_t result = 0;
                            TernaryRecursiveNumericPrinter<NTtemplate::size>::ApplyT(results, left, right_2, right_1, result);
                            if (result != 0) {
                                out << _printTernaryNumericInvariant(left, right_2, right_1, result) << std::endl;
                                numInvariants += __builtin_popcount(result);
                            }
                        #endif
                    }
                }
            }
        }

        return numInvariants;
    }

#endif
}
