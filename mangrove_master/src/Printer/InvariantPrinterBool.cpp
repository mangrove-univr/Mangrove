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

#include "config.cuh"
#include "Printer/InvariantPrinter.hpp"

#include "Mining/TemplateEngine.cuh"
#include "UserFunctions.cuh"

namespace mangrove
{
    const result_t AND_Idx         = GET_POSITION<BTtemplate, equal, AND>::value;
    const result_t IMPLY_Idx       = GET_POSITION<BTtemplate, equal, IMPLY>::value;
    const result_t R_IMPLY_Idx     = GET_POSITION<BTtemplate, equal, R_IMPLY>::value;
    const result_t XOR_Idx         = GET_POSITION<BTtemplate, equal, XOR>::value;
    const result_t OR_Idx          = GET_POSITION<BTtemplate, equal, OR>::value;
    const result_t NOR_Idx         = GET_POSITION<BTtemplate, equal, NOR>::value;
    const result_t XNOR_Idx        = GET_POSITION<BTtemplate, equal, XNOR>::value;
    const result_t NOT_R_IMPLY_Idx = GET_POSITION<BTtemplate, equal, NOT_R_IMPLY>::value;
    const result_t NOT_IMPLY_Idx   = GET_POSITION<BTtemplate, equal, NOT_IMPLY>::value;
    const result_t NAND_Idx        = GET_POSITION<BTtemplate, equal, NAND>::value;

    // /////////////////////////////////////////////////////////////////////////////
    // specialization of the class Generator for the BOOLEAN variables - Constructor
    // /////////////////////////////////////////////////////////////////////////////
    InvariantPrinter<bool>::InvariantPrinter(const MiningParamStr &variables, int vars, int traceLength)
    :  InvariantPrinterGuide<bool>(variables, vars),
       _traceLength(traceLength) {
        // ntd
    }

    InvariantPrinter<bool>::~InvariantPrinter() {
      // ntd
    }

    // ///////////////////////////////////////////////////////////////////////////////////
    // specialization of the class Generator for the BOOLEAN variables - Printing methods
    // ///////////////////////////////////////////////////////////////////////////////////

    std::string InvariantPrinter<bool>::_printUnaryBoolInvariant(int indexVar, bool result) {
        std::stringstream ss;

        if (!result) ss << "!";
        ss << _variables.name[indexVar];
        return ss.str();
    }

    std::string InvariantPrinter<bool>::_printBinaryBoolInvariant(int indexVarA, int indexVarB, bool equal) {
        std::stringstream ss;
        ss << _variables.name[indexVarA] << " ";

        if (!equal) ss << "!";
        ss << "= " << _variables.name[indexVarB];
        return ss.str();
    }

    std::string InvariantPrinter<bool>::_printTernaryBoolInvariant(int indexVarLeft, int indexVarRight1,
                                                                   int indexVarRight2, result_t result)
    {
        std::stringstream ss;
        ss << "(" << _variables.name[indexVarLeft]
           << "; " << _variables.name[indexVarRight1]
           << "; " << _variables.name[indexVarRight2] << ")\t" << "(";

        if (result & AND_Idx)         ss << "AND";
        if (result & IMPLY_Idx)       ss << ", IMPLY";
        if (result & R_IMPLY_Idx)     ss << ", R_IMPLY";
        if (result & XOR_Idx)         ss << ", XOR";
        if (result & OR_Idx)          ss << ", OR";
        if (result & NOR_Idx)         ss << ", NOR";
        if (result & XNOR_Idx)        ss << ", XNOR";
        if (result & NOT_R_IMPLY_Idx) ss << ", NOT_R_IMPLY";
        if (result & NOT_IMPLY_Idx)   ss << ", NOT_IMPLY";
        if (result & NAND_Idx)        ss << ", NAND";

        ss << ")";

        return ss.str();
    }

    // ///////////////////////////////////////////////////////////////////////////////////
    // specialization of the class Generator for the BOOLEAN variables - Support methods
    // ///////////////////////////////////////////////////////////////////////////////////

    bool InvariantPrinter<bool>::_isConstatTrue(int varA, ResultCollector<bool> & results) {
        bitCounter_t *unaryResult = static_cast<bitCounter_t *>(results.getVectorResult<1>());
        return unaryResult[varA] == _traceLength;
    }

    bool InvariantPrinter<bool>::_isConstatFalse(int varA, ResultCollector<bool> & results) {
        bitCounter_t *unaryResult = static_cast<bitCounter_t *>(results.getVectorResult<1>());
        return unaryResult[varA] == 0;
    }

    bool InvariantPrinter<bool>::_isConstat(int varA, ResultCollector<bool> & results) {
        return _isConstatFalse(varA, results) || _isConstatTrue(varA, results);
    }

    bool InvariantPrinter<bool>::_areEqual(int varA, int varB, ResultCollector<bool> & results) {
        bitCounter_t *unaryResult = static_cast<bitCounter_t *> (results.getVectorResult<1>());

        // do they have the same occurences?
        if (unaryResult[varA] != unaryResult[varB]) return false;

        // are they constant?
        if (unaryResult[varA] == 0 || (unaryResult[varA] == _traceLength)) return true;

        bitCounter_t *binaryResult  = static_cast<bitCounter_t *> (results.getVectorResult<2>());
        int indexAssocArray = results.getIndexFromTriplet2Pair(varA, varB);
        int indexBinaryResult = results.associativeArrayForBinaryResult[indexAssocArray];

        // (|A| == |B| == |A & B|) -> ==
        return (unaryResult[varA] == binaryResult[indexBinaryResult]);
    }

    bool InvariantPrinter<bool>::_areNotEqual(int varA, int varB, ResultCollector<bool> & results) {
        bitCounter_t *unaryResult = static_cast<bitCounter_t *> (results.getVectorResult<1>());

        // are they constant?
        if ((unaryResult[varA] == 0 && unaryResult[varB] == _traceLength) ||
            (unaryResult[varA] == _traceLength && unaryResult[varB] == 0)) return true;

        // do they cover the trace?
        if ((unaryResult[varA] + unaryResult[varB]) != _traceLength) return false;

        bitCounter_t *binaryResult  = static_cast<bitCounter_t *>(results.getVectorResult<2>());
        int indexAssocArray = results.getIndexFromTriplet2Pair(varA, varB);
        int indexBinaryResult = results.associativeArrayForBinaryResult[indexAssocArray];

        // (|A| + |B| - |A & B| == |T|) -> !=
        return  binaryResult[indexBinaryResult] == 0;
    }

    bool InvariantPrinter<bool>::_neverBothZero(int varA, int varB, ResultCollector<bool> & results) {
        // at least one variable is const at true
        if ( _isConstatTrue(varA, results) || _isConstatTrue(varB, results) ) return true;

        // they are both const at false
        if (_isConstatFalse(varA, results) && _isConstatFalse(varB, results)) return false;

        // they are always not equal
        if  (_areNotEqual(varA, varB, results)) return true;

        if ( (_isConstatFalse(varA, results) && !_isConstatTrue(varB, results)) ||
             (_isConstatFalse(varB, results) && !_isConstatTrue(varA, results)) )
              return false;

        bitCounter_t *unaryResult  = static_cast<bitCounter_t *> (results.getVectorResult<1>());
        bitCounter_t *binaryResult = static_cast<bitCounter_t *> (results.getVectorResult<2>());
        int indexAssocArray = results.getIndexFromTriplet2Pair(varA, varB);
        int indexBinaryResult = results.associativeArrayForBinaryResult[indexAssocArray];

        return (unaryResult[varA] + unaryResult[varB] - binaryResult[indexBinaryResult]) == _traceLength;
    }

    bool InvariantPrinter<bool>::_neverBothOne(int varA, int varB, ResultCollector<bool> & results) {
        // at least one variable is const at false
        if ( _isConstatFalse(varA, results) || _isConstatFalse(varB, results) ) return true;

        // they are both const at true
        if (_isConstatTrue(varA, results) && _isConstatTrue(varB, results)) return false;

        if ( (_isConstatTrue(varA, results) && !_isConstatFalse(varB, results)) ||
             (_isConstatTrue(varB, results) && !_isConstatFalse(varA, results)) )
          return false;

        // they are always not equals
        if  (_areNotEqual(varA, varB, results)) return true;

        bitCounter_t *binaryResult  = static_cast<bitCounter_t *> (results.getVectorResult<2>());
        int indexAssocArray = results.getIndexFromTriplet2Pair(varA, varB);
        int indexBinaryResult = results.associativeArrayForBinaryResult[indexAssocArray];

        return binaryResult[indexBinaryResult] == 0;
    }

    bool InvariantPrinter<bool>::_neverOneZero(int varA, int varB, ResultCollector<bool> & results) {
        if (_isConstatFalse(varA, results) || _isConstatTrue(varB, results)) return true;

        if (!_isConstatFalse(varA, results) && _isConstatFalse(varB, results)) return false;

        bitCounter_t *binaryResult = static_cast<bitCounter_t *> (results.getVectorResult<2>());
        int indexAssocArray = results.getIndexFromTriplet2Pair(varA, varB);
        int indexBinaryResult = results.associativeArrayForBinaryResult[indexAssocArray];

        bitCounter_t *unaryResult = static_cast<bitCounter_t *> (results.getVectorResult<1>());
        return unaryResult[varA] == binaryResult[indexBinaryResult];
    }

    bool InvariantPrinter<bool>::_neverZeroOne(int varA, int varB, ResultCollector<bool> & results) {
        return _neverOneZero(varB, varA, results);
    }


    int InvariantPrinter<bool>::_getImplicTernaryInvariantWithLeftConstant (int left, ResultCollector<bool> & results,
                                                                            std::ostream & out) {
        // number of ternary invaraints
        int number_ternaryInvariants = 0;

        // iterator for the variable right_1
        for (int right_1 = 0; right_1 < _vars; ++right_1)
        {
            if (right_1 == left) continue;

            // iterator for the variable right_2
            for (int right_2 = right_1 + 1; right_2 < _vars; ++right_2)
            {
                if (right_2 == left) continue;

                result_t result = 0;

                // left = 1
                if (_isConstatTrue(left, results))
                {
                    //// find AND ////
                    if (_isConstatTrue(right_1, results) &&
                        _isConstatTrue(right_2, results)) result |= AND_Idx;

                    //// find IMPLY //// OK
                    if (_neverOneZero(right_1, right_2, results)) result |= IMPLY_Idx;

                    //// find R_IMPLY //// OK
                    if (_neverZeroOne(right_1, right_2, results)) result |= R_IMPLY_Idx;

                    //// find XOR //// OK
                    if (_areNotEqual(right_1, right_2, results)) result |= XOR_Idx;

                    //// find OR ////
                    if (_neverBothZero(right_1, right_2, results)) result |= OR_Idx;

                    //// find NOR //// OK
                    if (_isConstatFalse(right_1, results) &&
                        _isConstatFalse(right_2, results)) result |= NOR_Idx;

                    //// find XNOR //// OK
                    if (_areEqual(right_1, right_2, results)) result |= XNOR_Idx;

                    //// find NOT_R_IMPLY //// OK
                    if (_isConstatFalse(right_1, results) &&
                        _isConstatTrue(right_2, results)) result |= NOT_R_IMPLY_Idx;

                    //// find NOT_IMPLY //// OK
                    if (_isConstatTrue(right_1, results) &&
                        _isConstatFalse(right_2, results)) result |= NOT_IMPLY_Idx;

                    //// find NAND //// OK
                    if (_neverBothOne(right_1, right_2, results)) result |= NAND_Idx;
                }
                // left = 0
                else
                {
                    //// find AND //// OK
                    if (_neverBothOne(right_1, right_2, results)) result |= AND_Idx;

                    //// find IMPLY //// OK
                    if (_isConstatTrue(right_1, results) &&
                        _isConstatFalse(right_2, results)) result |= IMPLY_Idx;

                    //// find R_IMPLY //// OK
                    if (_isConstatFalse(right_1, results) &&
                        _isConstatTrue(right_2, results)) result |= R_IMPLY_Idx;

                    //// find XOR //// OK
                    if (_areEqual(right_1, right_2, results)) result |= XOR_Idx;

                    //// find OR //// OK
                    if (_isConstatFalse(right_1, results) &&
                        _isConstatFalse(right_2, results)) result |= OR_Idx;

                    //// find NOR //// OK
                    if (_neverBothZero(right_1, right_2, results)) result |= NOR_Idx;

                    //// find XNOR //// OK
                    if (_areNotEqual(right_1, right_2, results)) result |= XNOR_Idx;

                    //// find NOT_R_IMPLY ////
                    if (_neverZeroOne(right_1, right_2, results)) result |= NOT_R_IMPLY_Idx;

                    //// find NOT_IMPLY //// OK
                    if (_neverOneZero(right_1, right_2, results)) result |= NOT_IMPLY_Idx;

                    //// find NAND ////
                    if (_isConstatTrue(right_1, results) &&
                        _isConstatTrue(right_2, results)) result |= NAND_Idx;
                }

                if (result != 0) {
                    out << _printTernaryBoolInvariant(left, right_1, right_2, result) << std::endl;
                    number_ternaryInvariants += __builtin_popcount(result);
                }
            }
        }

        return number_ternaryInvariants;
    }

    int InvariantPrinter<bool>::_getImplicTernaryInvariantWithRight1Constant (int left, int right_1, ResultCollector<bool> & results,
                                                                              std::ostream & out)
    {
        // number of ternary invaraints
        int number_ternaryInvariants = 0;

        // iterator for the variable right_2
        for (int right_2 = right_1 + 1; right_2 < _vars; ++right_2) {
            if (right_2 == left) continue;

            result_t result = 0;

            // right_1 = 1
            if (_isConstatTrue(right_1, results)) {
                //// find AND | IMPLY | XNOR | ////
                if (_areEqual(left, right_2, results)) result |= AND_Idx | IMPLY_Idx | XNOR_Idx;

                //// find XOR | NOT_IMPLY | NAND ////
                if (_areNotEqual(left, right_2, results)) result |= XOR_Idx | NOT_IMPLY_Idx | NAND_Idx;

                //// find R_IMPLY | OR ////
                //// already counted with left = 1 ////

                //// find NOR | NOT_R_IMPLY//// OK
                //// already counted with left = 0 ///
            }
            // right_1 = 0
            else {
                //// find XOR | OR | NOT_R_IMPLY ////
                if (_areEqual(left, right_2, results)) result |= XOR_Idx | OR_Idx | NOT_R_IMPLY_Idx;

                //// find R_IMPLY | NOR | XNOR ////
                if (_areNotEqual(left, right_2, results)) result |= R_IMPLY_Idx | NOR_Idx | XNOR_Idx;

                //// find IMPLY | NAND ////
                //// already counted with left = 1 ////

                //// find NOT_IMPLY | AND//// OK
                //// already counted with left = 0 ////
            }

            if (result != 0) {
              out << _printTernaryBoolInvariant(left, right_1, right_2, result) << std::endl;
              number_ternaryInvariants += __builtin_popcount(result);
          }
        }

        return number_ternaryInvariants;
    }


    int InvariantPrinter<bool>::_getImplicTernaryInvariantWithRight2Constant (int left, int right_1, int right_2,
                                                                              ResultCollector<bool> & results,
                                                                              std::ostream & out) {
        result_t result = 0;

        // right_2 = 1
        if (_isConstatTrue(right_2, results)) {
            //// find AND | R_IMPLY | XNOR | ////
            if (_areEqual(left, right_1, results)) result |= AND_Idx | R_IMPLY_Idx | XNOR_Idx;

            //// find XOR | NOT_R_IMPLY | NAND ////
            if (_areNotEqual(left, right_1, results)) result |= XOR_Idx | NOT_R_IMPLY_Idx | NAND_Idx;

            //// find IMPLY | OR ////
            //// already counted with left = 1 ///

            //// find NOR | NOT_IMPLY ////
            //// already counted with left = 0 ///
        }
        // right_2 = 0
        else {
            //// find XOR | OR | NOT_IMPLY | ////
            if (_areEqual(left, right_1, results)) result |= XOR_Idx | OR_Idx | NOT_IMPLY_Idx;

            //// find IMPLY | NOR | XNOR ////
            if (_areNotEqual(left, right_1, results)) result |= IMPLY_Idx | NOR_Idx | XNOR_Idx;

            //// find R_IMPLY | NAND ////
            //// already counted with left = 1 ////

            //// find AND | NOT_R_IMPLY | ////
            //// already counted with left = 0 ////
        }

        if (result != 0)
          out << _printTernaryBoolInvariant(left, right_1, right_2, result) << std::endl;

        return __builtin_popcount(result);
    }


    int InvariantPrinter<bool>::_getImplicTernaryInvariantWithRight1And2EqualOrNotEqual (int left, int right_1, int right_2,
                                                                                         ResultCollector<bool> & results, std::ostream & out)
    {
        result_t result = 0;

        // right_1 = right_2
        if (_areEqual(right_1, right_2, results)) {
            //// find AND | OR ////
            if (_areEqual(left, right_1, results)) result = AND_Idx | OR_Idx;

            //// find NAND | NOR ////
            if (_areNotEqual(left, right_1, results)) result |= NOR_Idx | NAND_Idx;

            //// IMPLY | R_IMPLY | XNOR  ////
            //// already counted with left = 1 ////

            //// find XOR | NOT_R_IMPLY | NOT_IMPLY ////
            //// already counted with left = 0 ////
        }
        // right_1 != right_2
        else {
            //// find IMPLY | NOT_R_IMPLY ////
            if (_areEqual(left, right_2, results)) result = IMPLY_Idx | NOT_R_IMPLY_Idx;

            //// find R_IMPLY | NOT_IMPLY ////
            if (_areEqual(left, right_1, results)) result |= R_IMPLY_Idx | NOT_IMPLY_Idx;

            //// find XOR | OR | NAND ////
            //// already counted with left = 1 ////

            //// find AND |  NOR | XNOR ////
            //// already counted with left = 0 ////
        }

        if (result !=0)
          out << _printTernaryBoolInvariant(left, right_1, right_2, result) << std::endl;

        return __builtin_popcount(result);
    }

    // /////////////////////////////////////////////////////////////////////////////////
    // specialization of the class Generator for the BOOLEAN variables - Public methods
    // /////////////////////////////////////////////////////////////////////////////////
    int InvariantPrinter<bool>::unaryInvariants(ResultCollector<bool> & results, std::ostream & out)
    {
        // number of unary invaraints
        int number_unaryInvariants = 0;

        // iterator for the variable
        for (int var_index = 0; var_index < _vars; ++var_index) {
            // is it a constant ?
            if (_isConstat(var_index, results)) {
              out << _printUnaryBoolInvariant(var_index, _isConstatTrue(var_index, results) ) << std::endl;
              ++number_unaryInvariants;
            }
        }

        return number_unaryInvariants;
    }

#ifndef BOOL_INFERENCE

    int InvariantPrinter<bool>::binaryInvariants(ResultCollector<bool> & results, std::ostream & out) {
        // number of binary invaraints
        int number_binaryInvariants = 0;

        // iterator for the variable var_1
        for (int first = 0; first < _vars; ++first) {
            //iterator for the variable var_2
            for (int second = first + 1; second < _vars; ++second) {

                if ( _areEqual(first, second, results) ) {
                    out << _printBinaryBoolInvariant(first, second, true) << std::endl;
                    ++number_binaryInvariants;
                }
                else if ( _areNotEqual(first, second, results) ) {
                    out << _printBinaryBoolInvariant(first, second, false) << std::endl;
                    ++number_binaryInvariants;
                }
            }
        }

        return number_binaryInvariants;
    }

    int InvariantPrinter<bool>::ternaryInvariants(ResultCollector<bool> & results, std::ostream & out) {
        result_t * ternaryResult = static_cast<result_t *> (results.getVectorResult<3>());

        // number of ternary invaraints
        int number_ternaryInvariants = 0;

        // index of the result array
        int index = 0;

        // iterator for the variable left
        for (int left = 0; left < _vars; ++left) {
            // iterator for the variable right_1
            for (int right_1 = 0; right_1 < _vars; ++right_1) {
                if (right_1 == left) continue;

                // iterator for the variable right_2
                for (int right_2 = right_1 + 1; right_2 < _vars; ++right_2) {
                    if (right_2 == left) continue;

                    result_t result = ternaryResult[index++];

                    if (result != 0) {
                        number_ternaryInvariants += __builtin_popcount(result);
                        out << _printTernaryBoolInvariant(left, right_1, right_2, result) << std::endl;
                    }
                }
            }
        }

        return number_ternaryInvariants;
    }

#else

    int InvariantPrinter<bool>::binaryInvariants(ResultCollector<bool> & results, std::ostream & out) {
        // number of binary invaraints
        int number_binaryInvariants = 0;

        // iterator for the variable var_1
        for (int first = 0; first < _vars; ++first) {
            // is it a constant ?
            if (_isConstat(first, results)) {
                #ifdef GET_ALL_INVARIANTS
                    // iterator for the variable var_2
                    for (int second = first + 1; second < _vars; ++second) {
                        // 1. All the other constants equal to the first variable.
                        // 2. All the other constants different from the first variable.
                        if (!_isConstat(second, results)) continue;

                        out << _printBinaryBoolInvariant(first, second, _areEqual(first, second, results)) << std::endl;
                        ++number_binaryInvariants;
                    }
                #endif

                continue;
            }

            //iterator for the variable var_2
            for (int second = first + 1; second < _vars; ++second) {
                if ( _areEqual(first, second, results) ) {
                    out << _printBinaryBoolInvariant(first, second, true) << std::endl;
                    ++number_binaryInvariants;
                }
                else if ( _areNotEqual(first, second, results) ) {
                    out << _printBinaryBoolInvariant(first, second, false) << std::endl;
                    ++number_binaryInvariants;
                }
            }
        }

        return number_binaryInvariants;
    }

    int InvariantPrinter<bool>::ternaryInvariants(ResultCollector<bool> & results, std::ostream & out)
    {
        bitCounter_t * unaryResult = static_cast<bitCounter_t *> (results.getVectorResult<1>());
        bitCounter_t * binaryResult = static_cast<bitCounter_t *> (results.getVectorResult<2>());
        result_t * ternaryResult = static_cast<result_t *> (results.getVectorResult<3>());

        // number of ternary invaraints
        int number_ternaryInvariants = 0;

        // index of the result array
        int index = 0;

        // iterator for the variable left
        for (int left = 0; left < _vars; ++left) {
            // We have a constant variable
            if (_isConstat(left, results)) {
                #ifdef GET_ALL_INVARIANTS
                    number_ternaryInvariants +=
                    _getImplicTernaryInvariantWithLeftConstant(left, results, out);
                #endif

                continue;
            }

            // iterator for the variable right_1
            for (int right_1 = 0; right_1 < _vars; ++right_1) {
                if (right_1 == left) continue;

                // is it a constant ?
                if (_isConstat(right_1, results)) {
                    #ifdef GET_ALL_INVARIANTS
                        number_ternaryInvariants +=
                        _getImplicTernaryInvariantWithRight1Constant(left, right_1, results, out);
                    #endif

                    continue;
                }

                // iterator for the variable right_2
                for (int right_2 = right_1 + 1; right_2 < _vars; ++right_2) {
                    if (right_2 == left) continue;

                    // is it a constant
                    if (_isConstat(right_2, results)) {
                        #ifdef GET_ALL_INVARIANTS
                            number_ternaryInvariants +=
                            _getImplicTernaryInvariantWithRight2Constant(left, right_1, right_2, results, out);
                        #endif

                        continue;
                    }

                    // we have either != or = between right_1 and right_2
                    if (_areEqual(right_1, right_2, results) ||
                        _areNotEqual(right_1, right_2, results)) {
                        #ifdef GET_ALL_INVARIANTS
                            number_ternaryInvariants +=
                            _getImplicTernaryInvariantWithRight1And2EqualOrNotEqual(left, right_1, right_2, results, out);
                        #endif

                        continue;
                    }

                    int rigth12_index = results.associativeArrayForBinaryResult[ results.getIndexFromTriplet2Pair(right_1, right_2)];

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

                    result_t result = ternaryResult[index++];
                    if (result != 0) {
                        number_ternaryInvariants += __builtin_popcount(result);
                        out << _printTernaryBoolInvariant(left, right_1, right_2, result) << std::endl;
                    }
                }
            }
        }

        return number_ternaryInvariants;
    }
#endif
}
