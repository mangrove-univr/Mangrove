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

#include "config.cuh"
#include "DataTypes/ModuleSTR.hpp"
#include "Inference/ResultCollector.hpp"
#include "Mining/TemplateEngine.cuh"

namespace mangrove
{
    template <typename T>
    class InvariantPrinterGuide
    {
      public:

        InvariantPrinterGuide(const InvariantPrinterGuide& e) = delete;
        InvariantPrinterGuide& operator =(const InvariantPrinterGuide& e) = delete;

        /// @brief Default destructor.
        virtual ~InvariantPrinterGuide();

        virtual int unaryInvariants   (ResultCollector<T> & results, std::ostream & out) = 0;
        virtual int binaryInvariants  (ResultCollector<T> & results, std::ostream & out) = 0;
        virtual int ternaryInvariants (ResultCollector<T> & results, std::ostream & out) = 0;

      protected:

        /// @brief Default costructor.
        InvariantPrinterGuide(const MiningParamStr &variables, int vars);

        // This variable stores the number of variables of a trace
        int _vars;

        // array of the variables' name
        MiningParamStr _variables;
    };


    template <typename T>
    class InvariantPrinter;


    template <>
    class InvariantPrinter<bool> : public InvariantPrinterGuide<bool>
    {
      public:

        /// @brief Default costructor.
        InvariantPrinter(const MiningParamStr &variables, int vars, int traceLength);

        InvariantPrinter(const InvariantPrinter& e) = delete;
        InvariantPrinter& operator =(const InvariantPrinter& e) = delete;

        /// @brief Default destructor.
        ~InvariantPrinter();

        int unaryInvariants   (ResultCollector<bool> & results, std::ostream & out);
        int binaryInvariants  (ResultCollector<bool> & results, std::ostream & out);
        int ternaryInvariants (ResultCollector<bool> & results, std::ostream & out);

      private:

        // This variable stores the length of a trace
        int _traceLength;

        std::string _printUnaryBoolInvariant(int indexVar, bool result);
        std::string _printBinaryBoolInvariant(int indexVarA, int indexVarB, bool equal);
        std::string _printTernaryBoolInvariant(int indexVarLeft, int indexVarRight1, int indexVarRight2, result_t result);

        bool _isConstatTrue(int varA, ResultCollector<bool> & results);
        bool _isConstatFalse(int varA, ResultCollector<bool> & results);
        bool _isConstat(int varA, ResultCollector<bool> & results);
        bool _areEqual(int varA, int varB, ResultCollector<bool> & results);
        bool _areNotEqual(int varA, int varB, ResultCollector<bool> & results);
        bool _neverBothZero(int varA, int varB, ResultCollector<bool> & results);
        bool _neverBothOne(int varA, int varB, ResultCollector<bool> & results);
        bool _neverOneZero(int varA, int varB, ResultCollector<bool> & results);
        bool _neverZeroOne(int varA, int varB, ResultCollector<bool> & results);

        int _getImplicTernaryInvariantWithLeftConstant (int left, ResultCollector<bool> & results, std::ostream & out);

        int _getImplicTernaryInvariantWithRight1Constant (int left, int right_1,
                                                          ResultCollector<bool> & results, std::ostream & out);

        int _getImplicTernaryInvariantWithRight2Constant (int left, int right_1, int right2,
                                                          ResultCollector<bool> & results, std::ostream & out);

        int _getImplicTernaryInvariantWithRight1And2EqualOrNotEqual (int left, int right_1, int right_2,
                                                                     ResultCollector<bool> & results, std::ostream & out);
    };

    template <>
    class InvariantPrinter<numeric_t> : public InvariantPrinterGuide<numeric_t>
    {
      public:

          /// @brief Default costructor.
          InvariantPrinter(const MiningParamStr& variables, int vars);

          InvariantPrinter(const InvariantPrinter& e) = delete;
          InvariantPrinter& operator =(const InvariantPrinter& e) = delete;

          /// @brief Default destructor.
          ~InvariantPrinter();

          int unaryInvariants   (ResultCollector<numeric_t> & results, std::ostream & out);
          int binaryInvariants  (ResultCollector<numeric_t> & results, std::ostream & out);
          int ternaryInvariants (ResultCollector<numeric_t> & results, std::ostream & out);

      private:

          bool _isConstat     (int varA, ResultCollector<numeric_t> & results);
          std::string _printUnaryNumericInvariant(int indexVar, num2_t *unaryResult);
          std::string _printBinaryNumericInvariant(int indexVarA, int indexVarB, result_t result);
          std::string _printTernaryNumericInvariant(int indexVarLeft, int indexVarRight1, int indexVarRight2, result_t result);
    };
}

#include "InvariantPrinter.i.hpp"
