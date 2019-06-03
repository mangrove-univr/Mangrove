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
#include "TemplateConf.cuh"
#include "DataTypes/TraceProp.cuh"

namespace mangrove
{
    /// @brief This class implements a wrapper storing the elaborated results
    template <typename T>
    class ResultCollectorGuide
    {
    public:
        /// @brief Default costructor.
        ResultCollectorGuide(int variables);

        /// @brief Copy costructor (Disabled).
        ResultCollectorGuide(const ResultCollectorGuide& e) = delete;

        /// @brief Operator = (Disabled).
        ResultCollectorGuide& operator =(const ResultCollectorGuide& e) = delete;

        /// @brief Default destructor.
        virtual
        ~ResultCollectorGuide();

        /// @brief implement the Mapping function
        int getIndexFromTriplet2Pair(int i, int j);

      protected:
        /// @brief Number of variables
        const int _vars;
    };

    template <typename T1>
    class ResultCollector;

    /// @brief This class implements a resultCollector of boolean variables.
    template <>
    class ResultCollector<bool> : public ResultCollectorGuide<bool>
    {
      public:
        /// @brief Default costructor.
        ResultCollector(int variables);

        /// @brief Copy costructor (Disabled).
        ResultCollector(const ResultCollector& e) = delete;

        /// @brief Operator = (Disabled).
        ResultCollector& operator =(const ResultCollector& e) = delete;

        /// @brief Mapping a pair of variable to a 3-arity result
        int associativeArrayForBinaryResult[DictionarySize<bool, MAX_VARS, 2>::value] = {};

        /// @brief Default destructor.
        ~ResultCollector();

        template <int ARITY>
        void * getVectorResult();

      private:
        // @brief unary result
        bitCounter_t _unaryResults[MAX_VARS];

        // @brief binary result
        bitCounter_t _binaryResults[DictionarySize<bool, MAX_VARS, 2>::value];

        /// @brief ternary result
        result_t* _ternaryResults;
    };

    //
    static const int halfNumericResult = DictionarySize<numeric_t, MAX_VARS, 2>::value / 2;

    /// @brief This class implements a resultCollector of float variables.
    template <>
    class ResultCollector<numeric_t> : public ResultCollectorGuide<numeric_t>
    {
      public:
        /// @brief Default costructor.
        ResultCollector(int variables);

        /// @brief Copy costructor (Disabled).
        ResultCollector(const ResultCollector& e) = delete;

        /// @brief Operator = (Disabled).
        ResultCollector& operator =(const ResultCollector& e) = delete;

        /// @brief Mapping a pair of variable to a 3-arity result
        int associativeArrayForBinaryResult[DictionarySize<numeric_t, MAX_VARS, 2>::value] = {};

        int *ternaryAssociativeArray;

        int getTAindex(int left, int right_1, int right_2);

        /// @brief Default destructor.
        ~ResultCollector();

        // @brief depending on the provided arity, it returns unary|binary|ternary vector result
        template <int ARITY> void * getVectorResult();

        // @brief returns the monotony vector describing the monotony property of
        //  each binary user-defined function
        int * getMonotony();

        // @brief returns the monotony vector describing the monotony property of
        //  each ternary user-defined function
        int * getTernaryMonotony();

        // @brief returns the commutative vector describing the commutative property of
        //  each ternary user-defined function
        int * getTernaryCommutative();

        // @brief returns the equivalence set
        int * getEquivalenceSets();

        // @brief sets formwards or backwards mining direction
        // direction equals at true means formwards direction
        void setForwardDirection(bool direction);

      private:
        // For Numeric data type we have the minimum and the maximum of each variable
        numeric_t _unaryResults[MAX_VARS * 2] = {};

        // @brief result
        result_t _binaryResults[DictionarySize<numeric_t, MAX_VARS, 2>::value] = {}; // forward and backward

        /// @brief ternary result
        result_t* _ternaryResults;

        // disjoint-set forests. Each var-index points to its father.
        int _equivalence_sets[MAX_VARS] = {};

        // if it is true, then we are performing invariant mining formwards,
        // otherwise invariant mining backwards.
        bool _forwardDirection;

        // This variable stores the monotony of each user-defined binary function
        int _monotony[NBtemplate::size] = {}; // positive(1), negative(-1), no monotony (0)

        // This variable stores the monotony of each user-defined ternary function
        int _ternaryMonotony[NTtemplate::size] = {}; // positive(1), negative(-1), no monotony (0)

        // This variable stores the commutative property of each user-defined ternary function
        int _ternaryCommutative[NTtemplate::size] = {}; // yes(1), no(0)
    };
}

#include "impl/ResultCollector.i.hpp"
