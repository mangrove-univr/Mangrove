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
#include "ResultCollector.hpp"
#include "Mining/TemplateEngine.cuh"

namespace mangrove
{
    template <typename T>
    class GeneratorGuide
    {
      public:
        /// @brief The dictionary containing pairs or triplets of variables
        entry_t* dictionary;

        /// @brief Default costructor.
        /// @param vars The number of variables in the trace
        /// @param traceLength The length of the trace
        GeneratorGuide(int vars);

        /// @brief Copy costructor. (Unsupported)
        GeneratorGuide(const GeneratorGuide& e) = delete;

        /// @brief Default destructor.
        virtual ~GeneratorGuide();

        /// @brief Assign operator. (Unsupported)
        GeneratorGuide& operator =(const GeneratorGuide& e) = delete;

      protected:
        // This variable stores the number of variables of a trace
        const int _vars;
    };


    template <typename T>
    class Generator;

    /// @brief This class implements a generator of pairs and triplets of boolean variables.
    template <>
    class Generator<bool> : public GeneratorGuide<bool>
    {
      public:
        /// @brief Default costructor.
        /// @param vars The number of variables in the trace
        /// @param traceLength The length of the trace
        Generator(int vars, int traceLength);

        /// @brief Copy costructor. (Unsupported)
        Generator(const Generator& e) = delete;

        /// @brief Default destructor.
        ~Generator();

        /// @brief This method generates the dictionary's entries.
        /// @param results The elaborated results in the previous steps.
        /// @return The number of entries generated in the dictionary.
        template<int ARITY>
        int generator(ResultCollector<bool> & results);

        /// @brief Assign operator. (Unsupported)
        Generator& operator =(const Generator& e) = delete;

      private:
        // This variable stores the length of a trace
        const int _traceLength;
    };

    //
    static const int halfNumericDictionary = DictionarySize<numeric_t, MAX_VARS, 2>::value;

    /// @brief This class implements a generator of pairs and triplets of boolean variables.
    template <>
    class Generator<numeric_t> : public GeneratorGuide<numeric_t>
    {

      public:
        /// @brief Default costructor.
        /// @param vars The number of variables in the trace
        Generator(int vars);

        /// @brief Copy costructor. (Unsupported)
        Generator(const Generator& e) = delete;

        /// @brief Default destructor.
        ~Generator();

        /// @brief This method generates the dictionary's entries.
        /// @param results The elaborated results in the previous steps.
        /// @return The number of entries generated in the dictionary.
        template<int ARITY>
        int generator(ResultCollector<numeric_t> & results);

        /// @brief Assign operator. (Unsupported)
        Generator& operator =(const Generator& e) = delete;

        void setForwardDirection(bool direction);

      private:

        bool _forwardDirection;

        bool _rangeOverlapping(ResultCollector<numeric_t> & results, int first, int second);
    };
}

#include "impl/Generator.i.hpp"
