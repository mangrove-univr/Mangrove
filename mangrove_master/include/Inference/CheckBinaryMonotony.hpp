#pragma once

#include "Inference/ResultCollector.hpp"
#include <chrono>
#include <random>
#include <iostream>

namespace mangrove {

    // Monotony test
    template <int F_INDEX>
    struct TEST_BINARY_INVARIANT {
        inline static void ApplyT (numeric_t x1, numeric_t x2, int *monotony) {

            if (monotony[F_INDEX] != MonotonyProperty::NOMONOTONY) {
                if (x2 < x1) std::swap(x1, x2);

                using T = typename GET_INVARIANT<F_INDEX, NBtemplate>::type::unary;
                numeric_t y1 = T()(x1);
                numeric_t y2 = T()(x2);

                if (std::isnan(y1) || std::isnan(y2))
                    __ERROR("F: " << F_INDEX << " unknown for num: " << x1);

                if (y1 < y2)
                    monotony[F_INDEX] = (monotony[F_INDEX] == MonotonyProperty::INCREASING ||
                                         monotony[F_INDEX] == MonotonyProperty::UNKNOWN)?
                                                                  MonotonyProperty::INCREASING :
                                                                  MonotonyProperty::NOMONOTONY;
                else if (y1 > y2)
                    monotony[F_INDEX] = (monotony[F_INDEX] == MonotonyProperty::DECREASING ||
                                         monotony[F_INDEX] == MonotonyProperty::UNKNOWN)?
                                                                  MonotonyProperty::DECREASING :
                                                                  MonotonyProperty::NOMONOTONY;
            }

            TEST_BINARY_INVARIANT<F_INDEX - 1>::ApplyT(x1, x2, monotony);
        }
    };

    template <>
    struct TEST_BINARY_INVARIANT<3> { inline static void ApplyT (numeric_t, numeric_t, int *) {} };

    template <int F_INDEX>
    void CheckBinaryMonotony(ResultCollector<numeric_t> & results, int variables) {
        const int ITERATION_NUMBER = 100;
        num2_t *unaryResult = static_cast<num2_t *> (results.getVectorResult<1>());

        // init the monotony array
        int * monotony = results.getMonotony();
        for (int i = 0; i < 4; ++i) monotony[i] = MonotonyProperty::INCREASING;
        for (int i = 4; i < F_INDEX; ++i) monotony[i] = MonotonyProperty::UNKNOWN;

        unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
        std::default_random_engine generator (seed);

        for (int var = 0; var < variables; ++var) {
            std::uniform_real_distribution<numeric_t> distribution(unaryResult[var][0],
                                                                   unaryResult[var][1]);
            for(int iter = 0; iter < ITERATION_NUMBER; ++iter) {
                // get 2 random values in the variable's range and test monotony
                numeric_t x1 = distribution(generator);
                numeric_t x2 = distribution(generator);

                TEST_BINARY_INVARIANT<F_INDEX - 1>::ApplyT(x1, x2, monotony);
            }
        }
    }
} //@mangrove
