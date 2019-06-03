#pragma once

#include "Inference/ResultCollector.hpp"
#include <chrono>
#include <random>
#include <iostream>

namespace mangrove {

    template <int F_INDEX>
    struct TEST_TERNARY_INVARIANT {
        inline static void ApplyT (numeric_t x1, numeric_t x2_1, numeric_t x2_2,
                                   int *monotony, int *commutative) {

            using invariant = typename GET_INVARIANT<F_INDEX-1, NTtemplate>::type;

            if (monotony[F_INDEX-1] != MonotonyProperty::NOMONOTONY) {
                if (x2_2 < x2_1) std::swap(x2_1, x2_2);

                numeric_t y1 = APPLY_FUN<1, invariant>::eval(x1, x2_1);
                numeric_t y2 = APPLY_FUN<1, invariant>::eval(x1, x2_2);

                if (std::isnan(y1) || std::isnan(y2))
                    __ERROR("F: " << F_INDEX << " unknown for num: " << x1);

                if (y1 < y2)
                    monotony[F_INDEX-1] = (monotony[F_INDEX-1] == MonotonyProperty::INCREASING ||
                                           monotony[F_INDEX-1] == MonotonyProperty::UNKNOWN)?
                                                                  MonotonyProperty::INCREASING :
                                                                  MonotonyProperty::NOMONOTONY;
                else if (y1 > y2)
                    monotony[F_INDEX-1] = (monotony[F_INDEX-1] == MonotonyProperty::DECREASING ||
                                           monotony[F_INDEX-1] == MonotonyProperty::UNKNOWN)?
                                                                  MonotonyProperty::DECREASING :
                                                                  MonotonyProperty::NOMONOTONY;
            }

            if (commutative[F_INDEX-1] != CommutativeProperty::NO) {

                numeric_t y1   = APPLY_FUN<1, invariant>::eval(x1, x2_1);
                numeric_t y1_r = APPLY_FUN<1, invariant>::eval(x2_1, x1);

                numeric_t y2   = APPLY_FUN<1, invariant>::eval(x1, x2_2);
                numeric_t y2_r = APPLY_FUN<1, invariant>::eval(x2_2, x1);

                if ((y1 == y1_r) && (y2 == y2_r))
                    commutative[F_INDEX-1] = CommutativeProperty::YES;
                else
                    commutative[F_INDEX-1] = CommutativeProperty::NO;
            }

            TEST_TERNARY_INVARIANT<F_INDEX - 1>::ApplyT(x1, x2_1, x2_2, monotony, commutative);
        }
    };

    template <>
    struct TEST_TERNARY_INVARIANT<0> {
        inline static void ApplyT (numeric_t, numeric_t, numeric_t, int *, int *) {}
    };

    template <int F_INDEX>
    void CheckTernaryMonotony(ResultCollector<numeric_t> & results, int variables) {

        const int ITERATION_NUMBER = 20;

        num2_t *unaryResult = static_cast<num2_t *> (results.getVectorResult<1>());

        // init the monotony and commutative array
        int* monotony = results.getTernaryMonotony();
        int* commutative = results.getTernaryCommutative();
        for (int i = 0; i < F_INDEX; ++i)  {
            monotony[i] = MonotonyProperty::UNKNOWN;
            commutative[i] = CommutativeProperty::UNKNOWNC;
        }

        unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
        std::default_random_engine generator (seed);

        for (int var_1 = 0; var_1 < variables; ++var_1) {
            std::uniform_real_distribution<numeric_t> distribution_1(unaryResult[var_1][0],
                                                                     unaryResult[var_1][1]);

            for (int var_1_iter = 0; var_1_iter < ITERATION_NUMBER; ++var_1_iter) {
                numeric_t x1 = distribution_1(generator);

                for (int var_2 = var_1 + 1; var_2 < variables; ++var_2) {
                   std::uniform_real_distribution<numeric_t> distribution_2(unaryResult[var_2][0],
                                                                            unaryResult[var_2][1]);

                   for (int var_2_iter = 0; var_2_iter < ITERATION_NUMBER; ++var_2_iter) {
                       numeric_t x2_2 = distribution_2(generator);
                       numeric_t x2_1 = distribution_2(generator);

                       TEST_TERNARY_INVARIANT<F_INDEX>::ApplyT(x1, x2_1, x2_2, monotony, commutative);
                   }
               }
           }
       }
    }
} //@mangrove
